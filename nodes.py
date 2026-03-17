import torch
import torch.nn.functional as F
import numpy as np
import math

from nodes import common_ksampler

# Import required modules
import comfy.sample
import comfy.samplers
import comfy.utils
# Prepare callback for progress
import latent_preview


NODE_CLASS_MAPPINGS = {}
NODE_DISPLAY_NAME_MAPPINGS = {}


def create_gaussian_filter(height, width, sigma, device):
    """Create Gaussian filter in frequency domain for FFT split"""
    # Create frequency grids
    u = torch.linspace(-0.5, 0.5, height, device=device)
    v = torch.linspace(-0.5, 0.5, width, device=device)
    U, V = torch.meshgrid(u, v, indexing='ij')
    
    # Calculate squared distance from center
    D2 = U**2 + V**2
    
    # Gaussian filter formula: exp(-2 * pi^2 * sigma^2 * D2)
    pi = math.pi
    gaussian = torch.exp(-2 * (pi**2) * (sigma**2) * D2)
    
    return gaussian

def apply_spatial_gaussian_blur(tensor, sigma):
    """
    Applies a spatial Gaussian blur. Much more accurate and artifact-free 
    for small blur radii compared to FFT (avoids edge wrap-around).
    """
    if sigma <= 0.0:
        return tensor
        
    device = tensor.device
    channels = tensor.shape[1]
    
    # Calculate kernel size based on sigma (3 sigma rule)
    k_size = int(2 * round(3.0 * sigma) + 1)
    if k_size < 3:
        k_size = 3
        
    # Create 1D Gaussian kernel
    x = torch.arange(k_size, dtype=torch.float32, device=device) - k_size // 2
    kernel_1d = torch.exp(-0.5 * (x / sigma) ** 2)
    kernel_1d = kernel_1d / kernel_1d.sum()
    
    # Create 2D kernel
    kernel_2d = kernel_1d.view(1, 1, -1, 1) * kernel_1d.view(1, 1, 1, -1)
    
    # Expand kernel for all channels (depthwise convolution)
    kernel_2d = kernel_2d.expand(channels, 1, k_size, k_size)
    
    # Pad and convolve
    pad = k_size // 2
    padded = F.pad(tensor, (pad, pad, pad, pad), mode='reflect')
    blurred = F.conv2d(padded, kernel_2d, groups=channels)
    
    return blurred


def HighFrequencyEnhancer(latent, high_freq_mult, sigma, denoise_threshold, mask_hardness, hf_pre_blur_sigma):
    # 1. Extract the latent samples tensor
    samples = latent["samples"].clone()
    batch_size, channels, height, width = samples.shape[:4]
    
    # Handle WAN format if needed
    is_wan = False
    if samples.ndim == 5:
        samples = samples.squeeze(2)
        is_wan = True
        batch_size, channels, height, width = samples.shape

    # --- Frequency Separation using FFT ---
    device = samples.device
    
    # Create Gaussian filter in frequency domain
    gaussian_filter = create_gaussian_filter(height, width, sigma, device)
    # Expand filter dimensions to match latent tensor [1, 1, H, W]
    gaussian_filter = gaussian_filter.view(1, 1, height, width)
    
    # Apply FFT to latent samples
    fft_latent = torch.fft.fft2(samples, dim=(-2, -1))
    fft_shifted = torch.fft.fftshift(fft_latent, dim=(-2, -1))
    
    # Apply low-pass filter
    low_freq_fft = fft_shifted * gaussian_filter
    
    # Inverse FFT to get low frequency component
    low_freq_shifted = torch.fft.ifftshift(low_freq_fft, dim=(-2, -1))
    low_freq = torch.fft.ifft2(low_freq_shifted, dim=(-2, -1)).real
    
    # High frequency is residual
    high_freq_original = samples - low_freq

    # --- Smart Mask Generation ---
    if denoise_threshold > 0:
        # Pre-blur for noise grouping if enabled (using optimized spatial blur)
        if hf_pre_blur_sigma > 0.0:
            high_freq_detection = apply_spatial_gaussian_blur(high_freq_original, hf_pre_blur_sigma)
        else:
            high_freq_detection = high_freq_original

        # Calculate magnitude and create soft gate mask
        magnitude = torch.abs(high_freq_detection)
        mask = torch.sigmoid((magnitude - denoise_threshold) * mask_hardness)
        
        # Apply mask and detail enhancement
        high_freq_final = high_freq_original * mask * high_freq_mult
        
    else:
        # Passthrough if no denoising requested
        mask = torch.ones_like(samples)
        high_freq_final = high_freq_original * high_freq_mult

    # --- Recombination ---
    enhanced_samples = low_freq + high_freq_final
    
    # --- Output Preparation ---
    # Prepare Latent Output
    enhanced_latent = latent.copy()
    if is_wan:
        enhanced_samples = enhanced_samples.unsqueeze(2)
    enhanced_latent["samples"] = enhanced_samples
    
    # Prepare Mask Preview Output
    # Calculate mean across channels to get grayscale intensity
    mask_preview = torch.mean(mask, dim=1, keepdim=True)  # [Batch, 1, H, W]
    mask_preview = mask_preview.repeat(1, 3, 1, 1)        # [Batch, 3, H, W]
    mask_preview = mask_preview.permute(0, 2, 3, 1)       # [Batch, H, W, 3]

    return (enhanced_latent, mask_preview)


class LatentFrequencyEnhancer_lrzjason:
    """
    ComfyUI Node for selective latent denoising and enhancement using FFT.
    """
    
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "latent": ("LATENT",),
                "high_freq_mult": ("FLOAT", {"default": 1.15, "min": 1.0, "max": 2.0, "step": 0.01, "label": "Detail Strength (HF Mult)"}),
                "sigma": ("FLOAT", {"default": 1.0, "min": 0.01, "max": 20.0, "step": 0.01, "label": "Frequency Split Sigma"}),
                "denoise_threshold": ("FLOAT", {"default": 0.05, "min": 0.0, "max": 1.0, "step": 0.001, "label": "Noise Threshold"}),
                "mask_hardness": ("FLOAT", {"default": 2.0, "min": 1.0, "max": 100.0, "step": 1.0, "label": "Mask Hardness (Transition)"}),
                "hf_pre_blur_sigma": ("FLOAT", {"default": 0.5, "min": 0.0, "max": 10.0, "step": 0.01, "label": "Noise Grouping (Pre-Blur)"}),
            },
        }

    RETURN_TYPES = ("LATENT", "IMAGE")
    RETURN_NAMES = ("enhanced_latent", "mask_preview")
    FUNCTION = "enhance"
    CATEGORY = "latent/enhancement"
    
    def enhance(self, latent, high_freq_mult, sigma, denoise_threshold, mask_hardness, hf_pre_blur_sigma):
        return HighFrequencyEnhancer(latent, high_freq_mult, sigma, denoise_threshold, mask_hardness, hf_pre_blur_sigma)
        

class LatentGaussianBlur_lrzjason:
    """
    ComfyUI Node to directly apply a Gaussian blur to a latent.
    """
    
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "latent": ("LATENT",),
                "sigma": ("FLOAT", {"default": 0.5, "min": 0.01, "max": 20.0, "step": 0.01, "label": "Blur Sigma"}),
                "edge_masking": ("BOOLEAN", {"default": False, "label": "Enable Edge Masking"}),
                "edge_threshold": ("FLOAT", {"default": 0.05, "min": 0.0, "max": 1.0, "step": 0.001, "label": "Edge Threshold"}),
                "edge_hardness": ("FLOAT", {"default": 20.0, "min": 1.0, "max": 100.0, "step": 1.0, "label": "Edge Mask Hardness"}),
            },
        }

    RETURN_TYPES = ("LATENT", "IMAGE")
    RETURN_NAMES = ("blurred_latent", "mask_preview")
    FUNCTION = "apply_blur"
    CATEGORY = "latent/enhancement"
    
    def apply_blur(self, latent, sigma, edge_masking, edge_threshold, edge_hardness):
        samples = latent["samples"].clone()
        
        # Handle WAN format if needed
        is_wan = False
        if samples.ndim == 5:
            samples = samples.squeeze(2)
            is_wan = True
            
        # 1. Apply Spatial Gaussian Blur (optimized for small, exact radii)
        blurred_samples = apply_spatial_gaussian_blur(samples, sigma)
        
        # 2. Apply Edge Masking if enabled
        if edge_masking:
            # Edges are represented by the high-frequency components
            high_freq = samples - blurred_samples
            magnitude = torch.abs(high_freq)
            
            # Create mask: approaching 1 near strong edges, 0 in flat areas
            mask = torch.sigmoid((magnitude - edge_threshold) * edge_hardness)
            
            # Apply blur proportionally (100% blurred where mask is strong)
            final_samples = samples * (1.0 - mask) + blurred_samples * mask
        else:
            final_samples = blurred_samples
            mask = torch.ones_like(samples)
            
        # 3. Prepare Latent Output
        blurred_latent = latent.copy()
        if is_wan:
            final_samples = final_samples.unsqueeze(2)
        blurred_latent["samples"] = final_samples
        
        # 4. Prepare Mask Preview Output
        mask_preview = torch.mean(mask, dim=1, keepdim=True)  # [B, 1, H, W]
        mask_preview = mask_preview.repeat(1, 3, 1, 1)        # [B, 3, H, W]
        mask_preview = mask_preview.permute(0, 2, 3, 1)       # [B, H, W, 3]
        
        return (blurred_latent, mask_preview)


class HFEPostProcessor:
    """
    Custom sampler with high-frequency enhancement during sampling process.
    """
    
    @classmethod
    def INPUT_TYPES(s):
        return {"required":
                    {
                        "model": ("MODEL",),
                        "noise_seed": ("INT", {"default": 0, "min": 0, "max": 0xffffffffffffffff, "control_after_generate": True}),
                        "steps": ("INT", {"default": 8, "min": 1, "max": 10000}),
                        "latent_image": ("LATENT", ),
                        "hfe_steps": ("INT", {"default": 2, "min": 1, "max": 100, "step": 1, "label": "Start HFE Step"}),
                        "cfg": ("FLOAT", {"default": 1.0, "min": 0.0, "max": 100.0, "step":0.1, "round": 0.01}),
                        "sampler_name": (comfy.samplers.KSampler.SAMPLERS, ),
                        "scheduler": (comfy.samplers.KSampler.SCHEDULERS, ),
                        "positive": ("CONDITIONING", ),
                        "negative": ("CONDITIONING", ),
                        "high_freq_mult": ("FLOAT", {"default": 1.05, "min": 1.0, "max": 2.0, "step": 0.01, "label": "Detail Strength (HF Mult)"}),
                        "sigma": ("FLOAT", {"default": 1.0, "min": 0.01, "max": 20.0, "step": 0.01, "label": "Frequency Split Sigma"}),
                        "denoise_threshold": ("FLOAT", {"default": 0.05, "min": 0.0, "max": 1.0, "step": 0.001, "label": "Noise Threshold"}),
                        "mask_hardness": ("FLOAT", {"default": 2.0, "min": 0.01, "max": 100.0, "step": 0.01, "label": "Mask Hardness (Transition)"}),
                        "hf_pre_blur_sigma": ("FLOAT", {"default": 0.5, "min": 0.0, "max": 10.0, "step": 0.01, "label": "Noise Grouping (Pre-Blur)"}),
                    }
                }

    RETURN_TYPES = ("LATENT", )
    FUNCTION = "sample"
    CATEGORY = "sampling"

    def sample(self, model, noise_seed, steps, latent_image, hfe_steps,
               cfg, sampler_name, scheduler, positive, negative, denoise=1.0,
               high_freq_mult=1.05, sigma=2, denoise_threshold=0.05, mask_hardness=2, hf_pre_blur_sigma=0.5):
        
        disable_noise = False
        latent = latent_image
        force_full_denoise = True
        
        start_hfe_step = steps
        steps = start_hfe_step + hfe_steps
        for i in range(start_hfe_step, steps):
            start_at_step = i
            end_at_step = i + 1
            if end_at_step >= steps:
                end_at_step = 10000

            latent = HighFrequencyEnhancer(latent, high_freq_mult, sigma, denoise_threshold, mask_hardness, hf_pre_blur_sigma)[0]
                                    
            latent = common_ksampler(model, noise_seed, steps, cfg, sampler_name, scheduler, positive, negative, latent, denoise=denoise, disable_noise=disable_noise, start_step=start_at_step, last_step=end_at_step, force_full_denoise=force_full_denoise)[0]    
            
        return (latent, )


# Register the nodes
NODE_CLASS_MAPPINGS["LatentFrequencyEnhancer_lrzjason"] = LatentFrequencyEnhancer_lrzjason
NODE_DISPLAY_NAME_MAPPINGS["LatentFrequencyEnhancer_lrzjason"] = "Latent Frequency Enhancer (lrzjason)"

NODE_CLASS_MAPPINGS["LatentGaussianBlur_lrzjason"] = LatentGaussianBlur_lrzjason
NODE_DISPLAY_NAME_MAPPINGS["LatentGaussianBlur_lrzjason"] = "Latent Gaussian Blur (lrzjason)"

NODE_CLASS_MAPPINGS["HFEPostProcessor (lrzjason)"] = HFEPostProcessor
NODE_DISPLAY_NAME_MAPPINGS["HFEPostProcessor (lrzjason)"] = "HFEPostProcessor (lrzjason)"
