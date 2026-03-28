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

def apply_spatial_gaussian_blur(tensor, kernel_size, sigma):
    """
    Applies a spatial Gaussian blur with a strictly bounded sampling range.
    """
    if sigma <= 0.0 or kernel_size < 3:
        return tensor
        
    # Ensure kernel size is odd
    if kernel_size % 2 == 0:
        kernel_size += 1
        
    device = tensor.device
    channels = tensor.shape[1]
    
    # Create 1D Gaussian kernel
    x = torch.arange(kernel_size, dtype=torch.float32, device=device) - kernel_size // 2
    kernel_1d = torch.exp(-0.5 * (x / sigma) ** 2)
    kernel_1d = kernel_1d / kernel_1d.sum()
    
    # Create 2D kernel
    kernel_2d = kernel_1d.view(1, 1, -1, 1) * kernel_1d.view(1, 1, 1, -1)
    
    # Expand kernel for depthwise convolution
    kernel_2d = kernel_2d.expand(channels, 1, kernel_size, kernel_size)
    
    # Pad and convolve using 'replicate' to avoid reflection edge artifacts
    pad = kernel_size // 2
    padded = F.pad(tensor, (pad, pad, pad, pad), mode='replicate')
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


def parse_target_channels(target_mode, custom_str, split_index, total_channels):
    """
    Parses the target mode and custom string to return a sorted list of valid channel indices.
    """
    if target_mode == "all":
        return list(range(total_channels))
    if target_mode == "low":
        return list(range(min(split_index, total_channels)))
    if target_mode == "high":
        return list(range(min(split_index, total_channels), total_channels))
    
    # Custom mode parsing
    parts = [p.strip().replace(' ', '') for p in custom_str.split(',') if p.strip()]
    if not parts:
        return []
    
    rules = []
    for part in parts:
        is_exclude = part.startswith('!')
        raw = part[1:] if is_exclude else part
        
        if '-' in raw:
            s_str, e_str = raw.split('-', 1)
            if s_str.isdigit() and e_str.isdigit():
                rules.append((is_exclude, int(s_str), int(e_str)))
        elif raw.isdigit():
            rules.append((is_exclude, int(raw), int(raw)))
            
    if not rules:
        return []
        
    # If the very first rule is an exclusion, we assume the user wants to start with ALL channels.
    # Otherwise, we start with NO channels.
    target_set = set(range(total_channels)) if rules[0][0] else set()
    
    for is_exclude, start, end in rules:
        # Handle cases where user writes ranges backwards (e.g., 10-0)
        r_start, r_end = min(start, end), max(start, end)
        current_range = set(range(r_start, r_end + 1))
        
        if is_exclude:
            target_set.difference_update(current_range)
        else:
            target_set.update(current_range)
            
    # Finally, filter out any out-of-bounds channels to prevent tensor crashes
    valid_channels = target_set.intersection(set(range(total_channels)))
    return sorted(list(valid_channels))


class LatentBlur_lrzjason:
    """
    Applies a Spatial Gaussian Blur to specific channels of a latent tensor.
    Sampling range is strictly bounded by kernel_size to prevent latent bleeding.
    """
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "latent": ("LATENT",),
                "kernel_size": ("INT", {"default": 3, "min": 3, "max": 31, "step": 2, "label": "Sampling Range (Kernel)"}),
                "sigma": ("FLOAT", {"default": 1.0, "min": 0.1, "max": 20.0, "step": 0.05, "label": "Blur Intensity (Sigma)"}),
                "channel_target": (["all", "low", "high", "custom"], {"default": "all"}),
                "split_index": ("INT", {"default": 64, "min": 1, "max": 512, "step": 1, "label": "Low/High Split Index"}),
                "custom_channels": ("STRING", {"default": "0-63", "multiline": False}),
            },
        }

    RETURN_TYPES = ("LATENT",)
    RETURN_NAMES = ("blurred_latent",)
    FUNCTION = "apply_blur"
    CATEGORY = "latent/enhancement"

    def apply_blur(self, latent, kernel_size, sigma, channel_target, split_index, custom_channels):
        samples = latent["samples"].clone()
        
        is_wan = False
        if samples.ndim == 5:
            samples = samples.squeeze(2)
            is_wan = True
            
        total_channels = samples.shape[1]
        target_indices = parse_target_channels(channel_target, custom_channels, split_index, total_channels)
        
        if target_indices and sigma > 0:
            selected_tensors = samples[:, target_indices, :, :]
            blurred = apply_spatial_gaussian_blur(selected_tensors, kernel_size, sigma)
            samples[:, target_indices, :, :] = blurred
            
        out_latent = latent.copy()
        if is_wan:
            samples = samples.unsqueeze(2)
        out_latent["samples"] = samples
        return (out_latent,)


class LatentSharpen_lrzjason:
    """
    Applies bounded Unsharp Masking to specific channels of a latent tensor.
    Includes an artifact_limit clamp to prevent the VAE from deep-frying.
    """
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "latent": ("LATENT",),
                "strength": ("FLOAT", {"default": 1.0, "min": 0.0, "max": 10.0, "step": 0.05, "label": "Sharpen Strength"}),
                "kernel_size": ("INT", {"default": 3, "min": 3, "max": 31, "step": 2, "label": "Sampling Range (Kernel)"}),
                "sigma": ("FLOAT", {"default": 1.0, "min": 0.1, "max": 10.0, "step": 0.05, "label": "Blur Radius (Sigma)"}),
                "channel_target": (["all", "low", "high", "custom"], {"default": "all"}),
                "split_index": ("INT", {"default": 64, "min": 1, "max": 512, "step": 1, "label": "Low/High Split Index"}),
                "custom_channels": ("STRING", {"default": "64-127, !80-90", "multiline": False}),
                "artifact_limit": ("FLOAT", {"default": 4.0, "min": 0.0, "max": 20.0, "step": 0.1, "label": "Anti-Bleed Clamp"}),
            },
        }

    RETURN_TYPES = ("LATENT",)
    RETURN_NAMES = ("sharpened_latent",)
    FUNCTION = "apply_sharpen"
    CATEGORY = "latent/enhancement"

    def apply_sharpen(self, latent, strength, kernel_size, sigma, channel_target, split_index, custom_channels, artifact_limit):
        samples = latent["samples"].clone()
        
        is_wan = False
        if samples.ndim == 5:
            samples = samples.squeeze(2)
            is_wan = True
            
        total_channels = samples.shape[1]
        target_indices = parse_target_channels(channel_target, custom_channels, split_index, total_channels)
        
        if target_indices and strength > 0:
            selected_tensors = samples[:, target_indices, :, :]
            
            # Unsharp mask formula
            blurred = apply_spatial_gaussian_blur(selected_tensors, kernel_size, sigma)
            sharpened = selected_tensors + strength * (selected_tensors - blurred)
            
            # Anti-Artifact Clamp (prevents VAE blowout)
            if artifact_limit > 0.0:
                sharpened = torch.clamp(sharpened, min=-artifact_limit, max=artifact_limit)
            
            samples[:, target_indices, :, :] = sharpened
            
        out_latent = latent.copy()
        if is_wan:
            samples = samples.unsqueeze(2)
        out_latent["samples"] = samples
        return (out_latent,)


class LatentInterpolate_lrzjason:
    """
    Interpolates (blends) between two latents on a per-channel basis.
    Allows injecting high-frequency texture from one latent into the layout structure of another.
    """
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "latent1": ("LATENT",),
                "latent2": ("LATENT",),
                "factor": ("FLOAT", {"default": 0.5, "min": 0.0, "max": 1.0, "step": 0.01, "label": "Blend Factor (0=L1, 1=L2)"}),
                "channel_target": (["all", "low", "high", "custom"], {"default": "all"}),
                "split_index": ("INT", {"default": 64, "min": 1, "max": 512, "step": 1, "label": "Low/High Split Index"}),
                "custom_channels": ("STRING", {"default": "64-127", "multiline": False}),
            },
        }

    RETURN_TYPES = ("LATENT",)
    RETURN_NAMES = ("interpolated_latent",)
    FUNCTION = "apply_interpolation"
    CATEGORY = "latent/enhancement"

    def apply_interpolation(self, latent1, latent2, factor, channel_target, split_index, custom_channels):
        s1 = latent1["samples"].clone()
        s2 = latent2["samples"].clone()
        
        is_wan = False
        if s1.ndim == 5:
            s1 = s1.squeeze(2)
            s2 = s2.squeeze(2)
            is_wan = True
            
        # Ensure latents can be blended by cropping to the minimum common dimensions
        min_b = min(s1.shape[0], s2.shape[0])
        min_c = min(s1.shape[1], s2.shape[1])
        min_h = min(s1.shape[2], s2.shape[2])
        min_w = min(s1.shape[3], s2.shape[3])
        
        s1 = s1[:min_b, :min_c, :min_h, :min_w]
        s2 = s2[:min_b, :min_c, :min_h, :min_w]
            
        target_indices = parse_target_channels(channel_target, custom_channels, split_index, min_c)
        
        if target_indices:
            # Linear interpolation (lerp) specifically on targeted channels
            s1[:, target_indices, :, :] = s1[:, target_indices, :, :] * (1.0 - factor) + s2[:, target_indices, :, :] * factor
            
        out_latent = latent1.copy()
        if is_wan:
            s1 = s1.unsqueeze(2)
        out_latent["samples"] = s1
        return (out_latent,)


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


class LatentColorAdjust_lrzjason:
    """
    ComfyUI Node to adjust contrast and saturation in the latent space.
    Incorporates a channel-split mechanism: applies full strength to low-level 
    (structure) channels and a weighted/dampened strength to high-level (texture) 
    channels to prevent harsh artifacts in modern multi-channel VAEs.
    """
    
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "latent": ("LATENT",),
                "contrast": ("FLOAT", {
                    "default": 1.0, 
                    "min": 0.0, 
                    "max": 3.0, 
                    "step": 0.01, 
                    "label": "Contrast"
                }),
                "saturation": ("FLOAT", {
                    "default": 1.0, 
                    "min": 0.0, 
                    "max": 3.0, 
                    "step": 0.01, 
                    "label": "Saturation"
                }),
                "split_channel": ("INT", {
                    "default": 64, 
                    "min": 1, 
                    "max": 256, 
                    "step": 1, 
                    "label": "Channel Split Index"
                }),
                "high_contrast_weight": ("FLOAT", {
                    "default": 0.25, 
                    "min": 0.0, 
                    "max": 1.0, 
                    "step": 0.01, 
                    "label": "High-Level Contrast Weight"
                }),
                "high_saturation_weight": ("FLOAT", {
                    "default": 0.25, 
                    "min": 0.0, 
                    "max": 1.0, 
                    "step": 0.01, 
                    "label": "High-Level Saturation Weight"
                }),
            },
        }

    RETURN_TYPES = ("LATENT",)
    RETURN_NAMES = ("adjusted_latent",)
    FUNCTION = "adjust_color"
    CATEGORY = "latent/enhancement"

    def adjust_color(self, latent, contrast, saturation, split_channel, high_contrast_weight, high_saturation_weight):
        samples = latent["samples"].clone()
        
        # Handle WAN format if needed
        is_wan = False
        if samples.ndim == 5:
            samples = samples.squeeze(2)
            is_wan = True
            
        channels = samples.shape[1]
        
        # Safely cap the split channel so it doesn't crash on standard 4-channel/16-channel models
        split = min(split_channel, channels)
        
        # Calculate effective strengths for the high-frequency channels
        # If contrast is 1.5 and weight is 0.25, eff_contrast = 1.0 + (0.5 * 0.25) = 1.125
        eff_contrast = 1.0 + (contrast - 1.0) * high_contrast_weight
        eff_saturation = 1.0 + (saturation - 1.0) * high_saturation_weight

        # --- 1. Contrast Adjustment ---
        if contrast != 1.0:
            # Calculate the spatial mean of the latents [Batch, Channels, 1, 1]
            mean = samples.mean(dim=[-2, -1], keepdim=True)
            
            # Apply to Low Channels (Structure/Layout)
            if split > 0:
                samples[:, :split] = (samples[:, :split] - mean[:, :split]) * contrast + mean[:, :split]
            
            # Apply dampened effect to High Channels (Texture/Detail)
            if split < channels:
                samples[:, split:] = (samples[:, split:] - mean[:, split:]) * eff_contrast + mean[:, split:]
                
        # --- 2. Saturation Adjustment ---
        # Assuming channel 0 is Luma-like, and channels 1+ are Chroma-like
        if saturation != 1.0 and channels > 1:
            # Apply to Low Channels (Chroma Structure)
            if split > 1:
                samples[:, 1:split] = samples[:, 1:split] * saturation
                
            # Apply dampened effect to High Channels (Chroma Detail)
            if split < channels:
                start_high = max(split, 1) # Ensure we don't accidentally touch channel 0
                samples[:, start_high:] = samples[:, start_high:] * eff_saturation
                
        # Prepare Latent Output
        adjusted_latent = latent.copy()
        if is_wan:
            samples = samples.unsqueeze(2)
        adjusted_latent["samples"] = samples
        
        return (adjusted_latent,)


class LatentMathExplorer_lrzjason:
    """
    An experimental node to discover how manipulating latent mathematics 
    affects the decoded image. Test variance, means, magnitudes, and local vs global processing.
    """
    
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "latent": ("LATENT",),
                "operation": (["scale_variance", "shift_mean", "scale_magnitude", "power_curve", "threshold_boost"],),
                "calc_mode": (["per_channel", "global"],),
                "factor": ("FLOAT", {"default": 1.0, "min": -10.0, "max": 10.0, "step": 0.01, "label": "Operation Factor"}),
                "channel_target": (["all", "low", "high", "custom"], {"default": "all"}),
                "split_index": ("INT", {"default": 64, "min": 1, "max": 512, "step": 1, "label": "HF Split Index"}),
                "custom_channels": ("STRING", {"default": "0-15", "multiline": False}),
                "high_freq_weight": ("FLOAT", {"default": 1.0, "min": 0.0, "max": 1.0, "step": 0.01, "label": "HF Dampening Weight"}),
                "artifact_limit": ("FLOAT", {"default": 5.0, "min": 0.0, "max": 20.0, "step": 0.1, "label": "Anti-Bleed Clamp (0=Off)"}),
            },
        }

    RETURN_TYPES = ("LATENT",)
    RETURN_NAMES = ("explored_latent",)
    FUNCTION = "explore_math"
    CATEGORY = "latent/experimental"

    def explore_math(self, latent, operation, calc_mode, factor, channel_target, 
                     split_index, custom_channels, high_freq_weight, artifact_limit):
                     
        samples = latent["samples"].clone()
        
        is_wan = False
        if samples.ndim == 5:
            samples = samples.squeeze(2)
            is_wan = True
            
        total_channels = samples.shape[1]
        target_indices = parse_target_channels(channel_target, custom_channels, split_index, total_channels)
        
        if not target_indices:
            return (latent,) # Do nothing if no channels selected
            
        # Extract only targeted channels to modify
        selected = samples[:, target_indices, :, :]
        
        # Determine the "neutral" value based on the operation
        # Additive ops (shift, threshold) use 0.0. Multiplicative ops (scale, power) use 1.0.
        is_additive = operation in ["shift_mean", "threshold_boost"]
        neutral_val = 0.0 if is_additive else 1.0
        
        # Build a multiplier tensor to handle High-Frequency dampening natively
        factor_tensor = torch.full((1, len(target_indices), 1, 1), neutral_val, device=samples.device, dtype=samples.dtype)
        
        for i, ch in enumerate(target_indices):
            if ch >= split_index:
                # Calculate dampened factor for High Frequencies
                eff_factor = neutral_val + (factor - neutral_val) * high_freq_weight
                factor_tensor[0, i, 0, 0] = eff_factor
            else:
                factor_tensor[0, i, 0, 0] = factor

        # Calculate Mean (Per-Channel vs Global across all targeted channels)
        if calc_mode == "per_channel":
            mean = selected.mean(dim=[-2, -1], keepdim=True)
        else: # global
            mean = selected.mean(dim=[-3, -2, -1], keepdim=True)

        # --- Apply the Experimental Math Operations ---
        if operation == "scale_variance":
            # (x - mean) * factor + mean
            processed = (selected - mean) * factor_tensor + mean
            
        elif operation == "shift_mean":
            # x + factor
            processed = selected + factor_tensor
            
        elif operation == "scale_magnitude":
            # x * factor (does not respect the mean)
            processed = selected * factor_tensor
            
        elif operation == "power_curve":
            # non-linear push: sign(x-mean) * abs(x-mean)^factor + mean
            diff = selected - mean
            # safeguard against fractional powers of negative numbers
            processed = torch.sign(diff) * (torch.abs(diff) ** factor_tensor) + mean
            
        elif operation == "threshold_boost":
            # If above mean, + factor. If below mean, - factor.
            processed = torch.where(selected > mean, selected + factor_tensor, selected - factor_tensor)

        # --- Recombination and Anti-Artifact Clamping ---
        if artifact_limit > 0.0:
            processed = torch.clamp(processed, min=-artifact_limit, max=artifact_limit)
            
        # Put processed channels back into the main tensor
        samples[:, target_indices, :, :] = processed

        out_latent = latent.copy()
        if is_wan:
            samples = samples.unsqueeze(2)
        out_latent["samples"] = samples
        return (out_latent,)


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

NODE_CLASS_MAPPINGS["LatentColorAdjust_lrzjason"] = LatentColorAdjust_lrzjason
NODE_DISPLAY_NAME_MAPPINGS["LatentColorAdjust_lrzjason"] = "Latent Color Adjust (lrzjason)"

NODE_CLASS_MAPPINGS["HFEPostProcessor (lrzjason)"] = HFEPostProcessor
NODE_DISPLAY_NAME_MAPPINGS["HFEPostProcessor (lrzjason)"] = "HFEPostProcessor (lrzjason)"

NODE_CLASS_MAPPINGS["LatentSharpen_lrzjason"] = LatentSharpen_lrzjason
NODE_DISPLAY_NAME_MAPPINGS["LatentSharpen_lrzjason"] = "Latent Sharpen (lrzjason)"

NODE_CLASS_MAPPINGS["LatentBlur_lrzjason"] = LatentBlur_lrzjason
NODE_DISPLAY_NAME_MAPPINGS["LatentBlur_lrzjason"] = "Latent Blur (lrzjason)"

NODE_CLASS_MAPPINGS["LatentInterpolate_lrzjason"] = LatentInterpolate_lrzjason
NODE_DISPLAY_NAME_MAPPINGS["LatentInterpolate_lrzjason"] = "Latent Interpolate (lrzjason)"

NODE_CLASS_MAPPINGS["LatentMathExplorer_lrzjason"] = LatentMathExplorer_lrzjason
NODE_DISPLAY_NAME_MAPPINGS["LatentMathExplorer_lrzjason"] = "Latent Math Explorer (lrzjason)"
