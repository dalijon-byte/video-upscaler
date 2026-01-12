#!/usr/bin/env python3
"""Test TensorRT acceleration with Stream-DiffVSR"""

import torch
import sys
import os

# Add parent directory to path
parent_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.append(parent_dir)

from pipeline.stream_diffvsr_pipeline import StreamDiffVSRPipeline, ControlNetModel, UNet2DConditionModel
from diffusers import DDIMScheduler
from temporal_autoencoder.autoencoder_tiny import TemporalAutoencoderTiny

def test_tensorrt_setup():
    """Test if TensorRT can be loaded and initialized"""
    print("Testing TensorRT setup...")
    
    device = torch.device('cuda')
    
    try:
        # Try to load components with TensorRT
        print("Loading components...")
        
        # Load components
        controlnet = ControlNetModel.from_pretrained(
            "Jamichsu/Stream-DiffVSR", 
            subfolder="controlnet",
            torch_dtype=torch.float16
        )
        
        unet = UNet2DConditionModel.from_pretrained(
            "Jamichsu/Stream-DiffVSR", 
            subfolder="unet",
            torch_dtype=torch.float16
        )
        
        vae = TemporalAutoencoderTiny.from_pretrained(
            "Jamichsu/Stream-DiffVSR", 
            subfolder="vae",
            torch_dtype=torch.float16
        )
        
        scheduler = DDIMScheduler.from_pretrained(
            "Jamichsu/Stream-DiffVSR", 
            subfolder="scheduler"
        )
        
        print("Components loaded successfully")
        
        # Try to create pipeline with TensorRT
        print("\nCreating pipeline with TensorRT...")
        tensorrt_kwargs = {
            "custom_pipeline": "/acceleration/tensorrt/sd_with_controlnet_ST",
            "image_height": 720,
            "image_width": 1280,
        }
        
        pipeline = StreamDiffVSRPipeline.from_pretrained(
            "Jamichsu/Stream-DiffVSR",
            controlnet=controlnet, 
            vae=vae, 
            unet=unet, 
            scheduler=scheduler,
            torch_dtype=torch.float16,
            **tensorrt_kwargs
        )
        
        print("Pipeline created with TensorRT")
        
        # Move to device
        pipeline = pipeline.to(device)
        
        # Set cached folder
        pipeline.set_cached_folder("Jamichsu/Stream-DiffVSR")
        
        print("\n✅ TensorRT setup test PASSED")
        print("Note: First run will compile TensorRT engines, which may take several minutes.")
        
        return True
        
    except Exception as e:
        print(f"\n❌ TensorRT setup test FAILED: {e}")
        print("\nTroubleshooting steps:")
        print("1. Make sure TensorRT is installed: `pip install tensorrt`")
        print("2. Check CUDA compatibility")
        print("3. Ensure you have enough GPU memory")
        print("4. The first run requires engine compilation for specific resolution")
        return False

if __name__ == "__main__":
    test_tensorrt_setup()