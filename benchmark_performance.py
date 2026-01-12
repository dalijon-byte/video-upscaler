#!/usr/bin/env python3
"""Benchmark performance of different Stream-DiffVSR configurations"""

import os
import sys
import time
import argparse
import torch
import cv2
from PIL import Image
from pathlib import Path

# Add parent directory to path
parent_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.append(parent_dir)

from inference import extract_frames_from_video, load_component, process_frames_in_batches
from pipeline.stream_diffvsr_pipeline import StreamDiffVSRPipeline, ControlNetModel, UNet2DConditionModel
from diffusers import DDIMScheduler
from temporal_autoencoder.autoencoder_tiny import TemporalAutoencoderTiny
from torchvision.models.optical_flow import raft_large, Raft_Large_Weights

def benchmark_configuration(config_name, pipeline, of_model, test_frames, batch_size, of_rescale_factor, num_inference_steps=4):
    """Benchmark a specific configuration"""
    print(f"\n{'='*60}")
    print(f"Benchmarking: {config_name}")
    print(f"{'='*60}")
    
    # Create temp directory for output
    temp_dir = f"./benchmark_temp_{config_name.replace(' ', '_').lower()}"
    os.makedirs(temp_dir, exist_ok=True)
    
    # Clear GPU cache
    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats()
    
    # Start timing
    start_time = time.time()
    memory_before = torch.cuda.memory_allocated() / 1024**3  # GB
    
    try:
        # Process frames
        frames_hr = process_frames_in_batches(
            pipeline, test_frames, of_model,
            num_inference_steps, batch_size, temp_dir,
            of_rescale_factor=of_rescale_factor
        )
        
        # Calculate metrics
        end_time = time.time()
        total_time = end_time - start_time
        memory_after = torch.cuda.memory_allocated() / 1024**3
        memory_peak = torch.cuda.max_memory_allocated() / 1024**3
        
        # Calculate frames per second
        fps = len(test_frames) / total_time if total_time > 0 else 0
        
        print(f"Results for {config_name}:")
        print(f"  Total time: {total_time:.2f} seconds")
        print(f"  Frames processed: {len(test_frames)}")
        print(f"  FPS: {fps:.2f}")
        print(f"  Memory before: {memory_before:.2f} GB")
        print(f"  Memory after: {memory_after:.2f} GB")
        print(f"  Peak memory: {memory_peak:.2f} GB")
        print(f"  Batch size: {batch_size}")
        print(f"  OF rescale factor: {of_rescale_factor}")
        
        # Clean up
        for file in os.listdir(temp_dir):
            os.remove(os.path.join(temp_dir, file))
        os.rmdir(temp_dir)
        
        return {
            "config_name": config_name,
            "total_time": total_time,
            "fps": fps,
            "peak_memory_gb": memory_peak,
            "batch_size": batch_size,
            "of_rescale_factor": of_rescale_factor,
            "success": True
        }
        
    except Exception as e:
        print(f"  ❌ Failed: {e}")
        
        # Clean up temp dir if it exists
        if os.path.exists(temp_dir):
            for file in os.listdir(temp_dir):
                os.remove(os.path.join(temp_dir, file))
            os.rmdir(temp_dir)
        
        return {
            "config_name": config_name,
            "total_time": None,
            "fps": None,
            "peak_memory_gb": None,
            "batch_size": batch_size,
            "of_rescale_factor": of_rescale_factor,
            "success": False,
            "error": str(e)
        }

def main():
    parser = argparse.ArgumentParser(description="Benchmark Stream-DiffVSR performance")
    parser.add_argument("--test_video", type=str, required=True, help="Path to test video file")
    parser.add_argument("--num_test_frames", type=int, default=10, help="Number of frames to test with")
    parser.add_argument("--model_id", default='Jamichsu/Stream-DiffVSR', type=str, help="Model ID")
    args = parser.parse_args()
    
    print("Stream-DiffVSR Performance Benchmark")
    print("="*60)
    
    # Extract test frames
    print(f"Extracting frames from {args.test_video}...")
    frames, fps, width, height = extract_frames_from_video(args.test_video)
    
    # Limit to test frames
    test_frames = frames[:args.num_test_frames]
    print(f"Testing with {len(test_frames)} frames ({width}x{height})")
    
    device = torch.device('cuda')
    results = []
    
    # Test configurations
    configurations = [
        # (name, use_fp16, batch_size, of_rescale_factor, enable_memory_optimizations)
        ("Baseline (FP32, BS=2, OF=4)", False, 2, 4, True),
        ("FP16 (BS=2, OF=4)", True, 2, 4, True),
        ("FP16 (BS=4, OF=4)", True, 4, 4, True),
        ("FP16 (BS=2, OF=8)", True, 2, 8, True),
        ("FP16 (BS=1, OF=8)", True, 1, 8, True),
        ("FP32 (BS=4, OF=8)", False, 4, 8, True),
    ]
    
    for config_name, use_fp16, batch_size, of_rescale_factor, enable_mem_opt in configurations:
        print(f"\n\nPreparing configuration: {config_name}")
        
        # Clear GPU cache between tests
        torch.cuda.empty_cache()
        
        # Determine dtype
        torch_dtype = torch.float16 if use_fp16 else None
        
        try:
            # Load components
            controlnet = load_component(ControlNetModel, None, args.model_id, "controlnet", torch_dtype)
            unet = load_component(UNet2DConditionModel, None, args.model_id, "unet", torch_dtype)
            vae = load_component(TemporalAutoencoderTiny, None, args.model_id, "vae", torch_dtype)
            scheduler = DDIMScheduler.from_pretrained(args.model_id, subfolder="scheduler")
            
            # Create pipeline
            pipeline_kwargs = {
                "controlnet": controlnet,
                "vae": vae,
                "unet": unet,
                "scheduler": scheduler,
            }
            
            if torch_dtype is not None:
                pipeline_kwargs["torch_dtype"] = torch_dtype
            
            pipeline = StreamDiffVSRPipeline.from_pretrained(
                args.model_id,
                **pipeline_kwargs
            )
            
            pipeline = pipeline.to(device)
            
            # Enable memory optimizations
            if enable_mem_opt:
                pipeline.enable_xformers_memory_efficient_attention()
                pipeline.enable_vae_slicing()
            
            # Load optical flow model
            of_model = raft_large(weights=Raft_Large_Weights.DEFAULT).to(device).eval()
            of_model.requires_grad_(False)
            
            if use_fp16:
                of_model = of_model.half()
            
            # Run benchmark
            result = benchmark_configuration(
                config_name, pipeline, of_model, test_frames,
                batch_size, of_rescale_factor
            )
            
            results.append(result)
            
            # Clean up
            del pipeline
            del of_model
            del controlnet
            del unet
            del vae
            torch.cuda.empty_cache()
            
        except Exception as e:
            print(f"  ❌ Configuration failed to load: {e}")
            results.append({
                "config_name": config_name,
                "success": False,
                "error": f"Load failed: {e}"
            })
    
    # Print summary
    print(f"\n\n{'='*60}")
    print("PERFORMANCE BENCHMARK SUMMARY")
    print(f"{'='*60}")
    
    successful_results = [r for r in results if r.get("success", False)]
    
    if successful_results:
        print("\nSuccessful configurations (sorted by FPS):")
        print("-"*80)
        print(f"{'Configuration':<30} {'FPS':<10} {'Time (s)':<10} {'Peak Mem (GB)':<15} {'Batch':<8} {'OF Factor':<10}")
        print("-"*80)
        
        for result in sorted(successful_results, key=lambda x: x.get("fps", 0), reverse=True):
            print(f"{result['config_name']:<30} {result['fps']:<10.2f} {result['total_time']:<10.2f} {result['peak_memory_gb']:<15.2f} {result['batch_size']:<8} {result['of_rescale_factor']:<10}")
    
    # Find optimal configuration
    if successful_results:
        best_fps = max(successful_results, key=lambda x: x.get("fps", 0))
        best_memory = min(successful_results, key=lambda x: x.get("peak_memory_gb", float('inf')))
        
        print(f"\n{'='*60}")
        print("RECOMMENDATIONS:")
        print(f"{'='*60}")
        print(f"🏆 Best FPS: {best_fps['config_name']}")
        print(f"   - {best_fps['fps']:.2f} FPS, {best_fps['total_time']:.2f} seconds")
        print(f"   - Peak memory: {best_fps['peak_memory_gb']:.2f} GB")
        print(f"   - Batch size: {best_fps['batch_size']}, OF factor: {best_fps['of_rescale_factor']}")
        
        print(f"\n💾 Most memory efficient: {best_memory['config_name']}")
        print(f"   - {best_memory['fps']:.2f} FPS, {best_memory['total_time']:.2f} seconds")
        print(f"   - Peak memory: {best_memory['peak_memory_gb']:.2f} GB")
        print(f"   - Batch size: {best_memory['batch_size']}, OF factor: {best_memory['of_rescale_factor']}")
        
        # Calculate estimated time for full video
        total_frames = len(frames)
        estimated_time_full = (total_frames / best_fps['fps']) if best_fps['fps'] > 0 else 0
        estimated_time_minutes = estimated_time_full / 60
        
        print(f"\n📊 For your {total_frames}-frame video:")
        print(f"   - Estimated processing time: {estimated_time_full:.0f} seconds ({estimated_time_minutes:.1f} minutes)")
        print(f"   - Using configuration: {best_fps['config_name']}")
    
    # Failed configurations
    failed_results = [r for r in results if not r.get("success", False)]
    if failed_results:
        print(f"\n{'='*60}")
        print("FAILED CONFIGURATIONS:")
        print(f"{'='*60}")
        for result in failed_results:
            print(f"❌ {result['config_name']}: {result.get('error', 'Unknown error')}")

if __name__ == "__main__":
    main()