#!/usr/bin/env python3
"""Stream-DiffVSR inference with reduced output size options"""

import os
import sys
import argparse
import time
from pathlib import Path
import torch
from accelerate.utils import set_seed
from PIL import Image
from torchvision.models.optical_flow import raft_large, Raft_Large_Weights
import cv2
import numpy as np

parent_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.append(parent_dir)

from pipeline.stream_diffvsr_pipeline import StreamDiffVSRPipeline, ControlNetModel, UNet2DConditionModel
from diffusers import DDIMScheduler
from temporal_autoencoder.autoencoder_tiny import TemporalAutoencoderTiny

torch.backends.cuda.matmul.allow_tf32 = True

def extract_frames_from_video(video_path, target_width=None, target_height=None):
    """Extract frames from a video file using OpenCV, optionally resizing."""
    cap = cv2.VideoCapture(video_path)
    frames = []
    frame_count = 0
    
    # Get video properties for later use
    fps = cap.get(cv2.CAP_PROP_FPS)
    original_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    original_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    
    # Calculate target dimensions if resizing
    if target_width and target_height:
        print(f"Resizing frames from {original_width}x{original_height} to {target_width}x{target_height}")
    
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        
        # Resize if requested
        if target_width and target_height:
            frame = cv2.resize(frame, (target_width, target_height), interpolation=cv2.INTER_AREA)
        
        # Convert BGR to RGB
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        # Convert to PIL Image
        pil_image = Image.fromarray(frame_rgb)
        frames.append(pil_image)
        frame_count += 1
    
    cap.release()
    
    # Use actual dimensions after resizing
    actual_width = target_width if target_width else original_width
    actual_height = target_height if target_height else original_height
    
    print(f"Extracted {frame_count} frames from video: {video_path}")
    print(f"Video properties: {actual_width}x{actual_height} @ {fps:.2f} fps")
    return frames, fps, actual_width, actual_height

def parse_args():
    parser = argparse.ArgumentParser(description="Stream-DiffVSR with reduced output size options.")
    parser.add_argument("--model_id", default='Jamichsu/Stream-DiffVSR', type=str, help="model_id of the model to be tested.")
    parser.add_argument("--out_path", default='./StreamDiffVSR_results/', type=str, help="Path to output folder.")
    parser.add_argument("--in_path", type=str, required=True, help="Path to input video file.")
    parser.add_argument("--num_inference_steps", type=int, default=4, help="Number of sampling steps")
    parser.add_argument("--batch_size", type=int, default=1, help="Number of frames to process at once")
    parser.add_argument("--enable_memory_optimizations", action='store_true', help="Enable memory optimizations")
    parser.add_argument("--of_rescale_factor", type=int, default=8, help="Rescale factor for optical flow computation")
    parser.add_argument("--use_fp16", action='store_true', help="Use half-precision (FP16)")
    
    # Size reduction options
    parser.add_argument("--reduce_input_size", action='store_true', help="Reduce input size before processing")
    parser.add_argument("--input_scale_factor", type=float, default=0.5, help="Scale factor for input reduction (e.g., 0.5 for half size)")
    parser.add_argument("--reduce_output_size", action='store_true', help="Reduce output size after upscaling")
    parser.add_argument("--output_scale_factor", type=float, default=0.5, help="Scale factor for output reduction")
    
    return parser.parse_args()

def load_component(cls, weight_path, model_id, subfolder, torch_dtype=None):
    path = weight_path if weight_path else model_id
    sub = None if weight_path else subfolder
    kwargs = {"subfolder": sub}
    if torch_dtype is not None:
        kwargs["torch_dtype"] = torch_dtype
    return cls.from_pretrained(path, **kwargs)

def process_frames_in_batches(pipeline, frames, of_model, num_inference_steps, batch_size, target_path, 
                             of_rescale_factor=4, output_scale_factor=1.0):
    """Process frames in batches, optionally reducing output size."""
    all_hr_frames = []
    
    for batch_start in range(0, len(frames), batch_size):
        batch_end = min(batch_start + batch_size, len(frames))
        batch_frames = frames[batch_start:batch_end]
        
        print(f"Processing batch {batch_start//batch_size + 1}/{(len(frames) + batch_size - 1)//batch_size}: frames {batch_start}-{batch_end-1}")
        
        output = pipeline(
            '', batch_frames,
            num_inference_steps=num_inference_steps,
            guidance_scale=0,
            of_model=of_model,
            of_rescale_factor=of_rescale_factor
        )
        
        batch_hr_frames = [frame[0] for frame in output.images]
        
        # Reduce output size if requested
        if output_scale_factor != 1.0:
            print(f"  Reducing output size by factor {output_scale_factor}")
            resized_frames = []
            for frame in batch_hr_frames:
                new_width = int(frame.width * output_scale_factor)
                new_height = int(frame.height * output_scale_factor)
                resized_frame = frame.resize((new_width, new_height), Image.Resampling.LANCZOS)
                resized_frames.append(resized_frame)
            batch_hr_frames = resized_frames
        
        all_hr_frames.extend(batch_hr_frames)
        
        # Save frames immediately to free memory
        for i, frame in enumerate(batch_hr_frames):
            frame_idx = batch_start + i
            frame_name = f"frame_{frame_idx:04d}.png"
            frame.save(os.path.join(target_path, frame_name))
        
        # Clear memory
        del output
        del batch_hr_frames
        torch.cuda.empty_cache()
    
    return all_hr_frames

def create_video_from_frames(frames_dir, output_video_path, fps, width, height, original_video_path=None):
    """Create a video from frames and optionally add audio from original video."""
    import subprocess
    
    # Get sorted list of frame files
    frame_files = sorted([f for f in os.listdir(frames_dir) if f.endswith('.png')])
    if not frame_files:
        print(f"Error: No PNG frames found in {frames_dir}")
        return False
    
    print(f"Creating video from {len(frame_files)} frames ({width}x{height})...")
    
    # Create video from frames using OpenCV
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    video_writer = cv2.VideoWriter(output_video_path, fourcc, fps, (width, height))
    
    for frame_file in frame_files:
        frame_path = os.path.join(frames_dir, frame_file)
        frame = cv2.imread(frame_path)
        if frame is None:
            print(f"Warning: Could not read frame {frame_file}")
            continue
        video_writer.write(frame)
    
    video_writer.release()
    print(f"Video created: {output_video_path}")
    
    # If original video has audio, extract and merge it
    if original_video_path and os.path.exists(original_video_path):
        print("Extracting audio from original video...")
        
        # Create temporary audio file
        temp_audio_path = os.path.join(frames_dir, "temp_audio.aac")
        
        # Extract audio using ffmpeg
        try:
            # First, extract audio
            extract_cmd = [
                'ffmpeg', '-i', original_video_path,
                '-vn', '-acodec', 'copy',
                temp_audio_path, '-y'
            ]
            subprocess.run(extract_cmd, check=True, capture_output=True)
            
            # Create final video with audio
            final_video_path = output_video_path.replace('.mp4', '_with_audio.mp4')
            merge_cmd = [
                'ffmpeg', '-i', output_video_path,
                '-i', temp_audio_path,
                '-c:v', 'copy', '-c:a', 'aac',
                '-map', '0:v:0', '-map', '1:a:0',
                '-shortest',
                final_video_path, '-y'
            ]
            subprocess.run(merge_cmd, check=True, capture_output=True)
            
            # Clean up temporary files
            os.remove(temp_audio_path)
            os.remove(output_video_path)  # Remove video without audio
            os.rename(final_video_path, output_video_path)  # Rename to original name
            
            print(f"Video with audio created: {output_video_path}")
            
        except subprocess.CalledProcessError as e:
            print(f"Warning: Could not merge audio: {e}")
            print("Video created without audio.")
        except FileNotFoundError:
            print("Warning: ffmpeg not found. Video created without audio.")
    
    return True

def main():
    args = parse_args()

    print("Stream-DiffVSR with Reduced Output Size")
    print("="*60)
    print("Run with arguments:")
    for arg, value in vars(args).items():
        print(f"  {arg}: {value}")

    set_seed(42)
    device = torch.device('cuda')
    
    # Determine dtype for FP16 support
    torch_dtype = torch.float16 if args.use_fp16 else None
    print(f"Using dtype: {torch_dtype}")

    # Calculate input dimensions if reducing input size
    input_target_width = None
    input_target_height = None
    
    if args.reduce_input_size:
        # First, get original dimensions to calculate target
        cap = cv2.VideoCapture(args.in_path)
        original_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        original_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        cap.release()
        
        input_target_width = int(original_width * args.input_scale_factor)
        input_target_height = int(original_height * args.input_scale_factor)
        print(f"Input reduction: {original_width}x{original_height} -> {input_target_width}x{input_target_height}")
    
    # Extract frames (with optional resizing)
    frames, fps, width, height = extract_frames_from_video(
        args.in_path, 
        target_width=input_target_width, 
        target_height=input_target_height
    )
    
    if not frames:
        print(f"Error: No frames extracted from video {args.in_path}")
        return
    
    # Calculate output dimensions
    # Stream-DiffVSR always does 4x upscale
    upscale_factor = 4
    output_width = width * upscale_factor
    output_height = height * upscale_factor
    
    # Apply output reduction if requested
    final_output_width = output_width
    final_output_height = output_height
    output_scale_factor = 1.0
    
    if args.reduce_output_size:
        output_scale_factor = args.output_scale_factor
        final_output_width = int(output_width * output_scale_factor)
        final_output_height = int(output_height * output_scale_factor)
        print(f"Output reduction: {output_width}x{output_height} -> {final_output_width}x{final_output_height}")
    
    # Adjust of_rescale_factor for very small inputs
    # RAFT requires dimensions to be divisible by 8 after downsampling
    adjusted_of_rescale_factor = args.of_rescale_factor
    
    # Check if after downsampling by of_rescale_factor, dimensions would be too small
    downsampled_height = height // adjusted_of_rescale_factor
    downsampled_width = width // adjusted_of_rescale_factor
    
    if downsampled_height < 32 or downsampled_width < 32:
        # Calculate maximum rescale factor that keeps dimensions reasonable
        max_rescale_height = max(1, height // 32)
        max_rescale_width = max(1, width // 32)
        max_rescale = min(max_rescale_height, max_rescale_width)
        
        if max_rescale < adjusted_of_rescale_factor:
            print(f"Warning: Input size {width}x{height} is too small for of_rescale_factor={adjusted_of_rescale_factor}")
            print(f"  Reducing of_rescale_factor to {max_rescale} to maintain RAFT compatibility")
            adjusted_of_rescale_factor = max_rescale
    
    # Also ensure dimensions are divisible by 8 after downsampling
    if height % (adjusted_of_rescale_factor * 8) != 0 or width % (adjusted_of_rescale_factor * 8) != 0:
        print(f"Note: Input dimensions {width}x{height} may need padding for RAFT with of_rescale_factor={adjusted_of_rescale_factor}")
        print(f"  The get_flow() function will automatically pad images to be divisible by 8")
    
    print(f"\nProcessing summary:")
    print(f"  Input: {width}x{height}")
    print(f"  After 4x upscale: {output_width}x{output_height}")
    print(f"  Final output: {final_output_width}x{final_output_height}")
    print(f"  Total frames: {len(frames)}")
    
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
    
    # Enable memory optimizations if requested
    if args.enable_memory_optimizations:
        print("Enabling memory optimizations...")
        pipeline.enable_xformers_memory_efficient_attention()
        pipeline.enable_vae_slicing()
    else:
        pipeline.enable_xformers_memory_efficient_attention()
    
    # Load optical flow model
    of_model = raft_large(weights=Raft_Large_Weights.DEFAULT).to(device).eval()
    of_model.requires_grad_(False)
    
    if args.use_fp16:
        print("Converting optical flow model to FP16...")
        of_model = of_model.half()
    
    # Generate output directory name from video filename
    video_name = os.path.splitext(os.path.basename(args.in_path))[0]
    target_path = os.path.join(args.out_path, video_name)
    os.makedirs(target_path, exist_ok=True)
    
    # Process frames in batches
    start_time = time.time()
    
    frames_hr = process_frames_in_batches(
        pipeline, frames, of_model,
        args.num_inference_steps, args.batch_size, target_path,
        of_rescale_factor=adjusted_of_rescale_factor,
        output_scale_factor=output_scale_factor
    )
    
    processing_time = time.time() - start_time
    print(f"\nUpscaled video and saved {len(frames_hr)} frames to {target_path}.")
    print(f"Total processing time: {processing_time:.2f} seconds ({processing_time/60:.1f} minutes)")
    print(f"Average time per frame: {processing_time/len(frames):.2f} seconds")
    
    # Create video from frames
    output_video_path = os.path.join(args.out_path, f"{video_name}_upscaled_{final_output_width}x{final_output_height}.mp4")
    create_video_from_frames(
        target_path, output_video_path, fps,
        final_output_width, final_output_height, args.in_path
    )
    
    # Clean up
    del frames
    del frames_hr
    torch.cuda.empty_cache()
    
    print(f"\n✅ Processing complete!")
    print(f"   Input: {width}x{height}")
    print(f"   Output: {final_output_width}x{final_output_height}")
    print(f"   Video saved: {output_video_path}")

if __name__ == "__main__":
    main()