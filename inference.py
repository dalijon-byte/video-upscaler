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

def extract_frames_from_video(video_path):
    """Extract frames from a video file using OpenCV."""
    cap = cv2.VideoCapture(video_path)
    frames = []
    frame_count = 0
    
    # Get video properties for later use
    fps = cap.get(cv2.CAP_PROP_FPS)
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        # Convert BGR to RGB
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        # Convert to PIL Image
        pil_image = Image.fromarray(frame_rgb)
        frames.append(pil_image)
        frame_count += 1
    
    cap.release()
    print(f"Extracted {frame_count} frames from video: {video_path}")
    print(f"Video properties: {width}x{height} @ {fps:.2f} fps")
    return frames, fps, width, height

def parse_args():
    parser = argparse.ArgumentParser(description="Test code for Stream-DiffVSR.")
    parser.add_argument("--model_id", default='stabilityai/stable-diffusion-x4-upscaler', type=str, help="model_id of the model to be tested.")
    parser.add_argument("--unet_pretrained_weight", type=str, help="UNet pretrained weight.")
    parser.add_argument("--controlnet_pretrained_weight", type=str, help="ControlNet pretrained weight.")
    parser.add_argument("--temporal_vae_pretrained_weight", type=str, help="Path to Temporal VAE.")
    parser.add_argument("--out_path", default='./StreamDiffVSR_results/', type=str, help="Path to output folder.")
    parser.add_argument("--in_path", type=str, required=True, help="Path to input folder (containing sets of LR images) or a video file.")
    parser.add_argument("--num_inference_steps", type=int, default=4, help="Number of sampling steps")
    parser.add_argument("--enable_tensorrt", action='store_true', help="Enable TensorRT. Note that the performance will drop if TensorRT is enabled.")
    parser.add_argument("--image_height", type=int, default=720, help="Height of the output images. Needed for TensorRT.")
    parser.add_argument("--image_width", type=int, default=1280, help="Width of the output images. Needed for TensorRT.")
    parser.add_argument("--batch_size", type=int, default=10, help="Number of frames to process at once (reduce if OOM)")
    parser.add_argument("--enable_memory_optimizations", action='store_true', help="Enable memory optimizations (CPU offload, VAE slicing)")
    parser.add_argument("--of_rescale_factor", type=int, default=4, help="Rescale factor for optical flow computation (reduce for memory savings)")
    return parser.parse_args()

def load_component(cls, weight_path, model_id, subfolder):
    path = weight_path if weight_path else model_id
    sub = None if weight_path else subfolder
    return cls.from_pretrained(path, subfolder=sub)

def process_frames_in_batches(pipeline, frames, of_model, num_inference_steps, batch_size, target_path, of_rescale_factor=4):
    """Process frames in batches to avoid OOM errors."""
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
    
    print(f"Creating video from {len(frame_files)} frames...")
    
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

    print("Run with arguments:")
    for arg, value in vars(args).items():
        print(f"  {arg}: {value}")

    set_seed(42)
    device = torch.device('cuda')

    controlnet = load_component(ControlNetModel, args.controlnet_pretrained_weight, args.model_id, "controlnet")
    unet = load_component(UNet2DConditionModel, args.unet_pretrained_weight, args.model_id, "unet")
    vae = load_component(TemporalAutoencoderTiny, args.temporal_vae_pretrained_weight, args.model_id, "vae")
    scheduler = DDIMScheduler.from_pretrained(args.model_id, subfolder="scheduler")

    tensorrt_kwargs = {
        "custom_pipeline": "/acceleration/tensorrt/sd_with_controlnet_ST",
        "image_height": args.image_height,
        "image_width": args.image_width,
    } if args.enable_tensorrt else {"custom_pipeline": None}
    
    pipeline = StreamDiffVSRPipeline.from_pretrained(
        args.model_id,
        controlnet=controlnet, 
        vae=vae, 
        unet=unet, 
        scheduler=scheduler,
        **tensorrt_kwargs
    )

    if args.enable_tensorrt:
        pipeline.set_cached_folder("Jamichsu/Stream-DiffVSR")

    pipeline = pipeline.to(device)
    
    # Enable memory optimizations if requested
    if args.enable_memory_optimizations and not args.enable_tensorrt:
        print("Enabling memory optimizations...")
        pipeline.enable_xformers_memory_efficient_attention()
        pipeline.enable_vae_slicing()
        # Note: enable_model_cpu_offload() may not be compatible with this pipeline's temporal processing
        # pipeline.enable_model_cpu_offload()
    elif not args.enable_tensorrt:
        # Only enable xformers if not using TensorRT
        pipeline.enable_xformers_memory_efficient_attention()
    
    of_model = raft_large(weights=Raft_Large_Weights.DEFAULT).to(device).eval()
    of_model.requires_grad_(False) 
    
    # Check if input is a video file or directory
    if os.path.isfile(args.in_path):
        # Single video file
        video_extensions = ['.mp4', '.avi', '.mov', '.mkv', '.flv', '.wmv']
        if any(args.in_path.lower().endswith(ext) for ext in video_extensions):
            print(f"Processing video file: {args.in_path}")
            frames, fps, width, height = extract_frames_from_video(args.in_path)
            if not frames:
                print(f"Error: No frames extracted from video {args.in_path}")
                return
            
            # Generate output directory name from video filename
            video_name = os.path.splitext(os.path.basename(args.in_path))[0]
            target_path = os.path.join(args.out_path, video_name)
            os.makedirs(target_path, exist_ok=True)
            
            # Process frames in batches
            frames_hr = process_frames_in_batches(
                pipeline, frames, of_model,
                args.num_inference_steps, args.batch_size, target_path,
                of_rescale_factor=args.of_rescale_factor
            )
            
            print(f"Upscaled video and saved {len(frames_hr)} frames to {target_path}.")
            
            # Create video from frames
            output_video_path = os.path.join(args.out_path, f"{video_name}_upscaled.mp4")
            # Note: StreamDiffVSR upscales by 4x, so output dimensions are 4x input
            output_width = width * 4
            output_height = height * 4
            create_video_from_frames(
                target_path, output_video_path, fps,
                output_width, output_height, args.in_path
            )
            
            del frames
            del frames_hr
            torch.cuda.empty_cache()
        else:
            print(f"Error: Input file {args.in_path} is not a supported video format.")
            return
    else:
        # Directory structure with sequences
        seqs = sorted(os.listdir(args.in_path))
        for seq in seqs:
            seq_path = os.path.join(args.in_path, seq)
            # Skip if not a directory
            if not os.path.isdir(seq_path):
                continue
                
            frame_names = sorted(os.listdir(seq_path))
            frames = []
            for frame_name in frame_names:
                frame_path = os.path.join(seq_path, frame_name)
                frames.append(Image.open(frame_path))

            # Process frames in batches
            seq_path_obj = Path(seq_path)
            target_path = os.path.join(args.out_path, seq_path_obj.parent.name, seq_path_obj.name)
            os.makedirs(target_path, exist_ok=True)
            
            frames_hr = process_frames_in_batches(
                pipeline, frames, of_model,
                args.num_inference_steps, args.batch_size, target_path,
                of_rescale_factor=args.of_rescale_factor
            )
            
            # Save frames with original names
            for frame, name in zip(frames_hr, frame_names):
                frame.save(os.path.join(target_path, name))
            
            print(f"Upscaled {seq} and saved to {target_path}.")
            
            del frames
            del frames_hr
            torch.cuda.empty_cache()

if __name__ == "__main__":
    main()