# Stream-DiffVSR: Enhanced Fork with Memory Optimizations and Video Generation

**Forked from:** [Jamichss/Stream-DiffVSR](https://github.com/jamichss/Stream-DiffVSR)
**Enhanced with:** Memory optimizations, CUDA OOM fixes, and automatic video generation

**Authors:** Hau-Shiang Shiu, Chin-Yang Lin, Dalibor Jonic, Zhixiang Wang, Chi-Wei Hsiao, Po-Fan Yu, Yu-Chih Chen, Yu-Lun Liu

<a href='https://jamichss.github.io/stream-diffvsr-project-page/'><img src='https://img.shields.io/badge/Project-Page-Green'></a> &nbsp;
<a href="https://huggingface.co/Jamichsu/Stream-DiffVSR"><img src="https://img.shields.io/badge/%F0%9F%A4%97%20Hugging%20Face-Model%20(v1)-blue"></a> &nbsp;
<a href="https://arxiv.org/abs/2512.23709"><img src="https://img.shields.io/badge/arXiv-2510.12747-b31b1b.svg"></a>
<a href="https://github.com/dalijon-byte/video-upscaler"><img src="https://img.shields.io/badge/GitHub-Fork-6e5494"></a>

## 🚀 Enhanced Features

This fork includes significant improvements over the original Stream-DiffVSR:

- **✅ Memory Optimization**: Batch processing and VAE slicing to prevent CUDA OOM errors
- **✅ Video Generation**: Automatic MP4 creation with preserved audio from original videos
- **✅ Optical Flow Fix**: Fixed memory issues in RAFT optical flow computation
- **✅ Comprehensive Testing**: Added test utilities for video generation workflows
- **✅ Production Ready**: Enhanced for processing long videos and high-resolution content

### TODO

- ✅ Release inference code and model weights  
- ⬜ Release training code 

## Abstract
Diffusion-based video super-resolution (VSR) methods achieve strong perceptual quality but remain impractical for latency-sensitive settings due to reliance on future frames and expensive multi-step denoising. We propose Stream-DiffVSR, a causally conditioned diffusion framework for efficient online VSR. Operating strictly on past frames, it combines a four-step distilled denoiser for fast inference, an Auto-regressive Temporal Guidance (ARTG) module injecting motion-aligned cues during latent denoising, and a lightweight temporal-aware decoder with a Temporal Processor Module (TPM) enhancing detail and temporal coherence. Stream-DiffVSR processes 720p frames in 0.328 seconds on an RTX4090 GPU and significantly outperforms prior diffusion-based methods. Compared with the online SOTA TMP~\citep{zhang2024tmp}, it boosts perceptual quality (LPIPS +0.095) while reducing latency by over 130X. Stream-DiffVSR achieves the lowest latency reported for diffusion-based VSR reducing initial delay from over 4600 seconds to 0.328 seconds, thereby making it the first diffusion VSR method suitable for low-latency online deployment.

## Usage

### Environment 
The code is based on Python 3.9, CUDA 11, and [diffusers](https://github.com/huggingface/diffusers), and our development and testing are primarily conducted on Ubuntu 24.04 LTS.

### Conda setup
```
git clone https://github.com/dalijon-byte/video-upscaler.git
cd video-upscaler
conda env create -f requirements.yml
conda activate stream-diffvsr
```
Users with RTX 6000 Pro or RTX 50-series GPUs may need to update their environment by following the instructions below. For more details, please refer to [Issue #10](https://github.com/jamichss/Stream-DiffVSR/issues/10). We thank [medienbueroleipzig](https://github.com/medienbueroleipzig), [tpc2233](https://github.com/tpc2233) and [b00nwin](https://github.com/b00nwin) for testing and providing the detailed instructions!
```
## Conda setup for RTX 6000 Pro / RTX 50-Series GPUs
git clone https://github.com/dalijon-byte/video-upscaler.git
cd video-upscaler
# 1. Create conda environment
conda create --prefix ./Diff_env python==3.10 -y
# 2. Activate the environment
conda activate ./Diff_env
# 3. Install pip dependencies
pip install -r requirements-cu12.txt
pip install --upgrade transformers peft diffusers accelerate
pip install xformers==0.0.32.post2
```
### Pretrained models
Pretrained models are available [here](https://huggingface.co/Jamichsu/Stream-DiffVSR). You don't need to download them explicitly as they are fetched with inference code.
### Inference
You can run the inference directly using the following command. No manual download of checkpoints is required, as the inference script will automatically fetch the necessary files.

**Basic usage:**
```
python inference.py \
    --model_id 'Jamichsu/Stream-DiffVSR' \
    --out_path 'YOUR_OUTPUT_PATH' \
    --in_path 'YOUR_INPUT_PATH' \
    --num_inference_steps 4
```

**Enhanced usage with memory optimizations and video generation:**
```
python inference.py \
    --model_id 'Jamichsu/Stream-DiffVSR' \
    --out_path 'YOUR_OUTPUT_PATH' \
    --in_path 'YOUR_INPUT_PATH' \
    --num_inference_steps 4 \
    --batch_size 10 \
    --enable_memory_optimizations \
    --of_rescale_factor 4
```

**New command-line arguments:**
- `--batch_size`: Number of frames to process at once (default: 10, reduce if OOM)
- `--enable_memory_optimizations`: Enable memory optimizations (VAE slicing, xformers)
- `--of_rescale_factor`: Rescale factor for optical flow computation (default: 4, reduce for memory savings)

The expected file structure for the inference input data is outlined below. The model processes individual video sequences contained within subdirectories.
```
YOUR_INPUT_PATH/
├── seq1/
│   ├── frame_0001.png
│   ├── frame_0002.png
│   └── ...
├── seq2/
│   ├── frame_0001.png
│   ├── frame_0002.png
│   └── ...
```

**Video file input support:** The inference script now supports direct video file input (MP4, AVI, MOV, MKV, FLV, WMV). When a video file is provided as input, the script will:
1. Extract frames from the video
2. Upscale the frames using Stream-DiffVSR
3. Save individual upscaled frames as PNG files
4. Generate a final MP4 video file with preserved audio from the original video

For additional acceleration using NVIDIA TensorRT, please execute the following command. Please note that utilizing TensorRT may introduce a slight degradation in the output quality while providing significant performance gains. Parameters image_height and image_width are required when using tensorRT; otherwise, they are not needed.

**Note:** **TensorRT** is mainly for speed/throughput, while **xFormers** helps reduce GPU memory usage. They are currently not compatible, so xFormers-based memory optimizations are unavailable when TensorRT is enabled, which may significantly increase GPU memory usage and lead to OOM issues at higher resolutions.

```
python inference.py \
    --model_id 'Jamichsu/Stream-DiffVSR' \
    --out_path 'YOUR_OUTPUT_PATH' \
    --in_path 'YOUR_INPUT_PATH' \
    --num_inference_steps 4 \
    --enable_tensorrt \
    --image_height <YOUR_OUTPUT_HEIGHT> \
    --image_width <YOUR_OUTPUT_WIDTH>
```

When executing the TensorRT command for the first time with a new output resolution, you may observe that the process takes an extended period to build the dedicated TensorRT engine. We kindly ask for your patience. Please note that this engine compilation is a one-time setup step for that specific resolution, essential for enabling subsequent accelerated inference at the same setting.

## Memory Optimization and CUDA OOM Solutions

The inference script has been enhanced with several memory optimization features to handle CUDA out of memory (OOM) errors, especially when processing long videos or high-resolution content:

### 1. Batch Processing
- Added `--batch_size` parameter to process frames in manageable chunks
- Default batch size is 10 frames, but can be reduced for memory-constrained GPUs
- Prevents the pipeline from attempting to upscale all frames simultaneously

### 2. Memory Optimizations
- Added `--enable_memory_optimizations` flag to enable:
  - **VAE slicing**: Processes the VAE in slices to reduce memory usage
  - **xFormers memory-efficient attention**: Optimizes attention mechanism memory usage
- These optimizations are automatically disabled when TensorRT is enabled (incompatible)

### 3. Optical Flow Memory Optimization
- Added `--of_rescale_factor` parameter (default: 4)
- Reduces memory usage during RAFT optical flow computation by downsampling images before flow computation
- The `get_flow()` function in `util/flow_utils.py` has been fixed to properly apply rescaling before passing images to the RAFT model

### 4. Video Generation with Audio
- When processing video files, the script now automatically:
  - Extracts frames from the input video
  - Processes frames with memory optimizations
  - Saves individual upscaled frames as PNG files
  - Generates a final MP4 video file with preserved audio from the original video
- Requires `ffmpeg` to be installed for audio extraction and merging

### Technical Details of Fixes:
1. **CUDA OOM at line 889**: Fixed by implementing batch processing to avoid upscaling all frames at once
2. **RAFT optical flow OOM**: Fixed by adding `of_rescale_factor` parameter and correcting the `get_flow()` function to downsample images before flow computation
3. **Video output**: Added `create_video_from_frames()` function to generate final video with audio

## Accelerating the Upscaling Process

To accelerate the upscaling process, you have several options:

1. **TensorRT** (`--enable_tensorrt`): Provides significant speed/throughput gains but may reduce quality slightly
2. **Memory optimizations** (`--enable_memory_optimizations`): Allows processing larger batches or higher resolutions without OOM errors
3. **Batch size tuning** (`--batch_size`): Adjust based on your GPU memory capacity
4. **Optical flow rescaling** (`--of_rescale_factor`): Reduces memory usage during optical flow computation

**Note:** TensorRT and xFormers memory optimizations are currently incompatible. Choose based on your priority:
- For maximum speed: Use TensorRT (but be aware of potential OOM at high resolutions)
- For memory efficiency: Use xFormers memory optimizations (but without TensorRT acceleration)

## Citation

If you find this work useful, please consider citing:

```bibtex
@article{shiu2025stream,
  title={Stream-DiffVSR: Low-Latency Streamable Video Super-Resolution via Auto-Regressive Diffusion},
  author={Shiu, Hau-Shiang and Lin, Chin-Yang and Wang, Zhixiang and Hsiao, Chi-Wei and Yu, Po-Fan and Chen, Yu-Chih and Liu, Yu-Lun},
  journal={arXiv preprint arXiv:2512.23709},
  year={2025}
}
```

<!--## Acknowledgement
This project is built upon the following open-source projects: [StreamDiffusion](https://github.com/cumulo-autumn/StreamDiffusion), [StableVSR](https://github.com/claudiom4sir/StableVSR) and [TAESD](https://github.com/madebyollin/taesd). We thank all the authors for their great repos.-->
