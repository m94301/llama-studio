# llama-studio
WebUI for managing llama-server sessions

This tool allows you to launch various llm models onto various llama-server sessions, fiddle with and save custom configurations, and launch llama-server sessions to various GPU up the the VRAM limit.  It is intended for the use case where you want fixed models on fixed ports for interaction with other toolsets, but can be used to play around as well.

Currently tested only on NVIDIA devices.  Consider this a MVP (Minimum Viable Product) release

# Gallery
## Main Page
- GPU Panel: Shows status of all GPU, including Power, Temp, VRAM Usage, and any sessions running on this GPU with URL to llama-server's mini webUI for testing
- Model Table: Shows status of all scanned models.  Click to Load, Unload, View console Log, or edit Configuration/Launch Args
<p align="left">
<img width="512" alt="1_llama_server_main" src="https://github.com/user-attachments/assets/e9b666c9-6c32-4829-8777-2dd2a731562e" />
</p>

## Simple Setup
Just point out the path of llama-server binary and model dir.  The llama-server binary will be tested with -version and --help to extract all command line arguments, and they will be stored for easy confguration later.  The model dir will be scanned for GGUF files and they will be added to the model table for config & launching.
<p align="left">
<img width="512"  alt="1a_initial_setup" src="https://github.com/user-attachments/assets/bc59114c-f251-49ff-801d-12568205b48a" />

</p>

## GPU Selection
Clicking Load on a model allows selection of GPU.  Load as many sessions as your GPU will hold.
In the example here, Qwen 3.6 is too big to fit on GPU 0 & 1, but could load to 2 or 3.  Adjust KV quant and context to pack more models per GPU, or dedicate whole GPU to maximize context.
<p align="left">
  <img width="512" alt="2_llama_server_gpu_selection" src="https://github.com/user-attachments/assets/c57a8a69-e7ee-4217-9487-e3808a88b554" />
</p> 

## Model Configuration
Since the tool is running llama-server sessions, the config is based around launch args.  But there is a rudimentary VRAM calculator for optimizing context and KV quantization in order to pack as much context into your VRAM as you can, without enlessly failing loads.
<p align="left">
<img width="512" alt="3_model_config" src="https://github.com/user-attachments/assets/9c19e4dd-07c4-4a8d-b1ad-7a2a7652890d" />
</p>

## Option Browser
Uses the real arg list parsed from **llama-server --help** to make a clickable, searchable table of options to add to launch args.  Nice if you don't like memorizing CLI args
<p align="left">
<img width="512" alt="4_option_browser" src="https://github.com/user-attachments/assets/33c6ee23-7471-41b9-b542-f58ae2a0b0d0" />
</p>


# Quickstart
## 1. Clone the repository
git clone https://github.com/m94301/llama-studio.git

cd llama-studio

## 2. Create a virtual environment (So my python imports don't pollute your main environment)
python3 -m venv venv

source venv/bin/activate

## 3. Install dependencies
pip install -r requirements.txt

## 4. Run the app (with setup guide)
./start.sh

# Requirements (For reference)
You can see it in requirements.txt, but just to note it here.  Tried to keep it as light as possible
- fastapi==0.115.5
- uvicorn[standard]==0.30.6
- pydantic==2.10.4
- pynvml>=12.0.0
- httpx==0.28.1
- python-multipart==0.0.9
- jinja2>=3.1.0
- gguf_parser>=0.0.6

## Multi-GPU Support (tensor split)

This fork adds support for loading models across multiple GPUs using llama-server's tensor split mode.

### How it works
- In the GPU selector modal, check multiple GPUs to enable tensor split
- VRAM calculator sums across all selected GPUs to determine if the model fits
- Load button only activates when combined VRAM is sufficient
- Automatically sets `CUDA_VISIBLE_DEVICES` for all selected GPUs
- Works with `-sm tensor` in your launch args

### Example
Dual RTX 5060 Ti 16GB (32GB combined) loading Qwen3.6-35B (21GB):
- Select GPU 0 + GPU 1
- Combined available: ~30GB ✓ Fits
- Hit Load
