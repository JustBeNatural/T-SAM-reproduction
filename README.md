# T-SAM Reproduction

This repository is a reproduction and compatibility-fixed implementation of:

**Text Embedding is Not All You Need: Attention Control for Text-to-Image Semantic Alignment with Text Self-Attention Maps**

The method is referred to here as **T-SAM**. It improves text-to-image semantic alignment by using text self-attention maps to guide cross-attention maps during inference.
The reproduction results are located in the `fix_sd` and `fix_tsam` folders; the following inputs were used for each: "An orange chair and a blue clock" and "a pair of green eyeglasses and a yellow wall clock on a table"

## Method Summary

T-SAM does **not** train Stable Diffusion again.

Instead, it performs test-time latent optimization during image generation:

1. Encode the text prompt with CLIP.
2. Extract text self-attention maps from the CLIP text encoder.
3. Extract cross-attention maps from the Stable Diffusion UNet.
4. Compute a loss between text self-attention relationships and cross-attention similarity relationships.
5. Update the current latent during denoising.
6. Decode the final latent into an image.

In short:

```text
text self-attention -> T-SAM loss -> latent update -> improved semantic alignment
```

The model weights are not updated. Only the generation-time latent is optimized.

## Tested Environment

```text
OS: Windows
Python: 3.11.15
GPU: NVIDIA GeForce RTX 5060 Laptop GPU
VRAM: 8GB
PyTorch: 2.11.0+cu128
Diffusers: 0.40.0
Transformers: 5.16.1
```

Python 3.10 or 3.11 is recommended. Python 3.14 is not recommended for this image-generation stack.

## Installation

From the project root:

```powershell
uv venv .venv --python 3.11
uv pip install --python .\.venv\Scripts\python.exe torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu128
uv pip install --python .\.venv\Scripts\python.exe -r requirements.txt
```

Optional CUDA check:

```powershell
.\.venv\Scripts\python.exe -c "import torch; print(torch.__version__); print(torch.cuda.is_available()); print(torch.cuda.get_device_name(0))"
```

## Quick Smoke Test

This command checks whether CUDA, model loading, pipeline execution, and image saving work. It disables T-SAM and runs only 2 denoising steps, so the image may look noisy.

```powershell
$env:PYTHONIOENCODING='utf-8'
.\.venv\Scripts\python.exe run.py --steps 2 --generation_dir ./generation_cli_smoke --no_update_latent --no_save_crossattn --debug_prompt
```

Output image:

```text
generation_cli_smoke/images/sd1_5x_2/
```

## Run T-SAM

Recommended T-SAM command:

```powershell
$env:PYTHONIOENCODING='utf-8'
.\.venv\Scripts\python.exe run.py --prompt "a pair of green eyeglasses and a yellow wall clock on a table, product photo" --negative_prompt "person, woman, man, human, face, body, nude, lingerie, underwear, nsfw, low quality" --steps 50 --seed 4913 --generation_dir ./fixed_tsam --no_save_crossattn --debug_tsam --debug_prompt
```

Output image:

```text
fixed_tsam/images/sd1_5x_2/
```

## Run Stable Diffusion Baseline

Use the same prompt and seed, but disable T-SAM:

```powershell
$env:PYTHONIOENCODING='utf-8'
.\.venv\Scripts\python.exe run.py --prompt "a pair of green eyeglasses and a yellow wall clock on a table, product photo" --negative_prompt "person, woman, man, human, face, body, nude, lingerie, underwear, nsfw, low quality" --steps 50 --seed 4913 --generation_dir ./fixed_sd --no_update_latent --no_save_crossattn --debug_prompt
```

Compare:

```text
fixed_tsam/images/sd1_5x_2/
fixed_sd/images/sd1_5x_2/
```

For a fair comparison, keep `prompt`, `negative_prompt`, `seed`, and `steps` the same. The only intended difference is whether `--no_update_latent` is used.

## Common Arguments

```text
--prompt                  Text prompt.
--negative_prompt         Things to avoid in the generated image.
--steps                   Number of denoising steps.
--seed                    Random seed.
--generation_dir          Output directory.
--no_update_latent        Disable T-SAM and run normal Stable Diffusion.
--max_iter_to_alter       Override how many early denoising steps use latent optimization.
--no_iterative_refinement Disable extra refinement steps.
--no_save_crossattn       Do not save cross-attention similarity maps.
--debug_tsam              Print whether T-SAM is active and show early loss values.
--debug_prompt            Print the exact prompt and output directory received by the script.
```

## Configuration

Main T-SAM parameters are in:

```text
configs/config.yaml
```

Important fields:

```yaml
guidance_scale: 7.5
max_iter_to_alter: 30
iterative_refinement_steps: [0, 10, 20]
scale_factor: 5
k: 3
update_latent: true
```

Parameter notes:

```text
update_latent
```

Enables T-SAM. If set to `false`, generation behaves like normal Stable Diffusion.

```text
max_iter_to_alter
```

Controls how many early denoising steps apply latent optimization. Larger values usually mean stronger T-SAM control, but can also reduce image stability.

```text
iterative_refinement_steps
```

Runs extra refinement at selected denoising steps. This is closer to the paper setting but slower.

```text
scale_factor
```

Controls latent update strength.

```text
k
```

Applies a power operation to text self-attention maps. The default value follows the reproduction setting used here.

## Code Structure

```text
run.py
```

Command-line entry point. It loads the config, parses arguments, loads the model, runs generation, and saves images.

```text
utils.py
```

Loads Stable Diffusion v1.5 and replaces CLIP self-attention layers with the custom attention module that records text self-attention maps.

```text
models/sd1_5/clip_sdpa_attention_x.py
```

Custom CLIP attention layer. It now matches the current Transformers CLIP attention behavior while storing attention maps for T-SAM.

```text
models/processors.py
```

Custom UNet attention processor. It records cross-attention maps.

```text
models/sd1_5/pipeline_stable_diffusion_x_2.py
```

Main Stable Diffusion pipeline with T-SAM loss and latent update logic.

```text
models/sd1_5/clip_compat.py
```

Compatibility helpers for different Transformers CLIP layouts.

## Reproduction Notes And Fixes

Several issues were fixed during reproduction.

### 1. Python Version

The system Python was 3.14, which was not a good target for this image-generation stack. A Python 3.11 virtual environment was created with `uv`.

### 2. Transformers CLIP Layout Changed

Older code expected:

```python
text_encoder.text_model.encoder.layers
```

Newer Transformers exposes:

```python
text_encoder.encoder.layers
```

This was fixed by adding:

```text
models/sd1_5/clip_compat.py
```

### 3. Custom CLIP Attention Broke Prompt Conditioning

This was the main reason earlier generations looked unrelated to the prompt.

The custom attention module did not correctly match the current Transformers CLIP attention behavior. In particular, the current CLIP text encoder passes `is_causal=True`, while the custom attention implementation initially ignored that causal behavior.

As a result, CLIP text embeddings were corrupted. A diagnostic comparison showed:

```text
custom text encoder vs original text encoder cosine similarity: about 0.165
```

This means the prompt conditioning was almost broken, causing images to ignore the prompt and sometimes produce unrelated or NSFW-like outputs.

The custom attention implementation was rewritten to preserve the original CLIP attention computation and only add attention-map recording. After the fix:

```text
custom text encoder vs original text encoder cosine similarity: 0.9999987
```

This confirms that prompt embeddings are now essentially preserved.

### 4. Iterative Refinement Threshold Was Hard-Coded

The original code only had loss thresholds for short prompts:

```python
loss_threshold = {5: ..., 6: ..., 7: ...}
```

Longer prompts caused errors such as:

```text
KeyError: 11
```

This was generalized to work with longer prompts.

### 5. Safety Checker Was Disabled

The original pipeline had the Stable Diffusion safety checker commented out. It has been restored so unsafe outputs can be filtered by the pipeline.

### 6. Added Debugging Flags

The following flags were added to make reproduction easier:

```text
--debug_tsam
--debug_prompt
--negative_prompt
--max_iter_to_alter
--no_iterative_refinement
```

These make it easier to verify whether T-SAM is active and whether the script receives the intended prompt.

## Important Reminder

For fair comparison:

```text
same prompt + same seed + same steps + T-SAM on/off
```

Changing only the seed changes the initial noise. Changing only the output folder does not change the generated content.

If two images look similar with the same seed, that can be normal. If they are exactly identical, check whether `--no_update_latent` was used and whether the output directories are different.
