# T-SAM 复现说明

这份代码对应论文 `Text Embedding is Not All You Need: Attention Control for Text-to-Image Semantic Alignment with Text Self-Attention Maps`。

## 论文思路一句话版

Stable Diffusion 会用文本 token 的 cross-attention map 决定图像里哪些区域响应哪些词。论文发现：仅靠 text embedding 不一定能表达好语法绑定关系，比如 `black car` 和 `white clock` 里，颜色应该和对应物体绑定。T-SAM 的做法是不训练模型，而是在采样时用文本编码器里的 self-attention map 当语法关系参考，优化 latent，让图像 cross-attention map 的 token 相似关系更接近文本 self-attention 关系。

## 已验证环境

- Windows
- GPU: NVIDIA GeForce RTX 5060 Laptop GPU, 8GB VRAM
- Python: 3.11.15
- PyTorch: `2.11.0+cu128`
- CUDA 可用性已验证为 `True`

## 第一次配置环境

在项目根目录执行：

```powershell
uv venv .venv --python 3.11
uv pip install --python .\.venv\Scripts\python.exe torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu128
uv pip install --python .\.venv\Scripts\python.exe -r requirements.txt
```

说明：不要用系统默认的 Python 3.14 跑这个项目。图像生成依赖栈对 Python 3.10/3.11 更稳。

## 先跑最小 demo

这个命令只跑 2 个 denoising step，关闭 T-SAM latent 优化，也不保存 cross-attention 相似度图。它的作用是检查模型下载、CUDA、pipeline 和图片保存是否正常。

```powershell
$env:PYTHONIOENCODING='utf-8'
.\.venv\Scripts\python.exe run.py --steps 2 --generation_dir ./generation_cli_smoke --no_update_latent --no_save_crossattn
```

输出图片：

```text
generation_cli_smoke/images/sd1_5x_2/
```

文本 self-attention 平均图：

```text
generation_cli_smoke/text_sa/sd1_5x_2/
```

## 跑一个短版 T-SAM

这个命令会启用论文的 attention control，但仍然只跑少量步数，适合检查梯度优化路径。

```powershell
$env:PYTHONIOENCODING='utf-8'
.\.venv\Scripts\python.exe run.py --prompt "a green glasses and a yellow clock" --steps 5 --max_iter_to_alter 2 --no_iterative_refinement --generation_dir ./generation_short_tsam --no_save_crossattn
```

## 跑接近默认设置的完整实验

默认配置在 `configs/config.yaml`：

- `update_latent: true` 表示启用 T-SAM
- `max_iter_to_alter: 30` 表示前 30 个 denoising step 做 latent 更新
- `iterative_refinement_steps: [0, 10, 20]` 表示这些步额外迭代 refine
- `k: 3` 对文本 self-attention 权重做幂次放大

完整运行：

```powershell
$env:PYTHONIOENCODING='utf-8'
.\.venv\Scripts\python.exe run.py --prompt "a green glasses and a yellow clock" --steps 50 --generation_dir ./generation_dir
```

完整设置会比 smoke test 慢很多，也更吃显存。如果显存不够，先加 `--no_save_crossattn`，或者把 `configs/config.yaml` 里的 `max_iter_to_alter` 改小。

## 代码里主要看哪里

- `run.py`: 命令行入口，负责读配置、加载模型、保存图片和 attention 图。
- `utils.py`: `load_model()` 会加载 SD 1.5，并把 CLIP self-attention 换成可记录 attention map 的自定义层。
- `models/sd1_5/clip_sdpa_attention_x.py`: 记录文本 self-attention map。
- `models/processors.py`: 记录 UNet cross-attention map。
- `models/sd1_5/pipeline_stable_diffusion_x_2.py`: T-SAM loss 和 latent 更新逻辑。

## 我已做的兼容修复

- 增加 `models/sd1_5/clip_compat.py`，兼容 Transformers 4.x/5.x 的 CLIP 结构差异。
- 修复新版 Transformers 给 CLIP attention 多传 `is_causal` 参数的问题。
- 修复 `attention_mask=None` 时自定义 attention 报错的问题。
- 修复关闭 latent 优化时 pipeline 仍进入 refinement 导致 `loss` 未初始化的问题。
- 给 `run.py` 增加 `--no_update_latent`、`--no_save_crossattn` 等入门开关。
- 让短步数运行时保存 cross-attention 图不会索引不存在的 timestep。
