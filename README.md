# 🎞️ VFIDiff — Motion-Aware Generative Multi-frame Interpolation

Diffusion-based **Video Frame Interpolation (VFI)** built upon **ResShift**, with **optical-flow–guided** modifications to the diffusion process.

> ✅ **Hack diffusion here:** `models/script_util.py` → `create_gaussian_diffusion()` → **`GaussianDiffusion`**

---

## 🧭 Overview

Pipeline:

**Inference / Training Script** → **ResShift Sampler** → **Gaussian Diffusion Model** → **Forward & Reverse Diffusion**

---

## 🗺️ Project Map

```text
inference / training entry
        ↓
 ResShiftSampler (sampler.py)
        ↓
 GaussianDiffusion (models/script_util.py)
        ↓
 forward: q_sample   +   reverse: p_sample*
```

---

## 🧩 Code Structure

### 🚀 Inference
- **Entry:** `inference_resshift.py`
- Creates a `ResShiftSampler`
- Calls `sampler.inference(...)`
- Sampler internally relies on a Gaussian diffusion model

### 🧠 Diffusion Model
- Built in: `sampler.py` → `build_model(...)`
- Config: `configs/realsr_swinunet_realesrgan256.yaml`
  - `target: models.script_util.create_gaussian_diffusion`
- Therefore, diffusion modifications should be made in:
  - `models/script_util.py`
    - `create_gaussian_diffusion()`
    - `GaussianDiffusion`

---

## 🔁 Diffusion Process

### ➕ Forward Diffusion (Noising)
- `GaussianDiffusion.q_sample`

### ➖ Reverse Diffusion (Sampling)
- `GaussianDiffusion.p_sample`
- `GaussianDiffusion.p_sample_loop`
- `GaussianDiffusion.p_sample_loop_progressive`

These functions define how samples are generated across diffusion timesteps.

---

## 🏋️ Training

```bash
CUDA_VISIBLE_DEVICES=0 torchrun \
  --standalone --nproc_per_node=1 --nnodes=1 \
  main.py \
  --cfg_path configs/realsr_swinunet_realesrgan256.yaml \
  --save_dir ./logs
```

---

## 🧪 Inference

```bash
python inference_resshift.py \
  -i /root/autodl-tmp/testdata/Val_SR/lq \
  -o /root/autodl-tmp/fasttest \
  --task realsr \
  --scale 4 \
  --version v3
```

---
