# Synthetic Brain Tumor MRI Generation for Robust Segmentation

**Robust Brain Tumor Segmentation using Synthetic Tumor Generation via LoRA-Finetuned Stable Diffusion**

---

## Table of Contents

1. [Overview](#1-overview)
2. [Full Pipeline](#2-full-pipeline)
3. [Environment Setup](#3-environment-setup)
4. [Data Preparation](#4-data-preparation)
5. [Step-by-Step Usage](#5-step-by-step-usage)
6. [Checking Results](#6-checking-results)
7. [Troubleshooting](#7-troubleshooting)
8. [File Structure](#8-file-structure)
9. [Corrections / Notes](#9-corrections--notes)

---

## 1. Overview

We synthesize glioblastoma (GBM) tumor patches on normal brain MRI using
**LoRA-finetuned Stable Diffusion**, then blend them into normal brain
slices via **Gaussian Blending** to address data scarcity and class
imbalance. The resulting synthetic data is used to fine-tune an
**Attention U-Net** segmentation model, which is validated against real
BraTS data.

```
[BraTS tumor slices] -> [LoRA Fine-tuning] -> [SD model with learned GBM texture]
                                                        |
[Normal brain MRI] -> [Irregular mask generation] -> [LoRA tumor patch generation] -> [Gaussian Blending] -> [Synthetic tumor MRI]
                                                                                                |
                                                                                [Attention U-Net training (synthetic+real mix)]
                                                                                                |
                                                                                [Evaluation on real BraTS data]
```

> The initial approach regenerated whole slices via SD Inpainting, but this
> damaged the background brain structure. We switched to generating tumor
> patches only, then blending them in. The earlier approach's code is kept
> for reference under `archive/`.

### Related Work
- **Ho et al. (2017)**: Philosophical basis for training CNNs on synthetic
  data (fluorescence microscopy nuclei segmentation)
- **Wyatt et al. (2022, AnoDDPM)**: Simplex noise-based brain anomaly
  detection

---

## 2. Full Pipeline

```
Phase 1-1  OASIS preprocessing   Normal brain 3D NIfTI -> 2D slice PNGs
Phase 1-2  Mask generation       Irregular GBM masks via Cubic Spline + Elastic Deformation
Phase 4-0  BraTS preprocessing   Convert real tumor data to 2D slices (run first; needed for LoRA training)
Phase 2-A  LoRA Fine-tuning      Learn BraTS tumor texture on SD v1.5
Phase 2-B  Tumor patch synthesis Generate 128x128 patches with LoRA SD + insert into OASIS via Gaussian Blending
Phase 3-A  Baseline training     Train Attention U-Net on real BraTS data only
Phase 3-B  Proposed training     Train on synthetic (70%) + real (30%) mixed data
Phase 4-1  Evaluation            Compute Dice / IoU / HD95
```

> An ablation study (component-removal comparison) was attempted but never
> completed successfully (see `archive/ablation_study_incomplete.py`). It
> has been removed from the pipeline and README. The Final Report's
> results include only the baseline-vs-proposed comparison (Table 4).

---

## 3. Environment Setup

### 3-1. Check Python version

```bash
python3 --version  # 3.9 or higher required
```

### 3-2. Create a virtual environment (recommended)

```bash
python3 -m venv venv
source venv/bin/activate        # Mac / Linux
# venv\Scripts\activate         # Windows
```

### 3-3. Install packages

**Mac (Apple Silicon M1/M2/M3) -- using PyTorch MPS:**
```bash
pip install --upgrade pip

# Install PyTorch separately (MPS-enabled build)
pip install torch torchvision

# Remaining packages
pip install -r requirements.txt
```

**CUDA GPU (Linux/Windows):**
```bash
pip install --upgrade pip

# PyTorch CUDA build (CUDA 12.1 shown; check pytorch.org for your version)
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu121

# Remaining packages
pip install -r requirements.txt
```

> **Apple Silicon note:**
> The MPS backend is enabled automatically. Stable Diffusion runs roughly
> 3-5x slower than on CUDA.

> **If you have a CUDA GPU:**
> Set `device: cuda` in `configs/config.yaml` (this is already the
> default).

---

## 4. Data Preparation

### 4-1. OASIS dataset (normal brains, required for Phase 1)

1. Create a free account at [https://www.oasis-brains.org](https://www.oasis-brains.org)
2. Download **OASIS-1** (you can use a subset of disc1-disc12)
3. Extract and place under:

```
data/raw/oasis/
├── OAS1_0001_MR1/
│   └── PROCESSED/MPRAGE/T88_111/
│       └── OAS1_0001_MR1_mpr_n4_anon_111_t88_gfc.nii.gz
├── OAS1_0002_MR1/
│   └── ...
```

> OASIS files are auto-discovered as long as they are `.nii.gz` or `.nii`,
> regardless of exact location.

### 4-2. BraTS dataset (real tumors, required for Phase 2/4)

**Option A -- Kaggle (simplest):**

```bash
pip install kaggle
# Set up API key at ~/.kaggle/kaggle.json, then:
kaggle datasets download -d awsaf49/brats2020-training-data
unzip brats2020-training-data.zip -d data/raw/brats/
```

**Option B -- Official site:**
[https://www.med.upenn.edu/cbica/brats2020/data.html](https://www.med.upenn.edu/cbica/brats2020/data.html)
(institutional email required)

After extraction:

```
data/raw/brats/
├── BraTS20_Training_001/
│   ├── BraTS20_Training_001_t1ce.nii.gz
│   ├── BraTS20_Training_001_t2.nii.gz
│   ├── BraTS20_Training_001_flair.nii.gz
│   └── BraTS20_Training_001_seg.nii.gz
├── BraTS20_Training_002/
│   └── ...
```

> This repository contains no data files. Both OASIS-1 and BraTS 2020
> must be downloaded/registered for separately under their own licenses.

---

## 5. Step-by-Step Usage

### Run everything automatically (in order)

```bash
bash run_pipeline.sh
```

---

### Phase 1-1: OASIS preprocessing

```bash
python phase1_preprocessing/preprocess_oasis.py \
    --input_dir  data/raw/oasis \
    --output_dir data/processed/normal_slices \
    --config     configs/config.yaml
```

**Output:** `data/processed/normal_slices/*.png` (~30 slices per volume)

---

### Phase 1-2: Irregular tumor mask generation

Generates masks that mimic the invasive boundary of real GBM tumors using
Cubic Spline interpolation and Elastic Deformation.

```bash
python phase1_preprocessing/generate_tumor_masks.py \
    --slice_dir  data/processed/normal_slices \
    --output_dir data/processed/tumor_masks \
    --config     configs/config.yaml \
    --seed       42
```

**Output:** `data/processed/tumor_masks/*.png`, `slice_mask_mapping.json`

---

### Phase 4-0: BraTS slice preprocessing (run first)

BraTS data is needed for LoRA training, so this must run before Phase 2.

```bash
python phase4_evaluation/prepare_brats_slices.py \
    --input_dir  data/raw/brats \
    --output_dir data/processed/brats_slices \
    --config     configs/config.yaml
```

---

### Phase 2-A: LoRA Fine-tuning -- learning tumor texture from BraTS

Fine-tunes SD v1.5 with LoRA on BraTS tumor bounding-box patches so it
learns to generate GBM-specific texture (enhancing rim, necrotic core,
edema) conditioned on the mask.

```bash
python phase2_inpainting/finetune_lora.py \
    --brats_dir  data/processed/brats_slices \
    --output_dir checkpoints/lora_gbm \
    --config     configs/config.yaml
```

- If you run out of GPU memory, lower `lora_finetuning.batch_size` to `1`
  in `config.yaml`.

**Output:** `checkpoints/lora_gbm/lora_final/` (LoRA weights + MaskEncoder
weights)

---

### Phase 2-B: Tumor patch generation + Gaussian Blending

```bash
python phase2_inpainting/generate_synthetic.py \
    --slice_dir  data/processed/normal_slices \
    --mask_dir   data/processed/tumor_masks \
    --output_dir data/synthetic \
    --lora_dir   checkpoints/lora_gbm/lora_final \
    --config     configs/config.yaml
```

Small test run (a limited number of images first):

```bash
python phase2_inpainting/generate_synthetic.py \
    --max_images 100 \
    --slice_dir  data/processed/normal_slices \
    --mask_dir   data/processed/tumor_masks \
    --output_dir data/synthetic \
    --lora_dir   checkpoints/lora_gbm/lora_final \
    --config     configs/config.yaml
```

**Output:**
- `data/synthetic/*_synthetic.png` -- synthetic tumor MRI
- `data/synthetic/gt_masks/` -- corresponding ground-truth masks

Visualize results:

```bash
python phase2_inpainting/visualize_results.py \
    --synthetic_dir data/synthetic \
    --n_samples     8 \
    --save          results/synthetic_preview.png
```

---

### Phase 3: Segmentation model training

Baseline (real BraTS data only, lr=1e-4 with cosine annealing + warmup):

```bash
python phase3_segmentation/train.py \
    --synthetic_dir data/synthetic \
    --brats_dir     data/processed/brats_slices \
    --real_ratio    1.0 \
    --config        configs/config.yaml \
    --exp_name      seg_baseline
```

Proposed method (fine-tune from the baseline checkpoint on 70% synthetic +
30% real data, lr=5e-6 with plain cosine annealing, no warmup -- per
Final Report Table 1):

```bash
python phase3_segmentation/train.py \
    --synthetic_dir data/synthetic \
    --brats_dir     data/processed/brats_slices \
    --real_ratio    0.3 \
    --config        configs/config.yaml \
    --exp_name      seg_proposed \
    --finetune_from checkpoints/seg_baseline/best.pth
```

Resume a previous run (same experiment, restores optimizer/scheduler state):

```bash
python phase3_segmentation/train.py \
    --exp_name seg_proposed \
    --resume   checkpoints/seg_proposed/last.pth
```

**Output:** `checkpoints/<exp_name>/best.pth`

---

### Phase 4-1: BraTS evaluation

```bash
python phase4_evaluation/evaluate_brats.py \
    --checkpoint checkpoints/seg_proposed/best.pth \
    --brats_dir  data/processed/brats_slices \
    --config     configs/config.yaml \
    --output_dir results/eval_proposed
```

**Output:**
- `results/eval_proposed/evaluation_results.json` -- Dice/IoU/HD95 scores
- `results/eval_proposed/qualitative_results.png` -- prediction visualization
- `results/eval_proposed/metrics_distribution.png` -- Dice distribution plot

---

## 6. Checking Results

```
results/
├── synthetic_preview.png        <- preview of LoRA-generated synthetic data
├── eval_baseline/
└── eval_proposed/
    ├── evaluation_results.json  <- Dice, IoU, HD95 (Final Report Table 4)
    ├── qualitative_results.png
    └── metrics_distribution.png
```

---

## 7. Troubleshooting

**Q. `FileNotFoundError: metadata.json not found`**
You likely skipped Phase 1-2. Check the execution order.

**Q. `CUDA out of memory`**
Reduce `batch_size` in `configs/config.yaml`.

**Q. SD model download is very slow**
This only happens once. Hugging Face cache location:
`~/.cache/huggingface/hub/`

**Q. OASIS / BraTS NIfTI files not found**
Make sure files are `.nii` or `.nii.gz` and located under `data/raw/`.

**Q. Errors on Apple Silicon (MPS)**
```bash
# In config.yaml, set:
device: cpu
```

---

## 8. File Structure

```
.
├── README.md
├── requirements.txt
├── run_pipeline.sh                    <- runs the full pipeline in order
├── configs/
│   └── config.yaml                    <- all hyperparameters
├── utils/
│   ├── io_utils.py                    <- NIfTI/PNG I/O, normalization
│   └── metrics.py                     <- Dice, IoU, Hausdorff
├── phase1_preprocessing/
│   ├── preprocess_oasis.py            <- 3D -> 2D slice extraction
│   └── generate_tumor_masks.py        <- irregular GBM mask generation
├── phase2_inpainting/
│   ├── finetune_lora.py               <- LoRA fine-tuning (patch generation training)
│   ├── generate_synthetic.py          <- LoRA patch generation + Gaussian Blending
│   └── visualize_results.py           <- synthetic result visualization
├── phase3_segmentation/
│   ├── dataset.py                     <- synthetic/BraTS/mixed Dataset classes
│   ├── unet.py                        <- Attention U-Net model
│   └── train.py                       <- training loop
├── phase4_evaluation/
│   ├── prepare_brats_slices.py        <- BraTS 3D -> 2D preprocessing
│   └── evaluate_brats.py              <- quantitative/qualitative evaluation
├── notebooks/
│   └── colab_pipeline.ipynb           <- the actual Colab notebook that produced the Final Report results
├── archive/                            <- deprecated/incomplete earlier versions (for reference)
│   ├── inpaint_pipeline.py             │ full-slice SD Inpainting approach
│   ├── inpaint_pipeline_v1.py          │ (discarded due to background structure damage)
│   ├── finetune_lora_v1_inpainting.py  │
│   ├── generate_tumor_masks_v1.py      │ simple ellipse masks (early version)
│   ├── ablation_study_incomplete.py    │ never completed due to a runtime error
│   ├── config_v5.yaml                  │ intermediate development config
│   └── config_ver1.yaml                │ initial config
├── data/                                (download separately -- not included)
├── checkpoints/                         (trained models -- not included)
└── results/                             (evaluation results and figures)
```

---

## 9. Corrections / Notes

The following report-vs-code discrepancies were found and resolved during
a final review before publishing:

- **`lora_finetuning.num_epochs`**: the config originally had 20; this was
  corrected to 15 to match Final Report Table 1.
- **`mask_generation.min_area_ratio` / `max_area_ratio`** (2.5%-12%): the
  Final Report text states "10-25% of brain area," but this appears to be
  a writing error in the paper. The mask-generation range actually used
  was the config's 2.5%-12%, which was kept as-is.
- **`lora_finetuning.image_size`**: originally set to 256, which conflicted
  with the hardcoded 128 patch size at inference time in
  `generate_synthetic.py`. Fixed to 128 to match Final Report Table 1 (and
  consistent with the paper's own key finding that train-inference
  consistency matters).
- **Ablation Study**: an ablation experiment was attempted but never
  completed (see `archive/ablation_study_incomplete.py`). This step has
  been removed from the pipeline and README. The Final Report's results
  include only the baseline-vs-proposed comparison.
- **Two-stage learning rate / scheduler**: the Final Report specifies a
  baseline learning rate of 1e-4 (cosine annealing with warmup) and a
  separate fine-tuning learning rate of 5e-6 (plain cosine annealing, no
  warmup) for the proposed model, which resumes from the baseline
  checkpoint. `train.py` now implements this explicitly via a
  `--finetune_from` flag: passing a baseline checkpoint path loads its
  weights only (optimizer/scheduler state is reinitialized) and switches
  to `finetune_learning_rate` / `finetune_num_epochs` / a
  warmup-free cosine scheduler, all read from `config.yaml`. `run_pipeline.sh`
  was updated so Phase 3-B calls Phase 3-A's checkpoint via `--finetune_from`
  instead of training from scratch.
- **Which notebook produced the reported results**: among several Colab
  notebook versions, `notebooks/colab_pipeline.ipynb` was confirmed to be
  the one that actually printed the exact numbers in Final Report Table 4
  (baseline Dice 0.8727; proposed method Dice 0.8773 / IoU 0.7918 / HD95
  11.66 at epoch 16), so only this version was kept.
- **Open question**: the "control experiment" numbers in Table 4 (BraTS
  +300 real slices only, Dice 0.8768 / IoU 0.7916 / HD95 11.64) could not
  be traced to an exact match in any of the notebooks reviewed so far.

## Author

Siun Lim, Dept. of Biomedical Engineering, Korea University
