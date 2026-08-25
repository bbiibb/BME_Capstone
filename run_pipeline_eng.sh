#!/usr/bin/env bash
# ============================================================
#  Full pipeline runner (matches the Final Report's method)
#  Before running: pip install -r requirements.txt
#
#  Approach: LoRA patch generation + Gaussian Blending
#        (the whole-slice inpainting approach is in archive/ -- deprecated)
# ============================================================
set -e

CONFIG="configs/config.yaml"
PROPOSED_EXP="seg_proposed"
BASELINE_EXP="seg_baseline"

echo "========================================"
echo " Phase 1-1: OASIS preprocessing (normal brain slices)"
echo "========================================"
python phase1_preprocessing/preprocess_oasis.py \
    --input_dir  data/raw/oasis \
    --output_dir data/processed/normal_slices \
    --config $CONFIG

echo ""
echo "========================================"
echo " Phase 1-2: Synthetic tumor mask generation"
echo "========================================"
python phase1_preprocessing/generate_tumor_masks.py \
    --slice_dir  data/processed/normal_slices \
    --output_dir data/processed/tumor_masks \
    --config $CONFIG \
    --seed 42

echo ""
echo "========================================"
echo " Phase 4-0: BraTS slice preprocessing"
echo " (run first -- needed as LoRA training data)"
echo "========================================"
python phase4_evaluation/prepare_brats_slices.py \
    --input_dir  data/raw/brats \
    --output_dir data/processed/brats_slices \
    --config $CONFIG

echo ""
echo "========================================"
echo " Phase 2-A: LoRA Fine-tuning"
echo " (learn BraTS tumor texture on SD v1.5)"
echo "========================================"
python phase2_inpainting/finetune_lora.py \
    --brats_dir  data/processed/brats_slices \
    --output_dir checkpoints/lora_gbm \
    --config     $CONFIG

echo ""
echo "========================================"
echo " Phase 2-B: Tumor patch generation + Gaussian Blending"
echo "========================================"
python phase2_inpainting/generate_synthetic.py \
    --slice_dir  data/processed/normal_slices \
    --mask_dir   data/processed/tumor_masks \
    --output_dir data/synthetic \
    --lora_dir   checkpoints/lora_gbm/lora_final \
    --config     $CONFIG

echo ""
echo "========================================"
echo " Phase 2-C: Visualize synthetic results"
echo "========================================"
python phase2_inpainting/visualize_results.py \
    --synthetic_dir data/synthetic \
    --n_samples 8 \
    --save results/synthetic_preview.png

echo ""
echo "========================================"
echo " Phase 3-A: Baseline training (real BraTS data only, lr=1e-4 cosine+warmup)"
echo "========================================"
python phase3_segmentation/train.py \
    --synthetic_dir data/synthetic \
    --brats_dir     data/processed/brats_slices \
    --real_ratio    1.0 \
    --config        $CONFIG \
    --exp_name      $BASELINE_EXP

echo ""
echo "========================================"
echo " Phase 3-B: Proposed method (fine-tune from baseline, lr=5e-6 cosine annealing)"
echo "========================================"
python phase3_segmentation/train.py \
    --synthetic_dir data/synthetic \
    --brats_dir     data/processed/brats_slices \
    --real_ratio    0.3 \
    --config        $CONFIG \
    --exp_name      $PROPOSED_EXP \
    --finetune_from checkpoints/$BASELINE_EXP/best.pth

echo ""
echo "========================================"
echo " Phase 4-1: BraTS evaluation (baseline vs proposed)"
echo "========================================"
python phase4_evaluation/evaluate_brats.py \
    --checkpoint checkpoints/$BASELINE_EXP/best.pth \
    --brats_dir  data/processed/brats_slices \
    --config     $CONFIG \
    --output_dir results/eval_baseline

python phase4_evaluation/evaluate_brats.py \
    --checkpoint checkpoints/$PROPOSED_EXP/best.pth \
    --brats_dir  data/processed/brats_slices \
    --config     $CONFIG \
    --output_dir results/eval_proposed

echo ""
echo "Pipeline complete!"
echo "Results: results/eval_baseline, results/eval_proposed"
