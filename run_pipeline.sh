#!/usr/bin/env bash
# ============================================================
#  전체 파이프라인 실행 스크립트 (Final Report 방법 기준)
#  실행 전: pip install -r requirements.txt
#
#  구성: LoRA 패치 생성 + Gaussian Blending 방식
#        (전체 슬라이스 inpainting 방식은 archive/ 참고 — 폐기된 접근)
# ============================================================
set -e

CONFIG="configs/config.yaml"
PROPOSED_EXP="seg_proposed"
BASELINE_EXP="seg_baseline"

echo "========================================"
echo " Phase 1-1: OASIS 전처리 (정상 뇌 슬라이스)"
echo "========================================"
python phase1_preprocessing/preprocess_oasis.py \
    --input_dir  data/raw/oasis \
    --output_dir data/processed/normal_slices \
    --config $CONFIG

echo ""
echo "========================================"
echo " Phase 1-2: 가상 종양 마스크 생성"
echo "========================================"
python phase1_preprocessing/generate_tumor_masks.py \
    --slice_dir  data/processed/normal_slices \
    --output_dir data/processed/tumor_masks \
    --config $CONFIG \
    --seed 42

echo ""
echo "========================================"
echo " Phase 4-0: BraTS 슬라이스 전처리"
echo " (LoRA 학습 데이터로 필요하므로 먼저 실행)"
echo "========================================"
python phase4_evaluation/prepare_brats_slices.py \
    --input_dir  data/raw/brats \
    --output_dir data/processed/brats_slices \
    --config $CONFIG

echo ""
echo "========================================"
echo " Phase 2-A: LoRA Fine-tuning"
echo " (BraTS 종양 텍스처를 SD v1.5에 학습)"
echo "========================================"
python phase2_inpainting/finetune_lora.py \
    --brats_dir  data/processed/brats_slices \
    --output_dir checkpoints/lora_gbm \
    --config     $CONFIG

echo ""
echo "========================================"
echo " Phase 2-B: 종양 패치 생성 + Gaussian Blending"
echo "========================================"
python phase2_inpainting/generate_synthetic.py \
    --slice_dir  data/processed/normal_slices \
    --mask_dir   data/processed/tumor_masks \
    --output_dir data/synthetic \
    --lora_dir   checkpoints/lora_gbm/lora_final \
    --config     $CONFIG

echo ""
echo "========================================"
echo " Phase 2-C: 합성 결과 시각화"
echo "========================================"
python phase2_inpainting/visualize_results.py \
    --synthetic_dir data/synthetic \
    --n_samples 8 \
    --save results/synthetic_preview.png

echo ""
echo "========================================"
echo " Phase 3-A: 베이스라인 학습 (실제 BraTS 데이터만)"
echo "========================================"
python phase3_segmentation/train.py \
    --synthetic_dir data/synthetic \
    --brats_dir     data/processed/brats_slices \
    --real_ratio    1.0 \
    --config        $CONFIG \
    --exp_name      $BASELINE_EXP

echo ""
echo "========================================"
echo " Phase 3-B: 제안 방법 학습 (합성 + 실제 데이터 혼합)"
echo "========================================"
python phase3_segmentation/train.py \
    --synthetic_dir data/synthetic \
    --brats_dir     data/processed/brats_slices \
    --real_ratio    0.3 \
    --config        $CONFIG \
    --exp_name      $PROPOSED_EXP

echo ""
echo "========================================"
echo " Phase 4-1: BraTS 평가 (베이스라인 vs 제안 방법)"
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
echo "파이프라인 완료!"
echo "결과: results/eval_baseline, results/eval_proposed"
