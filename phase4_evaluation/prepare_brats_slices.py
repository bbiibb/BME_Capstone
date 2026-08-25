"""
Phase 4 - Step 0: BraTS 3D 볼륨 → 2D 슬라이스 추출

BraTS 데이터셋(NIfTI 4채널: t1, t1ce, t2, flair)에서
T1ce 슬라이스와 분할 마스크를 PNG로 저장합니다.

BraTS 2020/2021 다운로드:
    https://www.med.upenn.edu/cbica/brats2020/data.html (계정 필요)
    Kaggle 미러: https://www.kaggle.com/datasets/awsaf49/brats2020-training-data

사용법:
    python phase4_evaluation/prepare_brats_slices.py \
        --input_dir data/raw/brats \
        --output_dir data/processed/brats_slices \
        --config configs/config.yaml
"""

import os
import sys
import argparse
import logging
from pathlib import Path

import numpy as np
import yaml
import cv2
from tqdm import tqdm

sys.path.insert(0, str(Path(__file__).parent.parent))
from utils.io_utils import load_nifti, normalize_volume, save_slice_png, save_mask_png

logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)s | %(message)s")
logger = logging.getLogger(__name__)

# BraTS 세그멘테이션 라벨
BRATS_LABELS = {0: "background", 1: "necrotic_core", 2: "edema", 4: "enhancing_tumor"}


def find_tumor_slices(seg_volume: np.ndarray,
                       axis: int = 2,
                       min_tumor_ratio: float = 0.005) -> list[int]:
    """종양이 충분히 포함된 슬라이스 인덱스 반환"""
    n = seg_volume.shape[axis]
    tumor_slices = []
    for i in range(n):
        s = np.take(seg_volume, i, axis=axis)
        tumor_ratio = (s > 0).mean()
        if tumor_ratio >= min_tumor_ratio:
            tumor_slices.append(i)
    return tumor_slices


def process_brats_subject(subject_dir: Path,
                            img_out: str,
                            mask_out: str,
                            cfg: dict,
                            target_size: tuple) -> int:
    """BraTS 피험자 폴더 처리 → 슬라이스 저장"""
    sid = subject_dir.name

    # 파일 찾기 (BraTS 파일명 패턴)
    t1ce_files = list(subject_dir.glob("*t1ce*.nii*")) + list(subject_dir.glob("*T1CE*.nii*"))
    seg_files  = list(subject_dir.glob("*seg*.nii*"))  + list(subject_dir.glob("*SEG*.nii*"))

    if not t1ce_files or not seg_files:
        logger.warning(f"필요 파일 없음: {sid}")
        return 0

    t1ce_vol, _ = load_nifti(str(t1ce_files[0]))
    seg_vol, _  = load_nifti(str(seg_files[0]))
    seg_vol = seg_vol.astype(np.uint8)

    # 정규화
    t1ce_norm = normalize_volume(t1ce_vol, method=cfg["preprocessing"]["intensity_norm"])

    # 종양 포함 슬라이스 선택
    axis = cfg["preprocessing"]["slice_axis"]
    tumor_indices = find_tumor_slices(seg_vol, axis=axis)

    saved = 0
    for idx in tumor_indices:
        img_slice  = np.take(t1ce_norm, idx, axis=axis)
        mask_slice = np.take(seg_vol, idx, axis=axis)

        # 리사이즈
        img_resized  = cv2.resize(img_slice,  target_size[::-1], interpolation=cv2.INTER_LINEAR)
        mask_resized = cv2.resize(mask_slice, target_size[::-1], interpolation=cv2.INTER_NEAREST)

        img_name  = f"{sid}_slice{idx:04d}_t1ce.png"
        mask_name = f"{sid}_slice{idx:04d}_seg.png"

        save_slice_png(img_resized, os.path.join(img_out, img_name))

        # 마스크 저장 (라벨값 보존 — 0,1,2,4)
        from PIL import Image as _Image
        os.makedirs(mask_out, exist_ok=True)
        _Image.fromarray(mask_resized.astype(np.uint8)).save(os.path.join(mask_out, mask_name))

        saved += 1

    return saved


def main():
    parser = argparse.ArgumentParser(description="BraTS 슬라이스 전처리")
    parser.add_argument("--input_dir",  default="data/raw/brats")
    parser.add_argument("--output_dir", default="data/processed/brats_slices")
    parser.add_argument("--config",     default="configs/config.yaml")
    args = parser.parse_args()

    with open(args.config) as f:
        cfg = yaml.safe_load(f)

    target_size = tuple(cfg["preprocessing"]["target_size"])
    img_out  = os.path.join(args.output_dir, "images")
    mask_out = os.path.join(args.output_dir, "masks")
    os.makedirs(img_out, exist_ok=True)
    os.makedirs(mask_out, exist_ok=True)

    # BraTS 피험자 폴더 탐색
    input_path = Path(args.input_dir)
    subject_dirs = [d for d in input_path.iterdir() if d.is_dir()]

    if not subject_dirs:
        logger.error(f"BraTS 피험자 폴더 없음: {args.input_dir}")
        logger.info("BraTS 데이터셋을 data/raw/brats/ 에 압축 해제 후 재실행하세요.")
        return

    logger.info(f"{len(subject_dirs)}명 피험자 처리 시작")

    total = 0
    for sd in tqdm(subject_dirs, desc="BraTS 전처리"):
        n = process_brats_subject(sd, img_out, mask_out, cfg, target_size)
        total += n

    logger.info(f"완료: {total}개 슬라이스 → {args.output_dir}")


if __name__ == "__main__":
    main()
