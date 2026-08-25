"""
Phase 1 - Step 1: OASIS 정상 뇌 MRI 데이터 전처리

OASIS-1 / OASIS-3 데이터셋에서 2D axial 슬라이스를 추출하고
강도 정규화 후 PNG로 저장합니다.

사용법:
    python phase1_preprocessing/preprocess_oasis.py \
        --input_dir data/raw/oasis \
        --output_dir data/processed/normal_slices \
        --config configs/config.yaml
"""

import os
import sys
import argparse
import logging
from pathlib import Path

import numpy as np
import yaml
from tqdm import tqdm

sys.path.insert(0, str(Path(__file__).parent.parent))
from utils.io_utils import (
    load_nifti, save_slice_png, normalize_volume, resize_slice
)

logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)s | %(message)s")
logger = logging.getLogger(__name__)


def extract_brain_mask(volume: np.ndarray, threshold: float = 0.01) -> np.ndarray:
    """Otsu 임계값 기반 뇌 영역 마스크 추출"""
    from skimage.filters import threshold_otsu
    from skimage.morphology import binary_closing, ball
    try:
        t = threshold_otsu(volume[volume > 0])
        mask = volume > t * threshold
    except Exception:
        mask = volume > volume.max() * threshold
    # 형태학적 폐쇄 연산으로 구멍 메우기
    mask = binary_closing(mask, ball(3))
    return mask.astype(np.uint8)


def select_informative_slices(volume: np.ndarray,
                               brain_mask: np.ndarray,
                               axis: int = 2,
                               num_slices: int = 30,
                               brain_ratio_threshold: float = 0.05) -> list[int]:
    """
    뇌 조직이 충분히 있는 슬라이스만 선택.
    중앙 ±비율로 균등 샘플링.
    """
    n = volume.shape[axis]
    valid = []
    for i in range(n):
        s = np.take(brain_mask, i, axis=axis)
        if s.mean() >= brain_ratio_threshold:
            valid.append(i)

    if not valid:
        return list(range(n // 4, 3 * n // 4))

    # 유효 슬라이스 중 균등 샘플링
    step = max(1, len(valid) // num_slices)
    selected = valid[::step][:num_slices]
    return selected


def process_volume(nifti_path: str,
                   output_dir: str,
                   subject_id: str,
                   cfg: dict) -> int:
    """단일 볼륨 처리 → 2D 슬라이스 PNG 저장, 저장된 슬라이스 수 반환"""
    try:
        volume, _ = load_nifti(nifti_path)
    except Exception as e:
        logger.warning(f"Load failed [{nifti_path}]: {e}")
        return 0

    # 강도 정규화
    volume_norm = normalize_volume(volume, method=cfg["preprocessing"]["intensity_norm"])

    # 뇌 마스크
    brain_mask = extract_brain_mask(
        volume_norm, threshold=cfg["preprocessing"]["brain_threshold"]
    )

    # 슬라이스 선택
    axis = cfg["preprocessing"]["slice_axis"]
    indices = select_informative_slices(
        volume_norm, brain_mask, axis=axis,
        num_slices=cfg["preprocessing"]["num_slices_per_volume"]
    )

    target_size = tuple(cfg["preprocessing"]["target_size"])
    saved = 0
    for idx in indices:
        slice_2d = np.take(volume_norm, idx, axis=axis)
        slice_resized = resize_slice(slice_2d, target_size)

        out_path = os.path.join(output_dir, f"{subject_id}_slice{idx:04d}.png")
        save_slice_png(slice_resized, out_path)
        saved += 1

    return saved


def main():
    parser = argparse.ArgumentParser(description="OASIS 뇌 MRI 전처리")
    parser.add_argument("--input_dir",  default="data/raw/oasis")
    parser.add_argument("--output_dir", default="data/processed/normal_slices")
    parser.add_argument("--config",     default="configs/config.yaml")
    args = parser.parse_args()

    with open(args.config) as f:
        cfg = yaml.safe_load(f)

    os.makedirs(args.output_dir, exist_ok=True)

    # NIfTI 파일 검색 (OASIS 폴더 구조: OAS1_XXXX/PROCESSED/.../t88_gfc.img 또는 .nii.gz)
    nifti_files = sorted(
        list(Path(args.input_dir).rglob("*.nii")) +
        list(Path(args.input_dir).rglob("*.nii.gz"))
    )

    if not nifti_files:
        logger.error(f"NIfTI 파일을 찾을 수 없습니다: {args.input_dir}")
        logger.info("OASIS 데이터셋을 data/raw/oasis/ 에 다운로드 후 재실행하세요.")
        logger.info("OASIS-1: https://www.oasis-brains.org/ (무료 계정 필요)")
        return

    logger.info(f"총 {len(nifti_files)}개 NIfTI 파일 발견")

    total_saved = 0
    for nifti_path in tqdm(nifti_files, desc="볼륨 처리 중"):
        subject_id = nifti_path.stem.replace(".nii", "")
        n = process_volume(str(nifti_path), args.output_dir, subject_id, cfg)
        total_saved += n

    logger.info(f"전처리 완료: {total_saved}개 슬라이스 저장 → {args.output_dir}")


if __name__ == "__main__":
    main()
