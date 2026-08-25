"""
Phase 1 - Step 2: 가상 종양 마스크 생성

Ho et al. (2017)의 합성 데이터 훈련 철학 계승:
  원문: "synthetic volumes with corresponding labeled volumes are automatically generated"
  → 타원체 핵(nuclei)을 대신하여 GBM 종양 특성을 반영한 불규칙 Spline 마스크를 생성

GBM 형태 모델링 (BraTS 서브영역 구조 반영):
  - 외곽 영역(라벨2): 불규칙 타원 → 부종(Edema) 인페인팅 유도
  - 내부 코어(라벨1): 편심된 소형 타원 → 괴사핵/강화종양 인페인팅 유도
  SD 프롬프트("necrotic core, enhancing rim, edema")와 구조적으로 일치

사용법:
    python phase1_preprocessing/generate_tumor_masks.py \
        --slice_dir data/processed/normal_slices \
        --output_dir data/processed/tumor_masks \
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
from scipy.interpolate import splprep, splev
from tqdm import tqdm

sys.path.insert(0, str(Path(__file__).parent.parent))
from utils.io_utils import load_png_as_float, save_mask_png, load_json, save_json

logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)s | %(message)s")
logger = logging.getLogger(__name__)


def get_brain_region(slice_2d: np.ndarray, threshold: float = 0.05) -> np.ndarray:
    """슬라이스에서 뇌 영역 마스크 추출 (두개골 제거 전제)"""
    brain = (slice_2d > threshold).astype(np.uint8)
    # 가장 큰 연결 성분만 유지
    n_labels, labels, stats, _ = cv2.connectedComponentsWithStats(brain, connectivity=8)
    if n_labels <= 1:
        return brain
    largest = 1 + np.argmax(stats[1:, cv2.CC_STAT_AREA])
    return (labels == largest).astype(np.uint8)


def generate_irregular_ellipse(center: tuple[int, int],
                                rx: int, ry: int,
                                irregularity: float = 0.5,
                                num_points: int = 12,
                                rng: np.random.Generator = None) -> np.ndarray:
    """
    불규칙한 타원형 종양 경계 생성.
    Spline 보간으로 부드럽지만 비정형적인 폐합 곡선 생성.

    Returns:
        (N, 2) 형태의 contour 좌표 배열
    """
    if rng is None:
        rng = np.random.default_rng()

    angles = np.linspace(0, 2 * np.pi, num_points, endpoint=False)
    # 반지름에 랜덤 불규칙성 추가
    radii_x = rx * (1 + irregularity * (rng.random(num_points) - 0.5))
    radii_y = ry * (1 + irregularity * (rng.random(num_points) - 0.5))

    # 회전 각도 추가 (종양이 기울어진 형태 모사)
    rotation = rng.uniform(0, np.pi)
    cos_r, sin_r = np.cos(rotation), np.sin(rotation)

    x = radii_x * np.cos(angles)
    y = radii_y * np.sin(angles)
    # 회전 변환
    x_rot = cos_r * x - sin_r * y + center[0]
    y_rot = sin_r * x + cos_r * y + center[1]

    # 폐합 곡선을 위해 시작점 복사
    x_rot = np.append(x_rot, x_rot[0])
    y_rot = np.append(y_rot, y_rot[0])

    # Spline 보간 (부드러운 경계)
    try:
        tck, _ = splprep([x_rot, y_rot], s=0, per=True, k=3)
        u_fine = np.linspace(0, 1, 200)
        x_fine, y_fine = splev(u_fine, tck)
        return np.column_stack([x_fine, y_fine]).astype(np.int32)
    except Exception:
        # Spline 실패 시 원래 포인트 사용
        return np.column_stack([x_rot, y_rot]).astype(np.int32)


def generate_tumor_mask(slice_2d: np.ndarray,
                         brain_mask: np.ndarray,
                         cfg_mask: dict,
                         rng: np.random.Generator) -> np.ndarray | None:
    """
    뇌 영역 내 랜덤 위치에 GBM 다중 영역 종양 마스크 생성.

    BraTS 라벨 구조를 반영한 동심원 설계 (Ho et al. 타원체 아이디어를 GBM으로 확장):
      - 라벨 2: 전체 종양 외곽 (부종, Edema) — 큰 불규칙 영역
      - 라벨 1: 종양 코어 (괴사핵 + 강화종양) — 전체 영역 내 ~50% 크기

    Inpainting 프롬프트("necrotic core, enhancing rim, edema")와 일치하도록
    마스크 경계가 SD에 의미있는 정보를 제공함.

    반환: uint8 이진 마스크 (종양 전체 영역=1, 배경=0)
          None: 유효한 위치를 찾지 못한 경우
    """
    h, w = slice_2d.shape
    brain_area = brain_mask.sum()
    if brain_area < h * w * 0.05:
        return None

    ys, xs = np.where(brain_mask > 0)

    # 전체 종양 크기 결정
    min_area = int(brain_area * cfg_mask["min_area_ratio"])
    max_area = int(brain_area * cfg_mask["max_area_ratio"])
    target_area = rng.integers(min_area, max_area + 1)

    aspect_ratio = rng.uniform(0.6, 1.4)
    rx_outer = int(np.sqrt(target_area / (np.pi * aspect_ratio)))
    ry_outer = int(rx_outer * aspect_ratio)
    rx_outer = max(8, rx_outer)
    ry_outer = max(8, ry_outer)

    # 중심 선택
    margin_x = rx_outer + 5
    margin_y = ry_outer + 5
    valid_indices = np.where(
        (xs > margin_x) & (xs < w - margin_x) &
        (ys > margin_y) & (ys < h - margin_y)
    )[0]
    if len(valid_indices) == 0:
        return None

    idx = rng.choice(valid_indices)
    center = (xs[idx], ys[idx])

    # ── 외곽 마스크 (전체 종양 + 부종) ──────────────────────────────────────
    outer_contour = generate_irregular_ellipse(
        center=center, rx=rx_outer, ry=ry_outer,
        irregularity=cfg_mask["irregularity"],
        num_points=cfg_mask["num_control_points"],
        rng=rng
    )
    outer_mask = np.zeros((h, w), dtype=np.uint8)
    cv2.fillPoly(outer_mask, [outer_contour], 1)
    outer_mask = outer_mask & brain_mask

    if outer_mask.sum() < min_area * 0.5:
        return None

    # ── 내부 코어 마스크 (괴사핵 + 강화종양, 약 40~60% 크기) ──────────────
    core_ratio = rng.uniform(0.4, 0.65)
    rx_inner = max(4, int(rx_outer * core_ratio))
    ry_inner = max(4, int(ry_outer * core_ratio))

    # 코어 중심을 약간 편심시켜 GBM 비대칭성 모사
    offset_x = int(rng.uniform(-rx_outer * 0.2, rx_outer * 0.2))
    offset_y = int(rng.uniform(-ry_outer * 0.2, ry_outer * 0.2))
    core_center = (center[0] + offset_x, center[1] + offset_y)

    inner_contour = generate_irregular_ellipse(
        center=core_center, rx=rx_inner, ry=ry_inner,
        irregularity=cfg_mask["irregularity"] * 0.8,  # 코어는 약간 덜 불규칙
        num_points=cfg_mask["num_control_points"],
        rng=rng
    )
    inner_mask = np.zeros((h, w), dtype=np.uint8)
    cv2.fillPoly(inner_mask, [inner_contour], 1)
    inner_mask = inner_mask & outer_mask  # 코어는 외곽 안에만

    # SD Inpainting은 단일 마스크(0/1)를 사용하므로 외곽 마스크 반환
    # 내부 구조는 SD 프롬프트("necrotic core, enhancing rim")가 처리
    return outer_mask


def process_slice(slice_path: str,
                   mask_output_dir: str,
                   cfg: dict,
                   rng: np.random.Generator) -> list[str]:
    """슬라이스 1장에서 여러 마스크 생성, 저장된 마스크 경로 리스트 반환"""
    slice_2d = load_png_as_float(slice_path)
    brain_mask = get_brain_region(slice_2d)

    stem = Path(slice_path).stem
    saved_masks = []

    for i in range(cfg["mask_generation"]["num_masks_per_image"]):
        mask = generate_tumor_mask(slice_2d, brain_mask, cfg["mask_generation"], rng)
        if mask is None:
            continue
        mask_name = f"{stem}_mask{i:02d}.png"
        mask_path = os.path.join(mask_output_dir, mask_name)
        save_mask_png(mask, mask_path)
        saved_masks.append(mask_name)

    return saved_masks


def main():
    parser = argparse.ArgumentParser(description="가상 종양 마스크 생성")
    parser.add_argument("--slice_dir",   default="data/processed/normal_slices")
    parser.add_argument("--output_dir",  default="data/processed/tumor_masks")
    parser.add_argument("--config",      default="configs/config.yaml")
    parser.add_argument("--seed",        type=int, default=42)
    args = parser.parse_args()

    with open(args.config) as f:
        cfg = yaml.safe_load(f)

    os.makedirs(args.output_dir, exist_ok=True)
    rng = np.random.default_rng(args.seed)

    slice_files = sorted(Path(args.slice_dir).glob("*.png"))
    if not slice_files:
        logger.error(f"슬라이스 PNG를 찾을 수 없습니다: {args.slice_dir}")
        logger.info("먼저 phase1_preprocessing/preprocess_oasis.py 를 실행하세요.")
        return

    logger.info(f"{len(slice_files)}개 슬라이스에서 마스크 생성 시작")

    # 슬라이스-마스크 매핑 정보
    mapping = {}
    total_masks = 0

    for slice_path in tqdm(slice_files, desc="마스크 생성 중"):
        masks = process_slice(str(slice_path), args.output_dir, cfg, rng)
        if masks:
            mapping[slice_path.name] = masks
            total_masks += len(masks)

    # 매핑 저장 (Phase 2에서 사용)
    meta_path = os.path.join(args.output_dir, "slice_mask_mapping.json")
    save_json(mapping, meta_path)

    logger.info(f"마스크 생성 완료: {total_masks}개 → {args.output_dir}")
    logger.info(f"매핑 파일: {meta_path}")


if __name__ == "__main__":
    main()
