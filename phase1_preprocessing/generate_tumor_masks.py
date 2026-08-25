"""
Phase 1-2 (v3): 3D 타원체 기반 종양 마스크 생성
────────────────────────────────────────────────
v2 대비 변경사항:
  - 슬라이스 독립 2D 마스크 → 볼륨 단위 3D 타원체 단면 투영
  - 같은 sid(볼륨) 슬라이스를 그룹핑하여 3D 공간적 연속성 확보
  - 파일명: {sid}_z{z_idx:03d}_e{e_idx}_mask_*.png

v2에서 유지되는 것:
  ✅ Cubic Spline 보간 (10개 제어점, irregularity 가변)
  ✅ Elastic Deformation (Gaussian displacement field, 보고서 식 2)
  ✅ 에지 밀도 기반 필터 (동심원 슬라이스 제거)
  ✅ 뇌 경계 margin (erode 기반, 외각 배치 방지)
  ✅ 마스크 겹침 방지 (타원체 간 IoU > overlap_threshold 시 재샘플링)
  ✅ z위치에 따른 irregularity 가변 (중심에서 멀수록 경계 더 불규칙)
"""

import argparse
import json
import re
from collections import defaultdict
from pathlib import Path

import cv2
import numpy as np
import yaml
from PIL import Image
from scipy.interpolate import CubicSpline
from scipy.ndimage import gaussian_filter
from tqdm import tqdm


# ─────────────────────────────────────────────────────────────
# 설정 로드
# ─────────────────────────────────────────────────────────────

def load_config(config_path: str) -> dict:
    with open(config_path) as f:
        return yaml.safe_load(f)


def load_slice(path: str) -> np.ndarray:
    """PNG 슬라이스를 [0,1] float32로 로드"""
    return np.array(Image.open(path).convert("L"), dtype=np.float32) / 255.0


# ─────────────────────────────────────────────────────────────
# 슬라이스 그룹핑
# ─────────────────────────────────────────────────────────────

def extract_sid_and_zidx(filename: str):
    """
    파일명에서 볼륨 ID(sid)와 슬라이스 인덱스(z_idx)를 분리.
    예: OAS1_0001_MR1_mpr-1_slice0040.png
        → sid='OAS1_0001_MR1_mpr-1', z_idx=40
    """
    stem = Path(filename).stem
    m = re.match(r'^(.+)_slice(\d+)$', stem)
    if m:
        return m.group(1), int(m.group(2))
    # fallback: 파일명 전체를 sid로, z_idx=0
    return stem, 0


def group_slices_by_volume(slice_paths: list) -> dict:
    """
    슬라이스 경로 목록을 볼륨(sid) 기준으로 그룹핑.
    반환: {sid: [(z_idx, path), ...]} — z_idx 오름차순 정렬
    """
    groups = defaultdict(list)
    for p in slice_paths:
        sid, z_idx = extract_sid_and_zidx(p.name)
        groups[sid].append((z_idx, p))
    # z_idx 기준 정렬
    return {sid: sorted(slices, key=lambda x: x[0])
            for sid, slices in groups.items()}


# ─────────────────────────────────────────────────────────────
# 슬라이스 품질 필터 (v2와 동일)
# ─────────────────────────────────────────────────────────────

def extract_brain_mask(img: np.ndarray) -> np.ndarray:
    """Otsu 임계값 기반 뇌 영역 이진 마스크 추출"""
    uint8 = (img * 255).astype(np.uint8)
    _, brain = cv2.threshold(uint8, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (7, 7))
    brain = cv2.morphologyEx(brain, cv2.MORPH_CLOSE, kernel)
    brain = cv2.morphologyEx(brain, cv2.MORPH_OPEN, kernel)
    return brain.astype(bool)


def is_valid_slice(img: np.ndarray,
                   brain_min: float = 0.08,
                   brain_max: float = 0.65,
                   center_min: float = 0.10,
                   edge_max: float = 0.20) -> bool:
    """슬라이스 품질 필터 (동심원 패턴, 빈 슬라이스 제거)
    
    [수정] edge_max 0.12 → 0.20
    OASIS 정상 뇌 슬라이스의 실측 edge_ratio 평균이 0.136 이므로
    0.12는 정상 슬라이스까지 걸러냄. 실제 동심원 아티팩트는
    보통 0.25 이상이므로 0.20으로 완화.
    """
    h, w = img.shape
    brain_ratio = (img > 0.08).mean()
    if not (brain_min <= brain_ratio <= brain_max):
        return False
    cy, cx = h // 2, w // 2
    center_patch = img[cy - h//8: cy + h//8, cx - w//8: cx + w//8]
    if center_patch.mean() < center_min:
        return False
    edges = cv2.Canny((img * 255).astype(np.uint8), 30, 100)
    if edges.mean() / 255.0 > edge_max:
        return False
    return True


# ─────────────────────────────────────────────────────────────
# 3D 타원체 정의 및 단면 계산
# ─────────────────────────────────────────────────────────────

class Ellipsoid3D:
    """
    3D 타원체: 중심(cy, cx, cz)과 각 축 반지름(ry, rx, rz).
    z축 = 슬라이스 방향 (axial 기준).

    슬라이스 z에서의 단면은 타원:
        (y - cy)²/ry_z² + (x - cx)²/rx_z² ≤ 1
    여기서 ry_z = ry * sqrt(1 - ((z-cz)/rz)²)
               rx_z = rx * sqrt(1 - ((z-cz)/rz)²)
    """

    def __init__(self, cy: float, cx: float, cz: float,
                 ry: float, rx: float, rz: float):
        self.cy, self.cx, self.cz = cy, cx, cz
        self.ry, self.rx, self.rz = ry, rx, rz

    def slice_radius(self, z: int):
        """
        z 슬라이스에서의 단면 타원 반지름 (ry_z, rx_z).
        z가 타원체 범위 밖이면 None 반환.
        """
        dz = (z - self.cz) / (self.rz + 1e-8)
        if abs(dz) >= 1.0:
            return None
        scale = np.sqrt(max(0.0, 1.0 - dz ** 2))
        return self.ry * scale, self.rx * scale

    def z_range(self):
        """타원체가 존재하는 z 슬라이스 범위"""
        return int(np.floor(self.cz - self.rz)), int(np.ceil(self.cz + self.rz))


def sample_ellipsoid(brain_mask: np.ndarray,
                     n_slices: int,
                     area_ratio_range: tuple,
                     rng: np.random.Generator) -> 'Ellipsoid3D':
    """
    뇌 마스크 내부에 3D 타원체 파라미터를 샘플링.

    brain_mask : 대표 슬라이스의 뇌 마스크 (위치 추정용)
    n_slices   : 볼륨의 전체 슬라이스 수 (z축 범위 제한용)
    """
    h, w = brain_mask.shape
    brain_area = brain_mask.sum()

    # 종양 크기 (뇌 면적 대비)
    target_ratio = rng.uniform(*area_ratio_range)
    target_area  = brain_area * target_ratio
    r_base       = np.sqrt(target_area / np.pi)

    # 각 축 반지름에 약간의 비대칭 적용 (실제 GBM은 완전한 구형이 아님)
    ry = r_base * rng.uniform(0.8, 1.2)
    rx = r_base * rng.uniform(0.8, 1.2)
    # z축(슬라이스 방향) 반지름: 3~8 슬라이스 범위
    rz = rng.uniform(3, min(8, n_slices * 0.15))

    # 중심 좌표 — 뇌 내부에 margin 확보
    margin = int(min(h, w) * 0.15)
    kernel = cv2.getStructuringElement(
        cv2.MORPH_ELLIPSE, (margin * 2 + 1, margin * 2 + 1))
    eroded = cv2.erode(brain_mask.astype(np.uint8), kernel)
    ys, xs = np.where(eroded > 0)
    if len(ys) == 0:
        ys, xs = np.where(brain_mask)
    idx = rng.integers(len(ys))
    cy, cx = float(ys[idx]), float(xs[idx])

    # z 중심: 볼륨 중간 30~70% 구간
    cz = rng.uniform(n_slices * 0.30, n_slices * 0.70)

    return Ellipsoid3D(cy, cx, cz, ry, rx, rz)


# ─────────────────────────────────────────────────────────────
# 단면 마스크 생성 (Spline + Elastic — v2와 동일 로직 유지)
# ─────────────────────────────────────────────────────────────

def generate_spline_contour(center: tuple,
                             ry: float, rx: float,
                             n_control: int = 10,
                             irregularity: float = 0.45,
                             rng: np.random.Generator = None) -> np.ndarray:
    """
    비원형 타원 단면에 Cubic Spline으로 불규칙 경계 생성.
    ry, rx: 타원의 y/x 반지름
    """
    if rng is None:
        rng = np.random.default_rng()

    cy, cx = center
    angles = np.linspace(0, 2 * np.pi, n_control, endpoint=False)

    # 각도별 타원 반지름 계산 후 불규칙 변동 추가
    r_ellipse = (ry * rx) / np.sqrt(
        (rx * np.cos(angles)) ** 2 + (ry * np.sin(angles)) ** 2 + 1e-8)
    radii = r_ellipse * (1 + irregularity * (rng.random(n_control) - 0.5) * 2)
    radii = np.clip(radii, r_ellipse * 0.4, r_ellipse * 1.8)

    angles_ext = np.append(angles, angles[0] + 2 * np.pi)
    radii_ext  = np.append(radii, radii[0])
    cs = CubicSpline(angles_ext, radii_ext, bc_type='periodic')

    t = np.linspace(0, 2 * np.pi, 360, endpoint=False)
    r = np.clip(cs(t), 1.0, None)

    ys = (cy + r * np.sin(t)).astype(int)
    xs = (cx + r * np.cos(t)).astype(int)
    return np.stack([ys, xs], axis=1)


def contour_to_mask(contour: np.ndarray, shape: tuple) -> np.ndarray:
    mask = np.zeros(shape, dtype=np.uint8)
    pts  = contour[:, [1, 0]].reshape(-1, 1, 2)
    cv2.fillPoly(mask, [pts], 1)
    return mask.astype(bool)


def apply_elastic_deformation(mask: np.ndarray,
                               alpha: float = 14.0,
                               sigma: float = 4.0,
                               rng: np.random.Generator = None) -> np.ndarray:
    """Gaussian smoothed displacement field로 마스크 탄성 변형 (보고서 식 2)"""
    if rng is None:
        rng = np.random.default_rng()
    h, w = mask.shape
    dx = gaussian_filter(rng.uniform(-1, 1, (h, w)), sigma) * alpha
    dy = gaussian_filter(rng.uniform(-1, 1, (h, w)), sigma) * alpha
    x, y = np.meshgrid(np.arange(w), np.arange(h))
    map_x = np.clip(x + dx, 0, w - 1).astype(np.float32)
    map_y = np.clip(y + dy, 0, h - 1).astype(np.float32)
    deformed = cv2.remap(mask.astype(np.float32), map_x, map_y,
                         interpolation=cv2.INTER_LINEAR)
    return deformed > 0.5


def masks_overlap(m1: np.ndarray, m2: np.ndarray,
                  threshold: float = 0.3) -> bool:
    """
    두 마스크의 겹침 비율(IoU 기준)이 threshold 이상이면 True.
    타원체 배치 시 겹침 방지에 사용.
    """
    inter = (m1 & m2).sum()
    union = min(m1.sum(), m2.sum())   # 작은 쪽 기준 (더 엄격)
    if union == 0:
        return False
    return (inter / union) > threshold


def ellipsoid_slice_to_mask(ellipsoid: 'Ellipsoid3D',
                             z: int,
                             img_shape: tuple,
                             brain_mask: np.ndarray,
                             n_control: int,
                             irregularity: float,
                             alpha: float,
                             sigma: float,
                             core_ratio_range: tuple,
                             rng: np.random.Generator) -> dict | None:
    """
    3D 타원체의 z 슬라이스 단면을 마스크로 변환.
    반환: {'outer': mask, 'core': mask} or None
    """
    radii = ellipsoid.slice_radius(z)
    if radii is None:
        return None   # 이 슬라이스에서 타원체 단면 없음

    ry_z, rx_z = radii
    if min(ry_z, rx_z) < 3:
        return None   # 단면이 너무 작음

    center = (ellipsoid.cy, ellipsoid.cx)

    # ── z위치에 따른 irregularity 가변 ───────────────
    # 타원체 중심(cz)에서 멀수록 단면이 작고 경계가 더 불규칙해짐
    # dz_norm: 0(중심) ~ 1(끝단), 끝단에서 irregularity 최대 1.4배
    dz_norm = abs(z - ellipsoid.cz) / (ellipsoid.rz + 1e-8)
    dz_norm = float(np.clip(dz_norm, 0.0, 1.0))
    irr_z   = irregularity * (1.0 + 0.4 * dz_norm)   # 최대 irregularity*1.4

    # ── 외곽 마스크 (Spline + Elastic) ───────────────
    outer_contour = generate_spline_contour(
        center, ry_z, rx_z, n_control, irr_z, rng)
    outer_mask = contour_to_mask(outer_contour, img_shape)
    outer_mask = apply_elastic_deformation(outer_mask, alpha, sigma, rng)
    outer_mask = outer_mask & brain_mask   # 뇌 영역 밖 제거

    if outer_mask.sum() < 50:
        return None

    # ── 내부 코어 마스크 ─────────────────────────────
    core_ratio  = rng.uniform(*core_ratio_range)
    core_ry     = ry_z * core_ratio
    core_rx     = rx_z * core_ratio
    # 편심 배치
    r_avg = (ry_z + rx_z) / 2
    offset_y = int(rng.uniform(-r_avg * 0.2, r_avg * 0.2))
    offset_x = int(rng.uniform(-r_avg * 0.2, r_avg * 0.2))
    core_center = (center[0] + offset_y, center[1] + offset_x)

    core_contour = generate_spline_contour(
        core_center, core_ry, core_rx,
        n_control, irr_z * 0.8, rng)
    core_mask = contour_to_mask(core_contour, img_shape)
    core_mask = apply_elastic_deformation(core_mask, alpha * 0.7, sigma, rng)
    core_mask = core_mask & outer_mask    # 반드시 외곽 내부

    return {"outer": outer_mask, "core": core_mask}


# ─────────────────────────────────────────────────────────────
# 메인 파이프라인
# ─────────────────────────────────────────────────────────────

def process_volumes(slice_dir: str,
                    output_dir: str,
                    cfg: dict,
                    seed: int = 42):
    """
    볼륨 단위로 3D 타원체를 생성하고, 각 슬라이스에 단면 마스크를 저장.

    저장 구조:
      {output_dir}/
        {sid}_z{z_idx:03d}_mask_outer.png    ← 외곽 부종 마스크
        {sid}_z{z_idx:03d}_mask_core.png     ← 내부 괴사핵 마스크
        {sid}_z{z_idx:03d}_mask_combined.png ← Inpainting 입력용 (= outer)
      ellipsoid_meta.json  ← 각 타원체 파라미터 기록 (재현성)
    """
    mask_cfg      = cfg.get("mask_generation", {})
    n_ellipsoids  = mask_cfg.get("num_masks_per_image", 3)   # 볼륨당 타원체 수
    area_min      = mask_cfg.get("min_area_ratio", 0.025)
    area_max      = mask_cfg.get("max_area_ratio", 0.12)
    n_control     = mask_cfg.get("num_control_points", 10)
    irregularity  = mask_cfg.get("irregularity", 0.45)
    alpha         = mask_cfg.get("elastic_alpha", 14.0)
    sigma         = mask_cfg.get("elastic_sigma", 4.0)
    core_min      = mask_cfg.get("core_ratio_min", 0.35)
    core_max      = mask_cfg.get("core_ratio_max", 0.55)
    brain_min     = mask_cfg.get("brain_ratio_min", 0.08)
    brain_max     = mask_cfg.get("brain_ratio_max", 0.65)
    center_min    = mask_cfg.get("center_signal_min", 0.10)
    edge_max      = mask_cfg.get("edge_ratio_max", 0.20)   # [수정] default 0.12→0.20
    overlap_thr   = mask_cfg.get("overlap_threshold", 0.30)   # [v3 추가] 겹침 방지 임계값

    Path(output_dir).mkdir(parents=True, exist_ok=True)
    slice_paths = sorted(Path(slice_dir).glob("*.png"))
    print(f"전체 슬라이스: {len(slice_paths)}장")

    # ── 볼륨별 그룹핑 ──────────────────────────────────
    groups = group_slices_by_volume(slice_paths)
    print(f"볼륨(피험자) 수: {len(groups)}개")

    rng = np.random.default_rng(seed)
    saved        = 0
    skipped_vol  = 0
    meta_records = []

    for sid, z_slices in tqdm(groups.items(), desc="볼륨 처리"):
        n_slices = len(z_slices)
        if n_slices < 3:
            skipped_vol += 1
            continue

        # 대표 슬라이스로 뇌 마스크 추출 (볼륨 중간 슬라이스 사용)
        mid_idx = n_slices // 2
        mid_img = load_slice(str(z_slices[mid_idx][1]))

        if not is_valid_slice(mid_img, brain_min, brain_max, center_min, edge_max):
            skipped_vol += 1
            continue

        brain_mask_rep = extract_brain_mask(mid_img)

        # ── 볼륨당 n_ellipsoids개 타원체 생성 (겹침 방지) ──
        # 대표 슬라이스(mid)에서 outer 마스크를 미리 생성해
        # 이미 배치된 타원체와 30% 이상 겹치면 재샘플링
        placed_ellipsoids  = []   # 확정된 Ellipsoid3D 목록
        placed_mid_masks   = []   # 대표 슬라이스 outer 마스크 (겹침 비교용)
        max_attempts       = n_ellipsoids * 8

        attempts = 0
        while len(placed_ellipsoids) < n_ellipsoids and attempts < max_attempts:
            attempts += 1
            candidate = sample_ellipsoid(
                brain_mask_rep, n_slices,
                (area_min, area_max), rng)

            # 대표 슬라이스에서 후보 마스크 생성
            cand_radii = candidate.slice_radius(int(round(candidate.cz)))
            if cand_radii is None:
                continue
            cand_ry, cand_rx = cand_radii
            cand_contour = generate_spline_contour(
                (candidate.cy, candidate.cx),
                cand_ry, cand_rx, n_control, irregularity, rng)
            cand_mask = contour_to_mask(cand_contour, mid_img.shape)

            # 기존 배치된 마스크와 겹침 확인
            overlap = any(
                masks_overlap(cand_mask, pm, threshold=overlap_thr)
                for pm in placed_mid_masks)
            if overlap:
                continue

            placed_ellipsoids.append(candidate)
            placed_mid_masks.append(cand_mask)

        for e_idx, ellipsoid in enumerate(placed_ellipsoids):

            # 타원체 파라미터 기록 (재현성)
            meta_records.append({
                "sid": sid,
                "ellipsoid_idx": e_idx,
                "cy": float(ellipsoid.cy),
                "cx": float(ellipsoid.cx),
                "cz": float(ellipsoid.cz),
                "ry": float(ellipsoid.ry),
                "rx": float(ellipsoid.rx),
                "rz": float(ellipsoid.rz),
            })

            # ── 각 슬라이스에 단면 마스크 저장 ─────────
            for z_idx, slice_path in z_slices:
                img = load_slice(str(slice_path))

                # 슬라이스 개별 품질 필터
                if not is_valid_slice(img, brain_min, brain_max,
                                      center_min, edge_max):
                    continue

                brain_mask = extract_brain_mask(img)

                result = ellipsoid_slice_to_mask(
                    ellipsoid, z_idx, img.shape, brain_mask,
                    n_control, irregularity, alpha, sigma,
                    (core_min, core_max), rng)

                if result is None:
                    continue   # 이 슬라이스에서 단면 없거나 너무 작음

                prefix = f"{output_dir}/{sid}_z{z_idx:03d}_e{e_idx}"
                Image.fromarray((result["outer"].astype(np.uint8)) * 255).save(
                    f"{prefix}_mask_outer.png")
                Image.fromarray((result["core"].astype(np.uint8)) * 255).save(
                    f"{prefix}_mask_core.png")
                # combined = outer (Inpainting 입력으로 사용)
                Image.fromarray((result["outer"].astype(np.uint8)) * 255).save(
                    f"{prefix}_mask_combined.png")
                saved += 1

    # 타원체 메타데이터 저장
    meta_path = Path(output_dir) / "ellipsoid_meta.json"
    with open(meta_path, "w") as f:
        json.dump(meta_records, f, indent=2)

    print(f"\n완료 — 마스크 저장: {saved}개 | 볼륨 스킵: {skipped_vol}개")
    print(f"타원체 메타: {meta_path}")
    return saved


# ─────────────────────────────────────────────────────────────
# CLI
# ─────────────────────────────────────────────────────────────

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="3D 타원체 기반 종양 마스크 생성 (v3)")
    parser.add_argument("--slice_dir",  required=True,  help="OASIS 슬라이스 디렉토리")
    parser.add_argument("--output_dir", required=True,  help="마스크 저장 디렉토리")
    parser.add_argument("--config",     required=True,  help="config.yaml 경로")
    parser.add_argument("--seed",       type=int, default=42)
    args = parser.parse_args()

    cfg = load_config(args.config)
    n = process_volumes(args.slice_dir, args.output_dir, cfg, args.seed)
    print(f"총 {n}개 마스크 생성 완료 → {args.output_dir}")
