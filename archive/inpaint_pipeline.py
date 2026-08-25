"""
Phase 2-B: SD Inpainting 합성 뇌종양 MRI 생성 (v3)
────────────────────────────────────────────────────
v3 변경사항:
  - 마스크 매칭 로직: build_pairs_v3() 적용
    파일명 규칙: {sid}_z{z_idx:03d}_e{e_idx}_mask_combined.png
  - 출력 파일명:  {sid}_z{z_idx:03d}_e{e_idx}_synthetic.png
  - 볼륨 → z → e 순 정렬로 처리 (같은 볼륨이 연속 처리됨)

기존 유지사항:
  - LoRA 가중치 로드 (lora_scale 적용)
  - config에서 inference_steps / guidance_scale / num_synthetic_images 읽기
  - metadata.json 저장
"""

import argparse
import json
import re
from pathlib import Path
from collections import defaultdict

import numpy as np
import torch
import yaml
from PIL import Image
from diffusers import StableDiffusionInpaintPipeline
from tqdm import tqdm


# ─────────────────────────────────────────────────────────────
# 설정 로드
# ─────────────────────────────────────────────────────────────

def load_config(config_path: str) -> dict:
    with open(config_path) as f:
        return yaml.safe_load(f)


# ─────────────────────────────────────────────────────────────
# v3 파일명 파싱 및 페어 빌드
# ─────────────────────────────────────────────────────────────

def extract_sid_and_zidx(filename: str):
    """
    슬라이스 파일명에서 sid, z_idx 추출.
    예: OAS1_0001_MR1_mpr-1_slice0040.png → ('OAS1_0001_MR1_mpr-1', 40)
    """
    stem = Path(filename).stem
    m = re.match(r'^(.+)_slice(\d+)$', stem)
    if m:
        return m.group(1), int(m.group(2))
    return stem, 0


def find_masks_for_slice(slice_path: Path, mask_lookup: dict) -> list:
    """
    캐싱된 mask_lookup dict에서 슬라이스에 대응하는 마스크 경로 목록 반환.
    Drive glob 호출 없이 메모리에서 탐색.
    """
    sid, z_idx = extract_sid_and_zidx(slice_path.name)
    results = []
    e_idx = 0
    while True:
        key = (sid, z_idx, e_idx)
        if key not in mask_lookup:
            break
        results.append(mask_lookup[key])
        e_idx += 1
    return results


def build_pairs_v3(slice_dir: str, mask_dir: str,
                   max_images: int = None) -> list:
    """
    v3 파일명 구조에 맞게 (슬라이스, 마스크) 페어 목록 생성.
    Drive FUSE 속도 문제 해결을 위해 마스크 목록을 한 번에 로드 후 dict로 캐싱.
    """
    slice_dir = Path(slice_dir)
    mask_dir  = Path(mask_dir)

    # ── 마스크 파일 목록 한 번에 캐싱 (슬라이스마다 glob 호출 방지) ──
    print("[inpaint] 마스크 목록 캐싱 중...")
    mask_lookup = {}   # {(sid, z_idx, e_idx): Path}
    for mp in mask_dir.glob("*_mask_combined.png"):
        m = re.match(r'(.+)_z(\d+)_e(\d+)_mask_combined', mp.stem)
        if m:
            sid_m  = m.group(1)
            z_m    = int(m.group(2))
            e_m    = int(m.group(3))
            mask_lookup[(sid_m, z_m, e_m)] = mp
    print(f"[inpaint] 마스크 캐싱 완료: {len(mask_lookup)}개")

    print("[inpaint] 슬라이스 목록 로드 중...")
    slice_paths = sorted(slice_dir.glob("*.png"))
    print(f"[inpaint] 슬라이스 {len(slice_paths)}장 로드 완료")
    pairs = []

    for sp in slice_paths:
        sid, z_idx = extract_sid_and_zidx(sp.name)
        # 해당 슬라이스의 모든 타원체(e_idx) 마스크 탐색
        e_idx = 0
        while True:
            key = (sid, z_idx, e_idx)
            if key not in mask_lookup:
                break
            pairs.append({
                "slice": sp,
                "mask":  mask_lookup[key],
                "sid":   sid,
                "z_idx": z_idx,
                "e_idx": e_idx,
            })
            e_idx += 1

    # 볼륨 → z → e 순 정렬
    pairs.sort(key=lambda x: (x["sid"], x["z_idx"], x["e_idx"]))

    if max_images is not None:
        pairs = pairs[:max_images]

    print(f"[inpaint] 전체 슬라이스: {len(slice_paths)}장 | "
          f"페어: {len(pairs)}개 "
          f"(평균 {len(pairs)/max(len(slice_paths),1):.1f} 마스크/슬라이스)")
    return pairs


def make_output_name(pair: dict) -> str:
    """합성 이미지 저장 파일명."""
    return f"{pair['sid']}_z{pair['z_idx']:03d}_e{pair['e_idx']}_synthetic.png"


# ─────────────────────────────────────────────────────────────
# 이미지 로드 유틸
# ─────────────────────────────────────────────────────────────

def load_image_rgb(path: str, size: int) -> Image.Image:
    """슬라이스(그레이스케일) → RGB PIL Image, size×size 리사이즈"""
    img = Image.open(path).convert("L").resize((size, size), Image.LANCZOS)
    return img.convert("RGB")


def load_mask(path: str, size: int) -> Image.Image:
    """마스크 PNG → L PIL Image, size×size 리사이즈"""
    return Image.open(path).convert("L").resize((size, size), Image.NEAREST)


# ─────────────────────────────────────────────────────────────
# 파이프라인 초기화
# ─────────────────────────────────────────────────────────────

def load_pipeline(model_id: str, lora_weights: str,
                  lora_scale: float, device: str,
                  model_cache: str = None) -> StableDiffusionInpaintPipeline:
    """SD Inpainting 파이프라인 로드 + LoRA 가중치 적용
    model_cache: 로컬 캐시 경로 (있으면 온라인 다운로드 없이 로컬에서 로드)
    """
    use_local = model_cache and Path(model_cache).exists()
    load_path = model_cache if use_local else model_id
    print(f"[inpaint] 모델 로드: {load_path} (로컬 캐시)" if use_local else f"[inpaint] 모델 로드: {load_path} (온라인)")
    pipe = StableDiffusionInpaintPipeline.from_pretrained(
        load_path,
        torch_dtype=torch.float16 if device == "cuda" else torch.float32,
        safety_checker=None,
        local_files_only=bool(use_local),
    ).to(device)

    lora_path = Path(lora_weights)
    if lora_path.exists():
        print(f"[inpaint] LoRA 가중치 로드: {lora_weights} (scale={lora_scale})")
        pipe.load_lora_weights(str(lora_path))
        pipe.fuse_lora(lora_scale=lora_scale)
    else:
        print(f"[inpaint] ⚠️  LoRA 경로 없음, 기본 모델로 진행: {lora_weights}")

    pipe.enable_attention_slicing()
    return pipe


# ─────────────────────────────────────────────────────────────
# 메인 파이프라인
# ─────────────────────────────────────────────────────────────

def run_inpainting(args):
    cfg         = load_config(args.config)
    inp_cfg     = cfg.get("inpainting", {})

    # config 값 (CLI 인수가 우선)
    model_id    = inp_cfg.get("model_id", "runwayml/stable-diffusion-inpainting")
    device      = inp_cfg.get("device", "cuda" if torch.cuda.is_available() else "cpu")
    prompt      = inp_cfg.get("prompt", "MRI of brain tumor glioblastoma, T1 contrast enhanced, "
                              "necrotic core, enhancing rim, peritumoral edema, realistic")
    neg_prompt  = inp_cfg.get("negative_prompt",
                              "blurry, artifact, low quality, unrealistic, cartoon, geometric, symmetric")
    steps       = inp_cfg.get("num_inference_steps", 30)
    guidance    = inp_cfg.get("guidance_scale", 7.5)
    img_size    = inp_cfg.get("image_size", 512)
    max_images  = args.max_images if args.max_images else inp_cfg.get("num_synthetic_images", 1000)

    Path(args.output_dir).mkdir(parents=True, exist_ok=True)

    # ── 페어 빌드 (v3) ────────────────────────────────────
    pairs = build_pairs_v3(args.slice_dir, args.mask_dir, max_images)
    if not pairs:
        print("[inpaint] ❌ 유효한 슬라이스-마스크 페어 없음. 경로와 파일명을 확인하세요.")
        return

    # 이미 생성된 파일 스킵 (중단 후 재시작 대비)
    existing = set(p.name for p in Path(args.output_dir).glob("*_synthetic.png"))
    pairs = [p for p in pairs if make_output_name(p) not in existing]
    print(f"[inpaint] 신규 생성 대상: {len(pairs)}개 (기존 {len(existing)}개 스킵)")

    if not pairs:
        print("[inpaint] 모두 완료되어 있음.")
        return

    # ── 모델 로드 ─────────────────────────────────────────
    pipe = load_pipeline(model_id, args.lora_weights, args.lora_scale, device,
                         model_cache=getattr(args, "model_cache", None))

    # ── 생성 루프 ─────────────────────────────────────────
    saved = 0
    errors = 0

    for pair in tqdm(pairs, desc="SD Inpainting"):
        try:
            image  = load_image_rgb(str(pair["slice"]), img_size)
            mask   = load_mask(str(pair["mask"]), img_size)

            result = pipe(
                prompt=prompt,
                negative_prompt=neg_prompt,
                image=image,
                mask_image=mask,
                num_inference_steps=steps,
                guidance_scale=guidance,
            ).images[0]

            out_name = make_output_name(pair)
            result.save(str(Path(args.output_dir) / out_name))
            saved += 1

        except Exception as e:
            print(f"\n[inpaint] 오류 [{pair['slice'].name}]: {e}")
            errors += 1
            continue

    # ── metadata.json 저장 ────────────────────────────────
    total_in_dir = len(list(Path(args.output_dir).glob("*_synthetic.png")))
    meta = {
        "total":    total_in_dir,
        "new":      saved,
        "errors":   errors,
        "prompt":   prompt,
        "steps":    steps,
        "guidance": guidance,
        "lora":     args.lora_weights,
        "lora_scale": args.lora_scale,
    }
    with open(Path(args.output_dir) / "metadata.json", "w") as f:
        json.dump(meta, f, indent=2)

    print(f"\n[inpaint] 완료 — 새로 생성: {saved}개 | 오류: {errors}개 | "
          f"전체 누적: {total_in_dir}개")
    print(f"[inpaint] 저장 경로: {args.output_dir}")


# ─────────────────────────────────────────────────────────────
# CLI
# ─────────────────────────────────────────────────────────────

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="SD Inpainting 합성 데이터 생성 (v3)")
    parser.add_argument("--slice_dir",    required=True,  help="OASIS 슬라이스 디렉토리")
    parser.add_argument("--mask_dir",     required=True,  help="종양 마스크 디렉토리 (v3)")
    parser.add_argument("--output_dir",   required=True,  help="합성 이미지 저장 디렉토리")
    parser.add_argument("--lora_weights", required=True,  help="LoRA 가중치 경로")
    parser.add_argument("--lora_scale",   type=float, default=0.9)
    parser.add_argument("--config",       required=True,  help="config.yaml 경로")
    parser.add_argument("--max_images",   type=int, default=None, help="생성할 최대 이미지 수")
    parser.add_argument("--model_cache",  type=str, default=None, help="로컬 모델 캐시 경로")
    args = parser.parse_args()

    run_inpainting(args)
