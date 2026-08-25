"""
Phase 2-B (v5): LoRA 종양 패치 생성 + OASIS 블렌딩
─────────────────────────────────────────────────────
학습된 LoRA + MaskEncoder로 마스크 → 종양 패치 생성 후
OASIS 정상 뇌 슬라이스에 Gaussian 블렌딩으로 삽입.

파이프라인:
  1. v3 마스크 로드 → bbox 크롭
  2. LoRA SD + MaskEncoder → 종양 패치 생성 (128×128)
  3. 히스토그램 매칭으로 OASIS 신호 강도 보정
  4. Gaussian 블렌딩으로 OASIS 슬라이스에 삽입
  5. GT 마스크 저장 (v3 마스크 그대로)
"""

import argparse
import json
import re
from pathlib import Path

import cv2
import numpy as np
import torch
import torch.nn as nn
import yaml
from PIL import Image
from scipy.ndimage import gaussian_filter
from skimage.exposure import match_histograms
from tqdm import tqdm

from finetune_lora import LoRALinear, MaskEncoder, inject_lora_to_unet


# ─────────────────────────────────────────────────────────────
# 설정 로드
# ─────────────────────────────────────────────────────────────

def load_config(path: str) -> dict:
    with open(path) as f:
        return yaml.safe_load(f)


# ─────────────────────────────────────────────────────────────
# 모델 로드
# ─────────────────────────────────────────────────────────────

def load_pipeline(model_id: str, lora_dir: str, device: str):
    """SD UNet + LoRA + MaskEncoder 로드."""
    from diffusers import (AutoencoderKL, UNet2DConditionModel,
                           DDIMScheduler)
    from transformers import CLIPTextModel, CLIPTokenizer

    print(f"[gen] 모델 로드: {model_id}")
    kw = {"local_files_only": True} if str(model_id).startswith("/") else {}
    tokenizer    = CLIPTokenizer.from_pretrained(model_id, subfolder="tokenizer", **kw)
    text_encoder = CLIPTextModel.from_pretrained(model_id, subfolder="text_encoder", **kw)
    vae          = AutoencoderKL.from_pretrained(model_id, subfolder="vae", **kw)
    unet         = UNet2DConditionModel.from_pretrained(model_id, subfolder="unet", **kw)
    scheduler    = DDIMScheduler.from_pretrained(model_id, subfolder="scheduler", **kw)

    vae.requires_grad_(False)
    text_encoder.requires_grad_(False)

    # LoRA 주입 + 가중치 로드
    lora_dir  = Path(lora_dir)
    cfg_path  = lora_dir / "config.json"
    with open(cfg_path) as f:
        lora_cfg = json.load(f)

    rank  = lora_cfg.get("rank", 8)
    alpha = lora_cfg.get("alpha", 8.0)
    unet, lora_layers = inject_lora_to_unet(unet, rank=rank, alpha=alpha)

    lora_state = torch.load(lora_dir / "lora_weights.pt",
                            map_location="cpu")
    for i, layer in enumerate(lora_layers):
        layer.lora_A.weight.data = lora_state[f"lora_{i}.A"]
        layer.lora_B.weight.data = lora_state[f"lora_{i}.B"]
    print(f"[gen] LoRA 가중치 로드 완료 (rank={rank})")

    # MaskEncoder 로드
    mask_encoder = MaskEncoder()
    mask_encoder.load_state_dict(
        torch.load(lora_dir / "mask_encoder.pt", map_location="cpu"))
    print("[gen] MaskEncoder 로드 완료")

    vae.to(device); text_encoder.to(device)
    unet.to(device); mask_encoder.to(device)
    unet.eval(); mask_encoder.eval()

    return {
        "vae": vae, "unet": unet, "scheduler": scheduler,
        "text_encoder": text_encoder, "tokenizer": tokenizer,
        "mask_encoder": mask_encoder,
    }


# ─────────────────────────────────────────────────────────────
# 종양 패치 생성
# ─────────────────────────────────────────────────────────────

@torch.no_grad()
def generate_tumor_patch(pipe: dict,
                          mask_crop: np.ndarray,
                          patch_size: int = 128,
                          num_steps: int = 30,
                          device: str = "cuda") -> np.ndarray:
    """
    마스크 크롭 → LoRA SD → 종양 패치 생성.

    mask_crop : [H, W] uint8
    반환      : [H, W] float32 [0,1]
    """
    vae  = pipe["vae"]
    unet = pipe["unet"]
    sched= pipe["scheduler"]
    mask_encoder = pipe["mask_encoder"]

    # 마스크 리사이즈 → tensor
    mask_resized = cv2.resize(mask_crop, (patch_size, patch_size),
                              interpolation=cv2.INTER_NEAREST)
    mask_t = torch.from_numpy(
        (mask_resized > 127).astype(np.float32)
    ).unsqueeze(0).unsqueeze(0).to(device)  # [1, 1, 128, 128]

    # 마스크 latent 인코딩
    mask_latent = mask_encoder(mask_t)  # [1, 4, 16, 16]

    # 랜덤 latent 초기화
    latent_h = patch_size // 8
    latent_w = patch_size // 8
    latents  = torch.randn(1, 4, latent_h, latent_w,
                           device=device, dtype=torch.float32)

    # 빈 텍스트 임베딩
    tokenizer    = pipe["tokenizer"]
    text_encoder = pipe["text_encoder"]
    tokens  = tokenizer([""], padding="max_length",
                        max_length=tokenizer.model_max_length,
                        return_tensors="pt").input_ids.to(device)
    embeds  = text_encoder(tokens)[0]  # [1, 77, 768]

    # DDIM 역방향 샘플링
    sched.set_timesteps(num_steps)
    for t in sched.timesteps:
        conditioned = latents + mask_latent
        noise_pred  = unet(conditioned, t,
                           encoder_hidden_states=embeds).sample
        latents     = sched.step(noise_pred, t, latents).prev_sample

    # latent → 이미지
    latents_scaled = latents / vae.config.scaling_factor
    decoded = vae.decode(latents_scaled).sample  # [1, 3, 128, 128]

    # [-1,1] → [0,1] 그레이스케일
    decoded_np = decoded[0].cpu().float().numpy()  # [3, H, W]
    if decoded_np.ndim == 3:
        img = decoded_np.mean(0)  # 채널 평균 → [H, W]
    else:
        img = decoded_np
    img = (img + 1.0) / 2.0
    img = np.clip(img, 0, 1).astype(np.float32)

    # 원래 마스크 크롭 크기로 리사이즈
    h, w = mask_crop.shape
    if (h, w) != (patch_size, patch_size):
        img = cv2.resize(img, (w, h), interpolation=cv2.INTER_LINEAR)

    return img.astype(np.float32)


# ─────────────────────────────────────────────────────────────
# 블렌딩
# ─────────────────────────────────────────────────────────────

def blend_patch_to_brain(brain: np.ndarray,
                          patch: np.ndarray,
                          target_mask: np.ndarray,
                          blend_sigma: float = 3.0) -> np.ndarray:
    """
    생성된 종양 패치를 OASIS 슬라이스의 마스크 위치에 Gaussian 블렌딩.
    뇌 구조 완전 보존, 마스크 영역만 교체.
    """
    result = brain.copy()
    ys, xs = np.where(target_mask)
    if len(ys) == 0:
        return result

    y1, y2 = ys.min(), ys.max() + 1
    x1, x2 = xs.min(), xs.max() + 1
    th, tw  = y2 - y1, x2 - x1

    # 패치 크기 맞추기
    if patch.shape != (th, tw):
        patch = cv2.resize(patch, (tw, th), interpolation=cv2.INTER_LINEAR)

    # 히스토그램 매칭 (OASIS 신호 강도에 맞게 보정)
    # patch가 3채널(RGB)이면 그레이스케일로 변환
    if patch.ndim == 3:
        patch = patch.mean(axis=2)
    patch = patch.astype(np.float32)
    if patch.max() > 1.0:
        patch = patch / 255.0
    ref_px = brain[brain > 0.05].reshape(-1, 1)
    if len(ref_px) > 10:
        patch = np.clip(
            match_histograms(patch.reshape(-1, 1), ref_px,
                             channel_axis=0).reshape(patch.shape), 0, 1)

    # Gaussian 블렌딩
    mask_crop   = target_mask[y1:y2, x1:x2].astype(np.float32)
    mask_smooth = np.clip(gaussian_filter(mask_crop, sigma=blend_sigma), 0, 1)
    brain_crop  = brain[y1:y2, x1:x2]
    result[y1:y2, x1:x2] = np.clip(
        patch * mask_smooth + brain_crop * (1 - mask_smooth), 0, 1)

    return result


# ─────────────────────────────────────────────────────────────
# 페어 빌드
# ─────────────────────────────────────────────────────────────

def build_pairs(slice_dir: str, mask_dir: str,
                max_images: int = None) -> list:
    slice_dir = Path(slice_dir)
    mask_dir  = Path(mask_dir)

    print("[gen] 마스크 캐싱 중...")
    mask_lookup = {}
    for mp in mask_dir.glob("*_mask_combined.png"):
        m = re.match(r'(.+)_z(\d+)_e(\d+)_mask_combined', mp.stem)
        if m:
            mask_lookup[(m.group(1), int(m.group(2)), int(m.group(3)))] = mp
    print(f"[gen] 마스크 {len(mask_lookup)}개")

    slice_paths = sorted(slice_dir.glob("*.png"))
    pairs = []

    def get_sid_z(fname):
        m = re.match(r'^(.+)_slice(\d+)$', Path(fname).stem)
        return (m.group(1), int(m.group(2))) if m else (Path(fname).stem, 0)

    for sp in slice_paths:
        sid, z_idx = get_sid_z(sp.name)
        for e_idx in range(10):
            if (sid, z_idx, e_idx) not in mask_lookup:
                break
            pairs.append({
                "slice":     sp,
                "mask":      mask_lookup[(sid, z_idx, e_idx)],
                "out_name":  f"{sid}_z{z_idx:03d}_e{e_idx}_synthetic.png",
                "mask_name": f"{sid}_z{z_idx:03d}_e{e_idx}_gt_mask.png",
            })

    pairs.sort(key=lambda x: x["out_name"])
    if max_images:
        pairs = pairs[:max_images]
    print(f"[gen] 페어: {len(pairs)}개")
    return pairs


# ─────────────────────────────────────────────────────────────
# 메인
# ─────────────────────────────────────────────────────────────

def run_generation(args):
    cfg      = load_config(args.config)
    inp_cfg  = cfg.get("inpainting", {})
    max_imgs = args.max_images or inp_cfg.get("num_synthetic_images", 1000)
    steps    = inp_cfg.get("num_inference_steps", 30)
    device   = "cuda" if torch.cuda.is_available() else "cpu"

    Path(args.output_dir).mkdir(parents=True, exist_ok=True)
    gt_dir = Path(args.output_dir) / "gt_masks"
    gt_dir.mkdir(exist_ok=True)

    # 모델 로드
    lora_cfg    = cfg.get("lora_finetuning", {})
    model_id    = lora_cfg.get("model_id", "runwayml/stable-diffusion-v1-5")
    pipe = load_pipeline(model_id, args.lora_dir, device)

    # 페어 빌드
    pairs = build_pairs(args.slice_dir, args.mask_dir, max_imgs)
    existing = set(p.name for p in Path(args.output_dir).glob("*_synthetic.png"))
    pairs = [p for p in pairs if p["out_name"] not in existing]
    print(f"[gen] 신규 생성: {len(pairs)}개 (기존 {len(existing)}개 스킵)")

    saved = errors = 0

    for pair in tqdm(pairs, desc="종양 패치 생성 + 블렌딩"):
        try:
            # 슬라이스 로드
            brain = np.array(Image.open(pair["slice"]).convert("L"),
                             dtype=np.float32) / 255.0

            # 마스크 로드
            mask_arr    = np.array(Image.open(pair["mask"]).convert("L"))
            target_mask = (mask_arr > 127)
            if target_mask.sum() < 50:
                continue

            # 마스크 bbox 크롭
            ys, xs = np.where(target_mask)
            y1, y2 = ys.min(), ys.max() + 1
            x1, x2 = xs.min(), xs.max() + 1
            mask_crop = mask_arr[y1:y2, x1:x2]

            # 종양 패치 생성
            patch = generate_tumor_patch(
                pipe, mask_crop,
                patch_size=128,
                num_steps=steps,
                device=device)

            # OASIS 슬라이스에 블렌딩
            synthetic = blend_patch_to_brain(
                brain, patch, target_mask,
                blend_sigma=args.blend_sigma)

            # 저장
            Image.fromarray(
                (np.clip(synthetic, 0, 1) * 255).astype(np.uint8)
            ).save(str(Path(args.output_dir) / pair["out_name"]))

            Image.fromarray(
                (target_mask.astype(np.uint8)) * 255
            ).save(str(gt_dir / pair["mask_name"]))

            saved += 1

        except Exception as e:
            print(f"\n[gen] 오류 [{pair['slice'].name}]: {e}")
            errors += 1

    total = len(list(Path(args.output_dir).glob("*_synthetic.png")))
    with open(Path(args.output_dir) / "metadata.json", "w") as f:
        json.dump({"total": total, "new": saved, "errors": errors,
                   "method": "lora_patch_blending"}, f, indent=2)

    print(f"\n[gen] 완료 — 새로 생성: {saved}개 | 오류: {errors}개 | 전체: {total}개")


# ─────────────────────────────────────────────────────────────
# CLI
# ─────────────────────────────────────────────────────────────

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="LoRA 패치 생성 + 블렌딩 (v5)")
    parser.add_argument("--slice_dir",   required=True)
    parser.add_argument("--mask_dir",    required=True)
    parser.add_argument("--output_dir",  required=True)
    parser.add_argument("--lora_dir",    required=True,
                        help="LoRA 가중치 디렉토리 (lora_final/)")
    parser.add_argument("--config",      required=True)
    parser.add_argument("--max_images",  type=int, default=None)
    parser.add_argument("--blend_sigma", type=float, default=3.0)
    args = parser.parse_args()

    run_generation(args)
