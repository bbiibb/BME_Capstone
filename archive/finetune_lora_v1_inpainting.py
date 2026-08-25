"""
Phase 2 - LoRA Fine-tuning: BraTS 종양 질감을 SD Inpainting에 학습

── 핵심 아이디어 ────────────────────────────────────────────────────────────────
SD Inpainting은 자연 이미지로 사전학습되어 있어, MRI 특유의 GBM 질감
(T1ce 강화 림, 괴사핵 저신호, 주변 부종)을 정확히 재현하지 못한다.

해결책: BraTS T1ce 슬라이스에서 실제 종양 패치를 추출하여 LoRA로 fine-tuning.
  - 학습 입력: 종양이 제거된 뇌 슬라이스 + 마스크  (SD가 "채워야 할 상황")
  - 학습 목표: 원본 T1ce 슬라이스 (종양이 있는 실제 모습)
  → SD가 "이 마스크 안에는 GBM이 들어가야 한다"는 것을 MRI 도메인으로 학습

── LoRA 설정 ────────────────────────────────────────────────────────────────────
rank=4, target: UNet의 cross-attention Q/K/V/out projection
학습 가능 파라미터: 전체의 약 0.1% → GPU 메모리 최소화, 빠른 수렴

── 실행 방법 ────────────────────────────────────────────────────────────────────
python phase2_inpainting/finetune_lora.py \
    --brats_dir  data/processed/brats_slices \
    --output_dir checkpoints/lora_gbm \
    --config     configs/config.yaml
"""

import os
import sys
import logging
import argparse
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from PIL import Image
from tqdm import tqdm
import yaml

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from utils.io_utils import load_png_as_float

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger(__name__)


# ──────────────────────────────────────────────────────────────────────────────
# 데이터셋
# ──────────────────────────────────────────────────────────────────────────────

class BraTSInpaintingDataset(Dataset):
    """
    LoRA 학습용 데이터셋.

    SD Inpainting의 학습 구조:
      - image      : 원본 T1ce 슬라이스 (종양 포함, 학습 목표)
      - masked_image: 종양 영역을 지운 슬라이스 (SD 입력 — "이 자리를 채워라")
      - mask       : 종양 이진 마스크 (1=인페인팅 영역)
      - prompt     : GBM 설명 텍스트

    BraTS 라벨: 1=괴사핵, 2=부종, 4=강화종양 → 전체 종양(>0) 마스크 사용
    """

    PROMPT = (
        "axial T1 contrast-enhanced MRI, glioblastoma multiforme, "
        "heterogeneous enhancing rim, central necrotic core, "
        "peritumoral white matter edema, medical imaging, grayscale"
    )

    def __init__(self,
                 brats_dir: str,
                 image_size: int = 256,
                 min_tumor_ratio: float = 0.01,
                 mask_dilation_px: int = 8,
                 **kwargs):
        import cv2
        from scipy.ndimage import binary_dilation

        self.image_size = image_size
        self.min_tumor_ratio = min_tumor_ratio
        self.mask_dilation_px = mask_dilation_px

        img_dir  = Path(brats_dir) / "images"
        mask_dir = Path(brats_dir) / "masks"

        if not img_dir.exists():
            raise FileNotFoundError(
                f"BraTS 이미지 폴더 없음: {img_dir}\n"
                "먼저 phase4_evaluation/prepare_brats_slices.py 를 실행하세요."
            )

        # 경로 수집
        mask_stems = {p.stem.replace("_seg", "") for p in mask_dir.glob("*_seg.png")}
        paths = []
        for img_path in sorted(img_dir.glob("*_t1ce.png")):
            stem = img_path.stem.replace("_t1ce", "")
            if stem in mask_stems:
                paths.append((str(img_path), str(mask_dir / f"{stem}_seg.png")))

        if not paths:
            raise FileNotFoundError(
                f"BraTS 슬라이스 없음: {brats_dir}\n"
                "먼저 phase4_evaluation/prepare_brats_slices.py 를 실행하세요."
            )

        # ── numpy 캐시 파일로 빠른 로딩 ───────────────────────────────────────
        # 첫 실행: Drive PNG → numpy 배열 변환 후 .npz 저장
        # 이후 실행: .npz 로드만 (수 초)
        max_samples = kwargs.get("max_samples", None)
        cache_path  = str(Path(brats_dir) / f"_lora_cache_{image_size}_n{max_samples or 'all'}.npz")

        if os.path.exists(cache_path):
            logger.info(f"numpy 캐시 로드: {cache_path}")
            data = np.load(cache_path)
            self.images = list(data["images"])
            self.masks  = list(data["masks"])
            logger.info(f"캐시 로드 완료: {len(self.images)}개 샘플")
        else:
            target_paths = paths[:max_samples] if max_samples else paths
            logger.info(f"RAM 캐싱 시작: {len(target_paths)}개 파일 로드 중...")
            imgs_list  = []
            masks_list = []
            for img_p, mask_p in tqdm(target_paths, desc="RAM 캐시 로드", ncols=80):
                img      = load_png_as_float(img_p)
                # BraTS 마스크: 라벨값 0,1,2,4 → > 0 이진화 (> 127 하면 전부 0됨)
                mask_raw = np.array(Image.open(mask_p).convert("L"))
                mask     = (mask_raw > 0).astype(np.float32)

                if mask.mean() < min_tumor_ratio:
                    continue

                if mask_dilation_px > 0:
                    struct = np.ones((mask_dilation_px, mask_dilation_px))
                    mask = binary_dilation(mask, structure=struct).astype(np.float32)

                img_r  = cv2.resize(img,  (image_size, image_size), interpolation=cv2.INTER_LINEAR)
                mask_r = cv2.resize(mask, (image_size, image_size), interpolation=cv2.INTER_NEAREST)

                imgs_list.append(img_r.astype(np.float32))
                masks_list.append(mask_r.astype(np.float32))

            self.images = imgs_list
            self.masks  = masks_list
            logger.info(f"RAM 캐싱 완료: {len(self.images)}개 샘플 (종양 비율 ≥ {min_tumor_ratio:.1%})")

            # .npz 저장 — 다음 실행부터 수 초 로드
            logger.info(f"numpy 캐시 저장 중: {cache_path}")
            np.savez_compressed(cache_path,
                                images=np.array(self.images),
                                masks=np.array(self.masks))
            logger.info("캐시 저장 완료")

    def __len__(self) -> int:
        return len(self.images)

    def __getitem__(self, idx: int) -> dict:
        img  = self.images[idx]   # (H,W) float32 [0,1]
        mask = self.masks[idx]    # (H,W) float32 [0,1]

        masked_img = img * (1.0 - mask)

        # RGB tensor [-1, 1]
        def to_rgb(a):
            rgb = np.stack([a] * 3, axis=0)
            return torch.from_numpy(rgb * 2.0 - 1.0)

        return {
            "image":        to_rgb(img),
            "masked_image": to_rgb(masked_img),
            "mask":         torch.from_numpy(mask).unsqueeze(0),
            "prompt":       self.PROMPT,
        }


# ──────────────────────────────────────────────────────────────────────────────
# LoRA 주입 유틸
# ──────────────────────────────────────────────────────────────────────────────

def inject_lora(unet, rank: int = 4, alpha: float = 4.0):
    """UNet cross-attention Q/K/V/out 레이어에 LoRA 어댑터 주입"""
    from peft import LoraConfig, get_peft_model

    target_modules = []
    for name, _ in unet.named_modules():
        if any(k in name for k in ["to_q", "to_k", "to_v", "to_out.0"]):
            target_modules.append(name)

    lora_cfg = LoraConfig(
        r=rank,
        lora_alpha=alpha,
        target_modules=list(set(target_modules)),
        lora_dropout=0.05,
        bias="none",
    )
    unet = get_peft_model(unet, lora_cfg)
    trainable = sum(p.numel() for p in unet.parameters() if p.requires_grad)
    total     = sum(p.numel() for p in unet.parameters())
    logger.info(f"LoRA 주입 (rank={rank}, alpha={alpha})")
    logger.info(f"학습 파라미터: {trainable:,} / {total:,} ({trainable/total*100:.4f}%)")
    return unet


def encode_images(vae, images):
    """이미지 → VAE latent"""
    latents = vae.encode(images).latent_dist.sample()
    return latents * vae.config.scaling_factor


def encode_prompt(text_encoder, tokenizer, prompt: str, device):
    """텍스트 → CLIP 임베딩"""
    tokens = tokenizer(
        prompt,
        padding="max_length",
        max_length=tokenizer.model_max_length,
        truncation=True,
        return_tensors="pt",
    )
    with torch.no_grad():
        emb = text_encoder(tokens.input_ids.to(device))[0]
    return emb


# ──────────────────────────────────────────────────────────────────────────────
# 학습 메인
# ──────────────────────────────────────────────────────────────────────────────

def train_lora(cfg_lora: dict, brats_dir: str, output_dir: str, device: torch.device):
    from diffusers import (
        StableDiffusionInpaintPipeline,
        DDPMScheduler,
    )
    from transformers import CLIPTextModel, CLIPTokenizer

    model_id   = cfg_lora["model_id"]
    rank       = cfg_lora["lora_rank"]
    alpha      = cfg_lora["lora_alpha"]
    lr         = cfg_lora["learning_rate"]
    num_epochs = cfg_lora["num_epochs"]
    batch_size = cfg_lora["batch_size"]
    image_size = cfg_lora["image_size"]
    save_every = cfg_lora.get("save_every_epochs", 1)
    min_tumor  = cfg_lora.get("min_tumor_ratio", 0.01)
    dilation   = cfg_lora.get("mask_dilation_px", 8)

    dtype = torch.float16 if device.type == "cuda" else torch.float32

    logger.info(f"모델 로드: {model_id}")
    pipe = StableDiffusionInpaintPipeline.from_pretrained(
        model_id, torch_dtype=dtype,
    )

    vae          = pipe.vae.to(device)
    text_encoder = pipe.text_encoder.to(device)
    tokenizer    = pipe.tokenizer
    unet         = pipe.unet.to(device)

    vae.requires_grad_(False)
    text_encoder.requires_grad_(False)

    noise_sched = DDPMScheduler.from_pretrained(model_id, subfolder="scheduler")

    unet = inject_lora(unet, rank=rank, alpha=alpha)
    unet.train()

    # 데이터셋 — num_workers=0: Drive FUSE와 멀티프로세싱 충돌 방지
    max_samples = cfg_lora.get("max_samples", None)
    dataset = BraTSInpaintingDataset(
        brats_dir, image_size=image_size,
        min_tumor_ratio=min_tumor, mask_dilation_px=dilation,
        max_samples=max_samples,
    )
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=True,
                        num_workers=0, pin_memory=False)

    optimizer = torch.optim.AdamW(
        filter(lambda p: p.requires_grad, unet.parameters()),
        lr=lr, weight_decay=1e-4,
    )
    total_steps = num_epochs * len(loader)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=total_steps, eta_min=lr * 0.1
    )

    os.makedirs(output_dir, exist_ok=True)
    scaler = torch.amp.GradScaler("cuda") if device.type == "cuda" else None

    logger.info(f"LoRA 학습 시작: {num_epochs} epochs × {len(loader)} steps/epoch")

    for epoch in range(num_epochs):
        epoch_loss = 0.0
        pbar = tqdm(loader, desc=f"Epoch [{epoch+1}/{num_epochs}]")

        for batch in pbar:
            images        = batch["image"].to(device, dtype=dtype)
            masked_images = batch["masked_image"].to(device, dtype=dtype)
            masks         = batch["mask"].to(device, dtype=dtype)
            prompts       = batch["prompt"]

            with torch.amp.autocast("cuda", enabled=(scaler is not None)):
                latents = encode_images(vae, images)

                t = torch.randint(
                    0, noise_sched.config.num_train_timesteps,
                    (latents.shape[0],), device=device
                ).long()

                noise = torch.randn_like(latents)
                noisy_latents = noise_sched.add_noise(latents, noise, t)

                masked_latents = encode_images(vae, masked_images)
                mask_resized = F.interpolate(
                    masks, size=latents.shape[-2:], mode="nearest"
                )
                unet_input = torch.cat([noisy_latents, mask_resized, masked_latents], dim=1)

                text_emb = encode_prompt(text_encoder, tokenizer, prompts[0], device)
                text_emb = text_emb.repeat(latents.shape[0], 1, 1)

                noise_pred = unet(unet_input, t, encoder_hidden_states=text_emb).sample

                loss = F.mse_loss(noise_pred, noise, reduction="none")
                weight = 1.0 + mask_resized.squeeze(1).unsqueeze(1)
                loss = (loss * weight).mean()

            optimizer.zero_grad()
            if scaler:
                scaler.scale(loss).backward()
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(unet.parameters(), 1.0)
                scaler.step(optimizer)
                scaler.update()
            else:
                loss.backward()
                torch.nn.utils.clip_grad_norm_(unet.parameters(), 1.0)
                optimizer.step()

            scheduler.step()
            epoch_loss += loss.item()
            pbar.set_postfix(loss=f"{loss.item():.4f}",
                             lr=f"{scheduler.get_last_lr()[0]:.2e}")

        avg_loss = epoch_loss / len(loader)
        logger.info(f"Epoch {epoch+1} 완료 | avg_loss={avg_loss:.4f}")

        # 매 에포크 Drive에 저장 (런타임 끊김 대비)
        if (epoch + 1) % save_every == 0 or (epoch + 1) == num_epochs:
            ckpt_path = os.path.join(output_dir, f"lora_epoch{epoch+1:04d}")
            os.makedirs(ckpt_path, exist_ok=True)
            unet.save_pretrained(ckpt_path)
            logger.info(f"LoRA 체크포인트 저장: {ckpt_path}")

    # 최종 저장
    final_path = os.path.join(output_dir, "lora_final")
    os.makedirs(final_path, exist_ok=True)
    unet.save_pretrained(final_path)
    logger.info(f"최종 LoRA 저장: {final_path}")
    return final_path


# ──────────────────────────────────────────────────────────────────────────────
# 메인
# ──────────────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="SD Inpainting LoRA Fine-tuning")
    parser.add_argument("--brats_dir",   default="data/processed/brats_slices")
    parser.add_argument("--output_dir",  default="checkpoints/lora_gbm")
    parser.add_argument("--config",      default="configs/config.yaml")
    args = parser.parse_args()

    with open(args.config) as f:
        cfg = yaml.safe_load(f)
    cfg_lora = cfg["lora_finetuning"]

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info(f"디바이스: {device}")

    try:
        import peft  # noqa
    except ImportError:
        logger.error("peft 미설치: pip install 'peft>=0.17.0'")
        return

    train_lora(cfg_lora, args.brats_dir, args.output_dir, device)
    logger.info("LoRA fine-tuning 완료!")


if __name__ == "__main__":
    main()
