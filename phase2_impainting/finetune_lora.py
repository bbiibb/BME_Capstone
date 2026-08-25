"""
Phase 2-A (v2): LoRA Fine-tuning — 마스크 → 종양 패치 생성 학습
─────────────────────────────────────────────────────────────────
변경사항 (기존 Inpainting 방식 → 패치 생성 방식):
  - 기존: SD Inpainting 모델에 전체 슬라이스 + 마스크 → 마스크 영역 채우기
  - 신규: SD v1.5 기반, BraTS 종양 크롭 패치를 마스크로 conditioning
          마스크 입력 → 종양 텍스처 이미지 생성 학습 (ControlNet 방식)

학습 데이터:
  - BraTS T1ce 슬라이스에서 종양 bbox 크롭 (128×128)
  - 대응 마스크 크롭 → conditioning 입력
  - 종양 크롭 이미지 → 생성 타겟

추론 시:
  - v3 마스크 → LoRA SD → 종양 패치 생성
  - 생성된 패치를 OASIS 슬라이스에 블렌딩
"""

import argparse
import json
import logging
import os
from pathlib import Path

import cv2
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import yaml
from PIL import Image
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from tqdm import tqdm

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger(__name__)


# ─────────────────────────────────────────────────────────────
# 설정
# ─────────────────────────────────────────────────────────────

def load_config(path: str) -> dict:
    with open(path) as f:
        return yaml.safe_load(f)


# ─────────────────────────────────────────────────────────────
# 데이터셋: BraTS 종양 크롭 패치
# ─────────────────────────────────────────────────────────────

class TumorPatchDataset(Dataset):
    """
    BraTS T1ce 슬라이스에서 종양 bbox를 크롭한 패치 데이터셋.

    반환:
      image : [1, H, W] float32  ← 종양 패치 (생성 타겟)
      mask  : [1, H, W] float32  ← 마스크 크롭 (conditioning)
    """

    def __init__(self,
                 brats_slices_dir: str,
                 patch_size: int = 128,
                 min_tumor_ratio: float = 0.05,
                 max_samples: int = 3000,
                 augment: bool = True,
                 seed: int = 42):

        self.patch_size = patch_size
        self.augment    = augment
        brats_dir  = Path(brats_slices_dir)
        image_dir  = brats_dir / "images"
        mask_dir   = brats_dir / "masks"

        # 평탄 구조 fallback
        if not image_dir.exists():
            img_paths  = sorted(brats_dir.glob("*.png"))
            mask_paths = {p.stem: p for p in brats_dir.glob("*_mask.png")}
            img_paths  = [p for p in img_paths if "_mask" not in p.stem]
        else:
            img_paths  = sorted(image_dir.glob("*.png"))
            mask_paths = {p.stem: p for p in mask_dir.glob("*.png")}

        # 종양 비율 필터링
        self.pairs = []
        rng = np.random.default_rng(seed)
        for ip in img_paths:
            mp = mask_paths.get(ip.stem)
            if mp is None:
                # 파일명 규칙 변환 시도
                # BraTS20_Training_001_slice0034_t1ce → _seg
                for old_sfx, new_sfx in [("_t1ce", "_seg"), ("_t1ce", "_mask"),
                                          ("_image", "_mask"), ("_image", "_seg")]:
                    alt = ip.stem.replace(old_sfx, new_sfx)
                    mp  = mask_paths.get(alt)
                    if mp is not None:
                        break
            if mp is None:
                continue
            mask_arr = np.array(Image.open(mp).convert("L"))
            # BraTS seg 마스크: 픽셀값 0/1/2/4 (raw) 또는 0/255
            max_val = int(mask_arr.max())
            if max_val <= 4:
                tumor_ratio = (mask_arr > 0).mean()
            else:
                tumor_ratio = (mask_arr > 127).mean()
            if tumor_ratio >= min_tumor_ratio:
                self.pairs.append((ip, mp))

        # 최대 샘플 수 제한 (셔플 후 선택)
        idx = rng.permutation(len(self.pairs))[:max_samples]
        self.pairs = [self.pairs[i] for i in idx]
        logger.info(f"TumorPatchDataset: {len(self.pairs)}개 패치")

    def _crop_patch(self, img_arr: np.ndarray,
                    mask_arr: np.ndarray) -> tuple:
        """종양 bbox 기준으로 패치 크롭 후 patch_size로 리사이즈."""
        h, w   = img_arr.shape
        # BraTS raw 마스크(0/1/2/4) 또는 0/255 모두 처리
        max_val = int(mask_arr.max())
        binary = (mask_arr > 0) if max_val <= 4 else (mask_arr > 127)
        ys, xs = np.where(binary)

        if len(ys) == 0:
            return None, None

        y1, y2 = ys.min(), ys.max()
        x1, x2 = xs.min(), xs.max()

        # 여백 20%
        pad_y = max(4, int((y2 - y1) * 0.2))
        pad_x = max(4, int((x2 - x1) * 0.2))
        y1 = max(0, y1 - pad_y)
        y2 = min(h, y2 + pad_y)
        x1 = max(0, x1 - pad_x)
        x2 = min(w, x2 + pad_x)

        img_crop  = img_arr[y1:y2, x1:x2]
        mask_crop = mask_arr[y1:y2, x1:x2]

        ps = self.patch_size
        img_resized  = cv2.resize(img_crop,  (ps, ps),
                                  interpolation=cv2.INTER_LINEAR)
        mask_resized = cv2.resize(mask_crop, (ps, ps),
                                  interpolation=cv2.INTER_NEAREST)

        return img_resized, mask_resized

    def __len__(self):
        return len(self.pairs)

    def __getitem__(self, idx):
        img_path, mask_path = self.pairs[idx]

        img_arr  = np.array(Image.open(img_path).convert("L"),
                            dtype=np.float32) / 255.0
        mask_arr = np.array(Image.open(mask_path).convert("L"))

        img_crop, mask_crop = self._crop_patch(img_arr, mask_arr)
        if img_crop is None:
            # fallback: 랜덤 다른 샘플
            return self.__getitem__((idx + 1) % len(self))

        # 증강 (수평 플립)
        if self.augment and np.random.rand() > 0.5:
            img_crop  = np.fliplr(img_crop).copy()
            mask_crop = np.fliplr(mask_crop).copy()

        # [0,1] → [-1,1] (SD 입력 범위)
        img_tensor  = torch.from_numpy(img_crop).unsqueeze(0).float()
        img_tensor  = img_tensor * 2.0 - 1.0

        # 마스크: [0,1] float
        mask_tensor = torch.from_numpy(
            (mask_crop > 127).astype(np.float32)
        ).unsqueeze(0)

        return {"image": img_tensor, "mask": mask_tensor}


# ─────────────────────────────────────────────────────────────
# LoRA 레이어
# ─────────────────────────────────────────────────────────────

class LoRALinear(nn.Module):
    """기존 Linear 레이어에 LoRA A·B 어댑터 주입."""

    def __init__(self, linear: nn.Linear, rank: int = 8, alpha: float = 8.0):
        super().__init__()
        self.linear = linear
        self.rank   = rank
        self.scale  = alpha / rank

        in_f  = linear.in_features
        out_f = linear.out_features

        self.lora_A = nn.Linear(in_f,  rank,  bias=False)
        self.lora_B = nn.Linear(rank,  out_f, bias=False)

        nn.init.kaiming_uniform_(self.lora_A.weight, a=np.sqrt(5))
        nn.init.zeros_(self.lora_B.weight)

        # 기존 가중치 동결
        for p in self.linear.parameters():
            p.requires_grad = False

    def forward(self, x):
        return self.linear(x) + self.lora_B(self.lora_A(x)) * self.scale


def inject_lora_to_unet(unet, rank: int = 8, alpha: float = 8.0):
    """UNet Cross-Attention 레이어에 LoRA 주입."""
    lora_layers = []
    for name, module in unet.named_modules():
        if isinstance(module, nn.Linear) and any(
            k in name for k in ["to_q", "to_k", "to_v", "to_out"]
        ):
            parent_name = ".".join(name.split(".")[:-1])
            attr_name   = name.split(".")[-1]
            parent      = unet
            for part in parent_name.split("."):
                parent = getattr(parent, part)
            lora_layer = LoRALinear(module, rank=rank, alpha=alpha)
            setattr(parent, attr_name, lora_layer)
            lora_layers.append(lora_layer)

    logger.info(f"LoRA 주입: {len(lora_layers)}개 레이어 (rank={rank})")
    return unet, lora_layers


# ─────────────────────────────────────────────────────────────
# 마스크 Encoder (ControlNet 방식)
# ─────────────────────────────────────────────────────────────

class MaskEncoder(nn.Module):
    """
    마스크 이미지를 SD UNet 입력과 같은 채널/해상도로 인코딩.
    [1, H, W] → [4, H/8, W/8] (latent 공간)
    """

    def __init__(self):
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Conv2d(1, 16, 3, padding=1),  nn.SiLU(),
            nn.Conv2d(16, 32, 3, stride=2, padding=1), nn.SiLU(),  # /2
            nn.Conv2d(32, 64, 3, stride=2, padding=1), nn.SiLU(),  # /4
            nn.Conv2d(64, 4,  3, stride=2, padding=1),             # /8
        )

    def forward(self, mask):
        return self.encoder(mask)


# ─────────────────────────────────────────────────────────────
# 학습
# ─────────────────────────────────────────────────────────────

def train_lora(cfg: dict, brats_dir: str, output_dir: str, device: str):
    """메인 학습 함수."""
    lora_cfg    = cfg.get("lora_finetuning", {})
    model_id    = lora_cfg.get("model_id", "runwayml/stable-diffusion-v1-5")
    rank        = lora_cfg.get("lora_rank", 8)
    alpha       = float(lora_cfg.get("lora_alpha", 8.0))
    lr          = float(lora_cfg.get("learning_rate", 5e-5))
    num_epochs  = lora_cfg.get("num_epochs", 10)
    batch_size  = lora_cfg.get("batch_size", 4)
    patch_size  = lora_cfg.get("image_size", 128)
    max_samples = lora_cfg.get("max_samples", 3000)
    save_every  = lora_cfg.get("save_every_epochs", 1)

    Path(output_dir).mkdir(parents=True, exist_ok=True)

    # ── 모델 로드 ────────────────────────────────────────────
    logger.info(f"SD 모델 로드: {model_id}")
    from diffusers import (AutoencoderKL, UNet2DConditionModel,
                           DDPMScheduler)
    from transformers import CLIPTextModel, CLIPTokenizer

    tokenizer   = CLIPTokenizer.from_pretrained(model_id, subfolder="tokenizer")
    text_encoder= CLIPTextModel.from_pretrained(model_id, subfolder="text_encoder")
    vae         = AutoencoderKL.from_pretrained(model_id, subfolder="vae")
    unet        = UNet2DConditionModel.from_pretrained(model_id, subfolder="unet")
    noise_sched = DDPMScheduler.from_pretrained(model_id, subfolder="scheduler")

    # VAE, text_encoder 동결
    vae.requires_grad_(False)
    text_encoder.requires_grad_(False)

    # LoRA 주입
    unet, lora_layers = inject_lora_to_unet(unet, rank=rank, alpha=alpha)

    # 마스크 인코더
    mask_encoder = MaskEncoder()

    vae.to(device)
    text_encoder.to(device)
    unet.to(device)
    mask_encoder.to(device)

    # ── 옵티마이저: LoRA + MaskEncoder만 학습 ────────────────
    params = (
        list(p for layer in lora_layers
             for p in [layer.lora_A.weight, layer.lora_B.weight])
        + list(mask_encoder.parameters())
    )
    optimizer = torch.optim.AdamW(params, lr=lr, weight_decay=1e-5)
    scaler    = torch.cuda.amp.GradScaler() if device == "cuda" else None

    # 빈 텍스트 임베딩 (텍스트 프롬프트 없이 마스크만 conditioning)
    with torch.no_grad():
        empty_tokens = tokenizer(
            [""] * batch_size,
            padding="max_length",
            max_length=tokenizer.model_max_length,
            return_tensors="pt"
        ).input_ids.to(device)
        empty_embeds = text_encoder(empty_tokens)[0]   # [B, 77, 768]

    # ── 데이터셋 ─────────────────────────────────────────────
    dataset = TumorPatchDataset(
        brats_dir, patch_size=patch_size,
        max_samples=max_samples, augment=True)
    loader  = DataLoader(dataset, batch_size=batch_size,
                         shuffle=True, num_workers=0, drop_last=True)

    logger.info(f"학습 시작: {num_epochs} epoch × {len(loader)} step")

    for epoch in range(1, num_epochs + 1):
        unet.train(); mask_encoder.train()
        total_loss = 0.0

        for batch in tqdm(loader, desc=f"Epoch {epoch}/{num_epochs}"):
            images = batch["image"].to(device)   # [B, 1, 128, 128]
            masks  = batch["mask"].to(device)    # [B, 1, 128, 128]

            # 그레이스케일 → 3채널 (SD 입력 형식)
            images_3ch = images.repeat(1, 3, 1, 1)  # [B, 3, 128, 128]

            with torch.cuda.amp.autocast(enabled=(scaler is not None)):
                # 1. 이미지를 latent로 인코딩
                latents = vae.encode(images_3ch).latent_dist.sample()
                latents = latents * vae.config.scaling_factor  # [B, 4, 16, 16]

                # 2. 노이즈 추가
                noise     = torch.randn_like(latents)
                timesteps = torch.randint(
                    0, noise_sched.config.num_train_timesteps,
                    (latents.shape[0],), device=device).long()
                noisy_latents = noise_sched.add_noise(latents, noise, timesteps)

                # 3. 마스크 인코딩 → latent에 더해서 conditioning
                mask_latents   = mask_encoder(masks)   # [B, 4, 16, 16]
                conditioned    = noisy_latents + mask_latents

                # 4. UNet 노이즈 예측
                text_embeds = empty_embeds[:latents.shape[0]]
                noise_pred  = unet(conditioned, timesteps,
                                   encoder_hidden_states=text_embeds).sample

                # 5. 손실 (예측된 노이즈 vs 실제 노이즈)
                loss = F.mse_loss(noise_pred, noise)

            optimizer.zero_grad()
            if scaler:
                scaler.scale(loss).backward()
                scaler.step(optimizer)
                scaler.update()
            else:
                loss.backward()
                optimizer.step()

            total_loss += loss.item()

        avg_loss = total_loss / len(loader)
        logger.info(f"Epoch {epoch}/{num_epochs} | avg_loss={avg_loss:.4f}")

        # 체크포인트 저장
        if epoch % save_every == 0:
            ckpt_dir = Path(output_dir) / f"epoch_{epoch:02d}"
            ckpt_dir.mkdir(parents=True, exist_ok=True)
            _save_checkpoint(unet, mask_encoder, lora_layers,
                             ckpt_dir, rank, alpha)
            logger.info(f"체크포인트 저장: {ckpt_dir}")

    # 최종 저장
    final_dir = Path(output_dir) / "lora_final"
    final_dir.mkdir(parents=True, exist_ok=True)
    _save_checkpoint(unet, mask_encoder, lora_layers,
                     final_dir, rank, alpha)
    logger.info(f"최종 모델 저장 완료: {final_dir}")


def _save_checkpoint(unet, mask_encoder, lora_layers, save_dir, rank, alpha):
    """LoRA 가중치 + MaskEncoder 저장."""
    import torch

    # LoRA 가중치
    lora_state = {}
    for i, layer in enumerate(lora_layers):
        lora_state[f"lora_{i}.A"] = layer.lora_A.weight.cpu()
        lora_state[f"lora_{i}.B"] = layer.lora_B.weight.cpu()
    torch.save(lora_state, save_dir / "lora_weights.pt")

    # MaskEncoder 가중치
    torch.save(mask_encoder.state_dict(),
               save_dir / "mask_encoder.pt")

    # 설정 저장
    config = {"rank": rank, "alpha": alpha,
              "method": "mask_to_patch_generation"}
    with open(save_dir / "config.json", "w") as f:
        import json
        json.dump(config, f, indent=2)


# ─────────────────────────────────────────────────────────────
# CLI
# ─────────────────────────────────────────────────────────────

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="LoRA Fine-tuning v2: 마스크 → 종양 패치 생성 학습")
    parser.add_argument("--brats_dir",   required=True)
    parser.add_argument("--output_dir",  required=True)
    parser.add_argument("--config",      required=True)
    parser.add_argument("--device",      default="cuda"
                        if torch.cuda.is_available() else "cpu")
    args = parser.parse_args()

    cfg = load_config(args.config)
    train_lora(cfg, args.brats_dir, args.output_dir, args.device)
