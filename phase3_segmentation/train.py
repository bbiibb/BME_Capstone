"""
Phase 3: U-Net 종양 분할 모델 학습

사용법:
    python phase3_segmentation/train.py \
        --synthetic_dir data/synthetic \
        --config configs/config.yaml \
        --exp_name baseline_synthetic

옵션:
    --brats_dir data/processed/brats_slices  # 실제 BraTS 데이터 추가 시
    --real_ratio 0.3                         # 실제 데이터 비율
    --resume checkpoints/exp001/last.pth     # 이어서 학습
    --drive_ckpt_dir /content/drive/.../checkpoints/exp001  # 드라이브 동기화
"""

import os
import sys
import argparse
import logging
import shutil
from pathlib import Path

import yaml
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from tqdm import tqdm

sys.path.insert(0, str(Path(__file__).parent.parent))
from phase3_segmentation.dataset import SyntheticTumorDataset, BraTSDataset, MixedDataset
from phase3_segmentation.unet import build_model
from utils.metrics import DiceLoss, DiceBCELoss, dice_coefficient

logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)s | %(message)s")
logger = logging.getLogger(__name__)


# ──────────────────────────────────────────────────────────────────────────────
# 학습 유틸리티
# ──────────────────────────────────────────────────────────────────────────────

def get_loss_fn(cfg: dict) -> nn.Module:
    loss_type = cfg["segmentation"]["loss"]
    if loss_type == "dice":
        return DiceLoss()
    elif loss_type == "bce":
        return nn.BCEWithLogitsLoss()
    elif loss_type == "dice_bce":
        return DiceBCELoss(
            dice_weight=cfg["segmentation"]["dice_weight"],
            bce_weight=cfg["segmentation"]["bce_weight"],
        )
    raise ValueError(f"Unknown loss: {loss_type}")


def get_scheduler(optimizer, cfg: dict, n_steps: int):
    sched_type = cfg["segmentation"]["scheduler"]
    warmup = cfg["segmentation"]["warmup_epochs"]

    if sched_type == "cosine":
        from torch.optim.lr_scheduler import CosineAnnealingLR, LinearLR, SequentialLR
        warmup_sched = LinearLR(optimizer, start_factor=0.1, total_iters=warmup)
        cosine_sched = CosineAnnealingLR(optimizer, T_max=n_steps - warmup, eta_min=1e-7)
        return SequentialLR(optimizer, [warmup_sched, cosine_sched], milestones=[warmup])
    elif sched_type == "step":
        from torch.optim.lr_scheduler import StepLR
        return StepLR(optimizer, step_size=30, gamma=0.5)
    else:
        return None


def train_one_epoch(model, loader, optimizer, criterion, device, scaler=None):
    model.train()
    total_loss = 0.0
    for batch in tqdm(loader, desc="  학습", leave=False):
        images = batch["image"].to(device)
        masks  = batch["mask"].to(device)

        optimizer.zero_grad()

        if scaler is not None:
            with torch.autocast(device_type=device.type, dtype=torch.float16):
                logits = model(images)
                loss   = criterion(logits, masks)
            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            scaler.step(optimizer)
            scaler.update()
        else:
            logits = model(images)
            loss   = criterion(logits, masks)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()

        total_loss += loss.item()

    return total_loss / len(loader)


@torch.no_grad()
def validate(model, loader, criterion, device):
    model.eval()
    total_loss = 0.0
    dice_scores = []

    for batch in tqdm(loader, desc="  검증", leave=False):
        images = batch["image"].to(device)
        masks  = batch["mask"].to(device)

        logits = model(images)
        loss   = criterion(logits, masks)
        total_loss += loss.item()

        preds = (torch.sigmoid(logits) > 0.5).cpu().numpy()
        gt    = masks.cpu().numpy()
        for p, g in zip(preds, gt):
            dice_scores.append(dice_coefficient(p[0], g[0]))

    return total_loss / len(loader), float(np.mean(dice_scores))


def save_checkpoint(state: dict, path: str):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    torch.save(state, path)


def sync_to_drive(local_ckpt_dir: str, drive_ckpt_dir: str):
    """로컬 체크포인트를 드라이브로 동기화 (매 epoch 호출)"""
    if not drive_ckpt_dir:
        return
    try:
        os.makedirs(drive_ckpt_dir, exist_ok=True)
        # last.pth, best.pth만 동기화 (용량 절약)
        for fname in ["last.pth", "best.pth"]:
            src = os.path.join(local_ckpt_dir, fname)
            dst = os.path.join(drive_ckpt_dir, fname)
            if os.path.exists(src):
                shutil.copy2(src, dst)
        logger.info(f"  드라이브 동기화 완료: {drive_ckpt_dir}")
    except Exception as e:
        logger.warning(f"  드라이브 동기화 실패 (무시): {e}")


# ──────────────────────────────────────────────────────────────────────────────
# 메인
# ──────────────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="종양 분할 모델 학습")
    parser.add_argument("--synthetic_dir",  default="data/synthetic")
    parser.add_argument("--brats_dir",      default=None,
                        help="실제 BraTS 슬라이스 경로 (선택)")
    parser.add_argument("--real_ratio",     type=float, default=0.0)
    parser.add_argument("--config",         default="configs/config.yaml")
    parser.add_argument("--exp_name",       default="exp001")
    parser.add_argument("--resume",         default=None,
                        help="이어서 학습할 last.pth 경로")
    parser.add_argument("--drive_ckpt_dir", default=None,
                        help="드라이브 체크포인트 동기화 경로 (런타임 끊김 대비)")
    args = parser.parse_args()

    # drive_ckpt_dir 환경변수 fallback
    drive_ckpt_dir = args.drive_ckpt_dir or os.environ.get("DRIVE_CKPT_DIR", "")

    with open(args.config) as f:
        cfg = yaml.safe_load(f)

    seg = cfg["segmentation"]
    target_size = tuple(cfg["preprocessing"]["target_size"])

    # 디바이스
    if torch.cuda.is_available():
        device = torch.device("cuda")
    elif torch.backends.mps.is_available():
        device = torch.device("mps")
    else:
        device = torch.device("cpu")
    logger.info(f"디바이스: {device}")

    # ── 데이터셋 ──────────────────────────────────────────────────────────────
    if args.real_ratio >= 1.0 and args.brats_dir:
        logger.info("베이스라인 모드: BraTS 실제 데이터만 사용 (real_ratio=1.0)")
        train_ds = BraTSDataset(args.brats_dir, target_size, augment=True)
        val_ds   = BraTSDataset(args.brats_dir, target_size, augment=False)
    else:
        train_ds = SyntheticTumorDataset(
            args.synthetic_dir, target_size, augment=True,
            split="train", val_ratio=seg["val_ratio"], test_ratio=seg["test_ratio"])
        val_ds = SyntheticTumorDataset(
            args.synthetic_dir, target_size, augment=False,
            split="val", val_ratio=seg["val_ratio"], test_ratio=seg["test_ratio"])
        if args.brats_dir and args.real_ratio > 0:
            try:
                brats_train = BraTSDataset(args.brats_dir, target_size, augment=True)
                train_ds = MixedDataset(train_ds, brats_train, real_ratio=args.real_ratio)
                logger.info(f"BraTS 혼합 학습 활성화 (real_ratio={args.real_ratio})")
            except FileNotFoundError as e:
                logger.warning(f"BraTS 데이터 로드 실패: {e}")

    num_workers = seg.get("num_workers", 2)
    train_loader = DataLoader(train_ds, batch_size=seg["batch_size"], shuffle=True,
                               num_workers=num_workers, pin_memory=True)
    val_loader   = DataLoader(val_ds,   batch_size=seg["batch_size"], shuffle=False,
                               num_workers=num_workers, pin_memory=True)

    logger.info(f"학습 샘플: {len(train_ds)} | 검증 샘플: {len(val_ds)}")

    # ── 모델 ──────────────────────────────────────────────────────────────────
    model     = build_model(cfg).to(device)
    logger.info(f"모델: {cfg['segmentation']['model']} | "
                f"파라미터: {sum(p.numel() for p in model.parameters()):,}")
    criterion = get_loss_fn(cfg)
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=seg["learning_rate"],
        weight_decay=seg["weight_decay"]
    )
    scheduler = get_scheduler(optimizer, cfg, seg["num_epochs"])

    use_amp = seg.get("mixed_precision", False) and device.type == "cuda"
    scaler  = torch.amp.GradScaler("cuda") if use_amp else None

    # ── 체크포인트 복원 ───────────────────────────────────────────────────────
    start_epoch = 0
    best_dice   = 0.0
    ckpt_dir    = os.path.join("checkpoints", args.exp_name)
    os.makedirs(ckpt_dir, exist_ok=True)

    if args.resume and os.path.exists(args.resume):
        logger.info(f"체크포인트 로드: {args.resume}")
        ckpt = torch.load(args.resume, map_location=device)
        model.load_state_dict(ckpt["model"])
        optimizer.load_state_dict(ckpt["optimizer"])
        # scheduler 복원
        if scheduler and ckpt.get("scheduler"):
            try:
                scheduler.load_state_dict(ckpt["scheduler"])
            except Exception:
                logger.warning("scheduler 복원 실패 — 초기값 사용")
        start_epoch = ckpt["epoch"] + 1
        best_dice   = ckpt.get("best_dice", 0.0)
        logger.info(f"복원 완료: epoch {start_epoch}부터 재개 | best_dice={best_dice:.4f}")
    else:
        if drive_ckpt_dir:
            logger.info(f"드라이브 동기화 경로: {drive_ckpt_dir}")

    # ── 학습 루프 ─────────────────────────────────────────────────────────────
    history = {"train_loss": [], "val_loss": [], "val_dice": []}

    for epoch in range(start_epoch, seg["num_epochs"]):
        logger.info(f"\nEpoch [{epoch + 1}/{seg['num_epochs']}] "
                    f"lr={optimizer.param_groups[0]['lr']:.2e}")

        train_loss         = train_one_epoch(model, train_loader, optimizer,
                                             criterion, device, scaler)
        val_loss, val_dice = validate(model, val_loader, criterion, device)

        if scheduler:
            scheduler.step()

        history["train_loss"].append(train_loss)
        history["val_loss"].append(val_loss)
        history["val_dice"].append(val_dice)

        logger.info(f"  train_loss={train_loss:.4f} | val_loss={val_loss:.4f} "
                    f"| val_dice={val_dice:.4f} {'★' if val_dice > best_dice else ''}")

        # ── Best 모델 저장 ────────────────────────────────────────────────────
        if val_dice > best_dice:
            best_dice = val_dice
            save_checkpoint(
                {"epoch": epoch, "model": model.state_dict(),
                 "optimizer": optimizer.state_dict(),
                 "scheduler": scheduler.state_dict() if scheduler else None,
                 "best_dice": best_dice, "cfg": cfg},
                os.path.join(ckpt_dir, "best.pth")
            )
            logger.info(f"  best.pth 저장 (dice={best_dice:.4f})")

        # ── last.pth 저장 (매 epoch — 런타임 끊겨도 재개 가능) ───────────────
        save_checkpoint(
            {"epoch": epoch, "model": model.state_dict(),
             "optimizer": optimizer.state_dict(),
             "scheduler": scheduler.state_dict() if scheduler else None,
             "best_dice": best_dice, "cfg": cfg},
            os.path.join(ckpt_dir, "last.pth")
        )

        # ── 드라이브 동기화 (매 epoch) ───────────────────────────────────────
        # 런타임이 끊겨도 드라이브에 최신 체크포인트가 유지됨
        sync_to_drive(ckpt_dir, drive_ckpt_dir)

        # ── 주기적 체크포인트 (10 epoch마다) ─────────────────────────────────
        if (epoch + 1) % 10 == 0:
            save_checkpoint(
                {"epoch": epoch, "model": model.state_dict(),
                 "optimizer": optimizer.state_dict(),
                 "scheduler": scheduler.state_dict() if scheduler else None,
                 "best_dice": best_dice},
                os.path.join(ckpt_dir, f"epoch{epoch + 1:04d}.pth")
            )

    logger.info(f"\n학습 완료! Best Val Dice: {best_dice:.4f}")
    if best_dice >= cfg["evaluation"]["dice_threshold"]:
        logger.info(f"목표 Dice({cfg['evaluation']['dice_threshold']}) 달성!")
    else:
        logger.warning(f"목표 Dice 미달 (현재: {best_dice:.4f})")

    # ── 학습 이력 저장 ────────────────────────────────────────────────────────
    import json
    hist_path = os.path.join(ckpt_dir, "history.json")
    with open(hist_path, "w") as f:
        json.dump(history, f, indent=2)
    logger.info(f"학습 이력 저장: {hist_path}")

    # 최종 드라이브 동기화
    sync_to_drive(ckpt_dir, drive_ckpt_dir)


if __name__ == "__main__":
    main()
