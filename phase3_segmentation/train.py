"""
Phase 3: U-Net Tumor Segmentation Model Training

Usage:
    # Baseline (real data only, lr=1e-4 cosine+warmup, num_epochs from config)
    python phase3_segmentation/train.py \
        --synthetic_dir data/synthetic \
        --brats_dir data/processed/brats_slices \
        --real_ratio 1.0 \
        --config configs/config.yaml \
        --exp_name seg_baseline

    # Proposed method (fine-tune from the baseline checkpoint, lr=5e-6
    # cosine annealing, finetune_num_epochs from config)
    python phase3_segmentation/train.py \
        --synthetic_dir data/synthetic \
        --brats_dir data/processed/brats_slices \
        --real_ratio 0.3 \
        --config configs/config.yaml \
        --exp_name seg_proposed \
        --finetune_from checkpoints/seg_baseline/best.pth

Options:
    --brats_dir data/processed/brats_slices  # to add real BraTS data
    --real_ratio 0.3                         # real-data mixing ratio
    --resume checkpoints/exp001/last.pth     # resume the same run (in case of runtime disconnects)
    --finetune_from checkpoints/seg_baseline/best.pth  # start fine-tuning from baseline weights
    --drive_ckpt_dir /path/to/drive/checkpoints/exp001  # optional Drive sync path
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
# Training utilities
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


def get_finetune_scheduler(optimizer, n_steps: int):
    """Fine-tuning stage scheduler: plain cosine annealing, no warmup.

    Per Final Report Table 1: the baseline uses Cosine + Warmup at
    lr=1e-4, while the fine-tuning stage (proposed model, resumed from
    the baseline checkpoint) uses Cosine Annealing only at lr=5e-6.
    """
    from torch.optim.lr_scheduler import CosineAnnealingLR
    return CosineAnnealingLR(optimizer, T_max=n_steps, eta_min=1e-8)


def train_one_epoch(model, loader, optimizer, criterion, device, scaler=None):
    model.train()
    total_loss = 0.0
    for batch in tqdm(loader, desc="  train", leave=False):
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

    for batch in tqdm(loader, desc="  val", leave=False):
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
    """Sync local checkpoints to Drive (called every epoch)"""
    if not drive_ckpt_dir:
        return
    try:
        os.makedirs(drive_ckpt_dir, exist_ok=True)
        # Only sync last.pth and best.pth (save space)
        for fname in ["last.pth", "best.pth"]:
            src = os.path.join(local_ckpt_dir, fname)
            dst = os.path.join(drive_ckpt_dir, fname)
            if os.path.exists(src):
                shutil.copy2(src, dst)
        logger.info(f"  Checkpoint sync complete: {drive_ckpt_dir}")
    except Exception as e:
        logger.warning(f"  Checkpoint sync failed (ignored): {e}")


# ──────────────────────────────────────────────────────────────────────────────
# Main
# ──────────────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="Tumor segmentation model training")
    parser.add_argument("--synthetic_dir",  default="data/synthetic")
    parser.add_argument("--brats_dir",      default=None,
                        help="Path to real BraTS slices (optional)")
    parser.add_argument("--real_ratio",     type=float, default=0.0)
    parser.add_argument("--config",         default="configs/config.yaml")
    parser.add_argument("--exp_name",       default="exp001")
    parser.add_argument("--resume",         default=None,
                        help="Path to last.pth to resume the same run (also restores LR/scheduler state)")
    parser.add_argument("--finetune_from",  default=None,
                        help="Load weights only from the baseline checkpoint and start fine-tuning "
                             "at a lower LR (uses finetune_learning_rate / finetune_num_epochs / "
                             "finetune_scheduler from config, matching Final Report Table 1)")
    parser.add_argument("--drive_ckpt_dir", default=None,
                        help="Optional checkpoint sync path (in case of runtime disconnects)")
    args = parser.parse_args()

    if args.resume and args.finetune_from:
        raise ValueError("--resume and --finetune_from cannot be used together "
                          "(--resume: continue the same run / --finetune_from: start a new fine-tuning run)")

    # drive_ckpt_dir env var fallback
    drive_ckpt_dir = args.drive_ckpt_dir or os.environ.get("DRIVE_CKPT_DIR", "")

    with open(args.config) as f:
        cfg = yaml.safe_load(f)

    seg = cfg["segmentation"]
    target_size = tuple(cfg["preprocessing"]["target_size"])

    # Device
    if torch.cuda.is_available():
        device = torch.device("cuda")
    elif torch.backends.mps.is_available():
        device = torch.device("mps")
    else:
        device = torch.device("cpu")
    logger.info(f"device: {device}")

    # -- Datasets --
    if args.real_ratio >= 1.0 and args.brats_dir:
        logger.info("Baseline mode: using real BraTS data only (real_ratio=1.0)")
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
                logger.info(f"BraTS mixed training enabled (real_ratio={args.real_ratio})")
            except FileNotFoundError as e:
                logger.warning(f"Failed to load BraTS data: {e}")

    num_workers = seg.get("num_workers", 2)
    train_loader = DataLoader(train_ds, batch_size=seg["batch_size"], shuffle=True,
                               num_workers=num_workers, pin_memory=True)
    val_loader   = DataLoader(val_ds,   batch_size=seg["batch_size"], shuffle=False,
                               num_workers=num_workers, pin_memory=True)

    logger.info(f"Train samples: {len(train_ds)} | Val samples: {len(val_ds)}")

    # -- Model --
    model     = build_model(cfg).to(device)
    logger.info(f"model: {cfg['segmentation']['model']} | "
                f"Params: {sum(p.numel() for p in model.parameters()):,}")
    criterion = get_loss_fn(cfg)

    is_finetune = args.finetune_from is not None
    if is_finetune:
        # Load only the weights from the baseline checkpoint (optimizer/scheduler
        # state is freshly initialized -- fine-tuning is a separate stage that
        # starts anew at a lower LR)
        if not os.path.exists(args.finetune_from):
            raise FileNotFoundError(f"finetune_from checkpoint not found: {args.finetune_from}")
        logger.info(f"Loading baseline weights (starting fine-tuning): {args.finetune_from}")
        base_ckpt = torch.load(args.finetune_from, map_location=device)
        model.load_state_dict(base_ckpt["model"])

        lr = seg["finetune_learning_rate"]
        num_epochs = seg["finetune_num_epochs"]
        logger.info(f"Fine-tuning mode: lr={lr:.1e} | epochs={num_epochs} | "
                     f"scheduler=cosine (no warmup)")
    else:
        lr = seg["learning_rate"]
        num_epochs = seg["num_epochs"]

    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=lr,
        weight_decay=seg["weight_decay"]
    )

    if is_finetune:
        scheduler = get_finetune_scheduler(optimizer, num_epochs)
    else:
        scheduler = get_scheduler(optimizer, cfg, num_epochs)

    use_amp = seg.get("mixed_precision", False) and device.type == "cuda"
    scaler  = torch.amp.GradScaler("cuda") if use_amp else None

    # -- Checkpoint restoration --
    start_epoch = 0
    best_dice   = 0.0
    ckpt_dir    = os.path.join("checkpoints", args.exp_name)
    os.makedirs(ckpt_dir, exist_ok=True)

    if args.resume and os.path.exists(args.resume):
        logger.info(f"Loading checkpoint: {args.resume}")
        ckpt = torch.load(args.resume, map_location=device)
        model.load_state_dict(ckpt["model"])
        optimizer.load_state_dict(ckpt["optimizer"])
        # Restore scheduler state
        if scheduler and ckpt.get("scheduler"):
            try:
                scheduler.load_state_dict(ckpt["scheduler"])
            except Exception:
                logger.warning("Failed to restore scheduler -- using initial state")
        start_epoch = ckpt["epoch"] + 1
        best_dice   = ckpt.get("best_dice", 0.0)
        logger.info(f"Restore complete: resuming from epoch {start_epoch} | best_dice={best_dice:.4f}")
    else:
        if drive_ckpt_dir:
            logger.info(f"Checkpoint sync path: {drive_ckpt_dir}")

    # -- Training loop --
    history = {"train_loss": [], "val_loss": [], "val_dice": []}

    for epoch in range(start_epoch, num_epochs):
        logger.info(f"\nEpoch [{epoch + 1}/{num_epochs}] "
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

        # -- Save best model ────────────────────────────────────────────────────
        if val_dice > best_dice:
            best_dice = val_dice
            save_checkpoint(
                {"epoch": epoch, "model": model.state_dict(),
                 "optimizer": optimizer.state_dict(),
                 "scheduler": scheduler.state_dict() if scheduler else None,
                 "best_dice": best_dice, "cfg": cfg},
                os.path.join(ckpt_dir, "best.pth")
            )
            logger.info(f"  Saved best.pth (dice={best_dice:.4f})")

        # -- Save last.pth every epoch (allows resuming after disconnects) --
        save_checkpoint(
            {"epoch": epoch, "model": model.state_dict(),
             "optimizer": optimizer.state_dict(),
             "scheduler": scheduler.state_dict() if scheduler else None,
             "best_dice": best_dice, "cfg": cfg},
            os.path.join(ckpt_dir, "last.pth")
        )

        # -- Optional checkpoint sync (every epoch) --
        # Keeps the latest checkpoint available even if the runtime disconnects
        sync_to_drive(ckpt_dir, drive_ckpt_dir)

        # -- Periodic checkpoint (every 10 epochs) --
        if (epoch + 1) % 10 == 0:
            save_checkpoint(
                {"epoch": epoch, "model": model.state_dict(),
                 "optimizer": optimizer.state_dict(),
                 "scheduler": scheduler.state_dict() if scheduler else None,
                 "best_dice": best_dice},
                os.path.join(ckpt_dir, f"epoch{epoch + 1:04d}.pth")
            )

    logger.info(f"\nTraining complete! Best Val Dice: {best_dice:.4f}")
    if best_dice >= cfg["evaluation"]["dice_threshold"]:
        logger.info(f"Reached target Dice ({cfg['evaluation']['dice_threshold']})!")
    else:
        logger.warning(f"Did not reach target Dice (current: {best_dice:.4f})")

    # -- Save training history --
    import json
    hist_path = os.path.join(ckpt_dir, "history.json")
    with open(hist_path, "w") as f:
        json.dump(history, f, indent=2)
    logger.info(f"Saved training history: {hist_path}")

    # Final checkpoint sync
    sync_to_drive(ckpt_dir, drive_ckpt_dir)


if __name__ == "__main__":
    main()
