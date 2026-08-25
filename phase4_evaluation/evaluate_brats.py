"""
Phase 4: BraTS 실제 데이터로 분할 모델 성능 평가

저장된 체크포인트를 불러와 BraTS 테스트 슬라이스에 대해 평가합니다.
목표: Dice Coefficient ≥ 0.85

사용법:
    python phase4_evaluation/evaluate_brats.py \
        --checkpoint checkpoints/exp001/best.pth \
        --brats_dir data/processed/brats_slices \
        --config configs/config.yaml \
        --output_dir results/brats_eval
"""

import os
import sys
import argparse
import logging
from pathlib import Path

import numpy as np
import yaml
import torch
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from tqdm import tqdm

sys.path.insert(0, str(Path(__file__).parent.parent))
from phase3_segmentation.dataset import BraTSDataset
from phase3_segmentation.unet import build_model
from utils.metrics import compute_all_metrics
from utils.io_utils import save_json

logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)s | %(message)s")
logger = logging.getLogger(__name__)


def load_model(checkpoint_path: str, cfg: dict, device: torch.device) -> torch.nn.Module:
    model = build_model(cfg).to(device)
    ckpt = torch.load(checkpoint_path, map_location=device)
    model.load_state_dict(ckpt["model"])
    model.eval()
    logger.info(f"모델 로드: {checkpoint_path} (epoch {ckpt.get('epoch', '?')})")
    return model


@torch.no_grad()
def run_evaluation(model, loader, device, threshold: float = 0.5) -> list[dict]:
    results = []
    for batch in tqdm(loader, desc="평가 중"):
        images = batch["image"].to(device)
        masks  = batch["mask"].numpy()
        paths  = batch["path"]

        logits = model(images)
        probs  = torch.sigmoid(logits).cpu().numpy()
        preds  = (probs > threshold).astype(np.uint8)

        for i in range(len(images)):
            metrics = compute_all_metrics(preds[i, 0], masks[i, 0])
            results.append({"path": paths[i], **metrics})

    return results


def summarize_results(results: list[dict]) -> dict:
    """결과 통계 요약"""
    metrics_keys = ["dice", "iou", "precision", "recall", "f1", "hausdorff"]
    summary = {}
    for k in metrics_keys:
        vals = [r[k] for r in results if np.isfinite(r[k])]
        if vals:
            summary[k] = {
                "mean":   float(np.mean(vals)),
                "std":    float(np.std(vals)),
                "median": float(np.median(vals)),
                "min":    float(np.min(vals)),
                "max":    float(np.max(vals)),
            }
    return summary


def plot_qualitative(model, dataset, device, output_dir: str,
                      n_samples: int = 12, threshold: float = 0.5):
    """정성 평가: 원본 / GT / 예측 / Overlay 4열 시각화"""
    indices = np.random.choice(len(dataset), min(n_samples, len(dataset)), replace=False)
    n_rows = len(indices)

    fig, axes = plt.subplots(n_rows, 4, figsize=(16, 4 * n_rows))
    if n_rows == 1:
        axes = axes[np.newaxis, :]

    col_titles = ["MRI (T1ce)", "Ground Truth", "예측 (Predicted)", "Overlay"]
    for col, title in enumerate(col_titles):
        axes[0, col].set_title(title, fontsize=13)

    model.eval()
    with torch.no_grad():
        for row, idx in enumerate(indices):
            sample = dataset[idx]
            image  = sample["image"].unsqueeze(0).to(device)
            mask   = sample["mask"].numpy()[0]

            logit  = model(image)
            pred   = (torch.sigmoid(logit) > threshold).cpu().numpy()[0, 0]

            img_np = sample["image"].numpy()[0]
            dice   = compute_all_metrics(pred, mask)["dice"]

            axes[row, 0].imshow(img_np, cmap="gray")
            axes[row, 1].imshow(mask, cmap="Reds", vmin=0, vmax=1)
            axes[row, 2].imshow(pred, cmap="Blues", vmin=0, vmax=1)

            # Overlay: 이미지 + GT(빨강) + 예측(파랑)
            axes[row, 3].imshow(img_np, cmap="gray")
            axes[row, 3].contour(mask, levels=[0.5], colors="red",  linewidths=1.5)
            axes[row, 3].contour(pred, levels=[0.5], colors="cyan", linewidths=1.5)
            axes[row, 3].set_xlabel(f"Dice={dice:.3f}", fontsize=10)

            for ax in axes[row]:
                ax.axis("off")

    plt.tight_layout()
    save_path = os.path.join(output_dir, "qualitative_results.png")
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close()
    logger.info(f"정성 결과 저장: {save_path}")


def plot_metrics_distribution(results: list[dict], output_dir: str):
    """Dice 점수 분포 히스토그램"""
    dice_scores = [r["dice"] for r in results]

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))

    # 히스토그램
    ax1.hist(dice_scores, bins=30, color="steelblue", edgecolor="white", alpha=0.8)
    ax1.axvline(np.mean(dice_scores), color="red",    linestyle="--", label=f"평균={np.mean(dice_scores):.3f}")
    ax1.axvline(np.median(dice_scores), color="green", linestyle="--", label=f"중앙값={np.median(dice_scores):.3f}")
    ax1.axvline(0.85, color="orange", linestyle="-",  linewidth=2, label="목표(0.85)")
    ax1.set_xlabel("Dice Coefficient")
    ax1.set_ylabel("빈도")
    ax1.set_title("Dice 점수 분포")
    ax1.legend()

    # Box plot for all metrics
    metrics_keys = ["dice", "iou", "precision", "recall", "f1"]
    data = [[r[k] for r in results] for k in metrics_keys]
    ax2.boxplot(data, labels=metrics_keys, patch_artist=True,
                boxprops=dict(facecolor="steelblue", alpha=0.6))
    ax2.set_title("분할 지표 분포")
    ax2.set_ylabel("Score")
    ax2.axhline(0.85, color="orange", linestyle="--", label="목표 Dice=0.85")
    ax2.legend()

    plt.tight_layout()
    save_path = os.path.join(output_dir, "metrics_distribution.png")
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close()
    logger.info(f"지표 분포 저장: {save_path}")


def main():
    parser = argparse.ArgumentParser(description="BraTS 평가")
    parser.add_argument("--checkpoint",  required=True)
    parser.add_argument("--brats_dir",   default="data/processed/brats_slices")
    parser.add_argument("--config",      default="configs/config.yaml")
    parser.add_argument("--output_dir",  default="results/brats_eval")
    parser.add_argument("--threshold",   type=float, default=0.5)
    parser.add_argument("--batch_size",  type=int,   default=16)
    args = parser.parse_args()

    with open(args.config) as f:
        cfg = yaml.safe_load(f)

    os.makedirs(args.output_dir, exist_ok=True)

    device = (torch.device("cuda") if torch.cuda.is_available()
              else torch.device("mps") if torch.backends.mps.is_available()
              else torch.device("cpu"))
    logger.info(f"디바이스: {device}")

    # 데이터셋
    target_size = tuple(cfg["preprocessing"]["target_size"])
    try:
        dataset = BraTSDataset(args.brats_dir, target_size, augment=False, binary_mask=True)
    except FileNotFoundError as e:
        logger.error(str(e))
        return

    loader = DataLoader(dataset, batch_size=args.batch_size, shuffle=False, num_workers=4)
    logger.info(f"평가 샘플: {len(dataset)}개")

    # 모델 로드 & 평가
    model   = load_model(args.checkpoint, cfg, device)
    results = run_evaluation(model, loader, device, args.threshold)
    summary = summarize_results(results)

    # 결과 출력
    logger.info("\n─── 평가 결과 요약 ───────────────────────────────")
    target_dice = cfg["evaluation"]["dice_threshold"]
    for k, v in summary.items():
        bar = "★" if k == "dice" and v["mean"] >= target_dice else ""
        logger.info(f"  {k:12s}: mean={v['mean']:.4f} ± {v['std']:.4f}  "
                    f"[{v['min']:.3f} ~ {v['max']:.3f}] {bar}")

    if "dice" in summary:
        achieved = summary["dice"]["mean"] >= target_dice
        logger.info(f"\n목표 Dice({target_dice}) {'달성 ✓' if achieved else '미달 ✗'}")

    # 저장
    save_json({"summary": summary, "per_sample": results},
              os.path.join(args.output_dir, "evaluation_results.json"))

    # 시각화
    plot_metrics_distribution(results, args.output_dir)
    plot_qualitative(model, dataset, device, args.output_dir, n_samples=12)

    logger.info(f"\n모든 결과 저장 완료: {args.output_dir}")


if __name__ == "__main__":
    main()
