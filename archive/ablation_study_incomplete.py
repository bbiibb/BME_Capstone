"""
Phase 4: Ablation Study — 합성 데이터 유무에 따른 성능 비교

비교 조건:
  A. 베이스라인: BraTS 학습 데이터 + 기본 증강(회전/뒤집기)만 사용
  B. 제안 방법: 합성 종양 데이터 추가 학습

각 조건의 체크포인트를 평가하고 성능 차이를 시각화합니다.

사용법:
    python phase4_evaluation/ablation_study.py \
        --baseline_ckpt checkpoints/baseline/best.pth \
        --proposed_ckpt checkpoints/proposed/best.pth \
        --brats_dir data/processed/brats_slices \
        --config configs/config.yaml \
        --output_dir results/ablation
"""

import os
import sys
import argparse
import logging
from pathlib import Path

import numpy as np
import yaml
import torch
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from torch.utils.data import DataLoader

sys.path.insert(0, str(Path(__file__).parent.parent))
from phase3_segmentation.dataset import BraTSDataset
from phase3_segmentation.unet import build_model
from utils.metrics import compute_all_metrics
from utils.io_utils import save_json

logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)s | %(message)s")
logger = logging.getLogger(__name__)

METRICS = ["dice", "iou", "precision", "recall", "f1"]


def evaluate_model(checkpoint_path: str, dataset: BraTSDataset,
                    cfg: dict, device: torch.device,
                    batch_size: int = 16, threshold: float = 0.5) -> list[dict]:
    """모델 평가 → per-sample 결과 리스트 반환"""
    model = build_model(cfg).to(device)
    ckpt  = torch.load(checkpoint_path, map_location=device)
    model.load_state_dict(ckpt["model"])
    model.eval()

    loader  = DataLoader(dataset, batch_size=batch_size, shuffle=False, num_workers=2)
    results = []

    with torch.no_grad():
        for batch in loader:
            images = batch["image"].to(device)
            masks  = batch["mask"].numpy()
            logits = model(images)
            preds  = (torch.sigmoid(logits) > threshold).cpu().numpy().astype(np.uint8)

            for i in range(len(images)):
                m = compute_all_metrics(preds[i, 0], masks[i, 0])
                results.append(m)

    return results


def paired_ttest(scores_a: list[float], scores_b: list[float]) -> tuple[float, float]:
    """Paired t-test (scipy 기반)"""
    from scipy.stats import ttest_rel
    t_stat, p_val = ttest_rel(scores_a, scores_b)
    return float(t_stat), float(p_val)


def plot_comparison(results_a: list[dict], results_b: list[dict],
                     label_a: str, label_b: str, output_dir: str):
    """지표별 박스플롯 비교 + p-value 표시"""
    fig, axes = plt.subplots(1, len(METRICS), figsize=(5 * len(METRICS), 6))
    colors = {"baseline": "#4C72B0", "proposed": "#DD8452"}

    for ax, metric in zip(axes, METRICS):
        vals_a = [r[metric] for r in results_a if np.isfinite(r[metric])]
        vals_b = [r[metric] for r in results_b if np.isfinite(r[metric])]

        bp = ax.boxplot(
            [vals_a, vals_b],
            labels=[label_a, label_b],
            patch_artist=True,
            widths=0.5,
        )
        for patch, color in zip(bp["boxes"], [colors["baseline"], colors["proposed"]]):
            patch.set_facecolor(color)
            patch.set_alpha(0.7)

        # p-value
        if len(vals_a) == len(vals_b):
            t, p = paired_ttest(vals_a, vals_b)
            sig = "***" if p < 0.001 else "**" if p < 0.01 else "*" if p < 0.05 else "ns"
            ax.set_title(f"{metric.upper()}\n(p={p:.3f} {sig})", fontsize=11)
        else:
            ax.set_title(metric.upper(), fontsize=11)

        ax.set_ylim(0, 1.05)

        # 평균 수치 표시
        for j, (vals, color) in enumerate([(vals_a, colors["baseline"]),
                                             (vals_b, colors["proposed"])], 1):
            ax.text(j, np.mean(vals) + 0.02, f"{np.mean(vals):.3f}",
                    ha="center", fontsize=9, color="black", fontweight="bold")

        if metric == "dice":
            ax.axhline(0.85, color="red", linestyle="--", linewidth=1.5, label="목표(0.85)")
            ax.legend(fontsize=8)

    legend_patches = [
        mpatches.Patch(color=colors["baseline"], alpha=0.7, label=label_a),
        mpatches.Patch(color=colors["proposed"], alpha=0.7, label=label_b),
    ]
    fig.legend(handles=legend_patches, loc="lower center", ncol=2, fontsize=11,
               bbox_to_anchor=(0.5, -0.02))

    plt.suptitle("Ablation Study: 합성 데이터 추가 효과 비교", fontsize=14, y=1.02)
    plt.tight_layout()
    save_path = os.path.join(output_dir, "ablation_comparison.png")
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close()
    logger.info(f"비교 플롯 저장: {save_path}")


def plot_dice_improvement(results_a: list[dict], results_b: list[dict],
                           label_a: str, label_b: str, output_dir: str):
    """샘플별 Dice 개선량 히스토그램"""
    n = min(len(results_a), len(results_b))
    diff = [results_b[i]["dice"] - results_a[i]["dice"] for i in range(n)]

    fig, ax = plt.subplots(figsize=(9, 5))
    colors = ["#DD8452" if d > 0 else "#4C72B0" for d in diff]
    ax.bar(range(n), diff, color=colors, alpha=0.7, width=1.0)
    ax.axhline(0, color="black", linewidth=1)
    ax.axhline(np.mean(diff), color="red", linestyle="--",
               label=f"평균 개선량={np.mean(diff):+.4f}")
    ax.set_xlabel("샘플 인덱스")
    ax.set_ylabel("Dice 개선량 (제안 - 베이스라인)")
    ax.set_title(f"샘플별 Dice 개선량 ({label_b} vs {label_a})")
    ax.legend()

    plt.tight_layout()
    save_path = os.path.join(output_dir, "dice_improvement.png")
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close()
    logger.info(f"개선량 플롯 저장: {save_path}")


def main():
    parser = argparse.ArgumentParser(description="Ablation Study")
    parser.add_argument("--baseline_ckpt", required=True)
    parser.add_argument("--proposed_ckpt", required=True)
    parser.add_argument("--brats_dir",     default="data/processed/brats_slices")
    parser.add_argument("--config",        default="configs/config.yaml")
    parser.add_argument("--output_dir",    default="results/ablation")
    parser.add_argument("--batch_size",    type=int, default=16)
    args = parser.parse_args()

    with open(args.config) as f:
        cfg = yaml.safe_load(f)

    os.makedirs(args.output_dir, exist_ok=True)

    device = (torch.device("cuda") if torch.cuda.is_available()
              else torch.device("mps") if torch.backends.mps.is_available()
              else torch.device("cpu"))

    target_size = tuple(cfg["preprocessing"]["target_size"])
    try:
        test_dataset = BraTSDataset(args.brats_dir, target_size, augment=False, binary_mask=True)
    except FileNotFoundError as e:
        logger.error(str(e))
        return

    logger.info(f"테스트 샘플: {len(test_dataset)}개")

    label_a = "Baseline (증강만)"
    label_b = "제안 (합성 데이터 추가)"

    logger.info("베이스라인 평가 중...")
    results_a = evaluate_model(args.baseline_ckpt, test_dataset, cfg, device, args.batch_size)

    logger.info("제안 방법 평가 중...")
    results_b = evaluate_model(args.proposed_ckpt, test_dataset, cfg, device, args.batch_size)

    # 요약 출력
    logger.info("\n─── Ablation Study 결과 ─────────────────────────────")
    logger.info(f"{'지표':<12} {'베이스라인':>12} {'제안 방법':>12} {'개선량':>10} {'p-value':>10}")
    logger.info("─" * 60)

    ablation_summary = {}
    for metric in METRICS:
        vals_a = [r[metric] for r in results_a if np.isfinite(r[metric])]
        vals_b = [r[metric] for r in results_b if np.isfinite(r[metric])]
        mean_a, mean_b = np.mean(vals_a), np.mean(vals_b)
        delta = mean_b - mean_a

        t, p = paired_ttest(vals_a, vals_b) if len(vals_a) == len(vals_b) else (0, 1)
        sig = "***" if p < 0.001 else "**" if p < 0.01 else "*" if p < 0.05 else "ns"

        logger.info(f"{metric:<12} {mean_a:>12.4f} {mean_b:>12.4f} {delta:>+10.4f} {p:>8.4f} {sig}")
        ablation_summary[metric] = {
            "baseline": float(mean_a), "proposed": float(mean_b),
            "delta": float(delta), "p_value": float(p), "significant": sig
        }

    # 결과 저장
    save_json({"summary": ablation_summary,
               "baseline_samples": results_a,
               "proposed_samples": results_b},
              os.path.join(args.output_dir, "ablation_results.json"))

    # 시각화
    plot_comparison(results_a, results_b, label_a, label_b, args.output_dir)
    plot_dice_improvement(results_a, results_b, label_a, label_b, args.output_dir)

    logger.info(f"\n결과 저장 완료: {args.output_dir}")


if __name__ == "__main__":
    main()
