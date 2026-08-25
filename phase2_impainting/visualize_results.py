"""
Phase 2 결과 시각화: 원본 / 마스크 / 합성 결과 3열 비교 플롯
"""

import os
import sys
import argparse
from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec

sys.path.insert(0, str(Path(__file__).parent.parent))
from utils.io_utils import load_png_as_float, load_mask_png, load_json


def visualize_samples(synthetic_dir: str, n_samples: int = 8, save_path: str = None):
    meta_path = os.path.join(synthetic_dir, "metadata.json")
    if not os.path.exists(meta_path):
        print(f"메타데이터 없음: {meta_path}")
        return

    meta = load_json(meta_path)
    samples = meta["samples"][:n_samples]

    n_cols = 3  # 원본 | 마스크 | 합성
    n_rows = len(samples)
    fig = plt.figure(figsize=(12, 4 * n_rows))
    gs = gridspec.GridSpec(n_rows, n_cols, hspace=0.05, wspace=0.05)

    col_titles = ["정상 뇌 MRI (원본)", "종양 마스크", "합성 뇌종양 MRI"]

    for row, sample in enumerate(samples):
        syn_img_path = os.path.join(synthetic_dir, sample["image"])
        syn_mask_path = os.path.join(synthetic_dir, sample["mask"])

        syn_img = load_png_as_float(syn_img_path)
        syn_mask = load_mask_png(syn_mask_path)

        # 원본 슬라이스 경로 복원 시도
        src_slice = sample.get("source_slice", "")
        src_path = syn_img_path.replace("images/", "").replace("_mask", "_orig")

        imgs = [None, syn_mask, syn_img]

        for col in range(n_cols):
            ax = fig.add_subplot(gs[row, col])
            if col == 1:
                ax.imshow(syn_mask, cmap="Reds", vmin=0, vmax=1)
            else:
                img = syn_img if col == 2 else syn_img  # col=0은 원본이 없을 수도 있음
                ax.imshow(img, cmap="gray", vmin=0, vmax=1)
                if col == 2:
                    # 마스크 경계 오버레이
                    ax.contour(syn_mask, levels=[0.5], colors="yellow", linewidths=1)

            ax.axis("off")
            if row == 0:
                ax.set_title(col_titles[col], fontsize=13, pad=6)

    fig.suptitle("Stable Diffusion Inpainting 합성 결과", fontsize=15, y=1.01)

    if save_path:
        plt.savefig(save_path, bbox_inches="tight", dpi=150)
        print(f"저장됨: {save_path}")
    else:
        plt.show()
    plt.close()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--synthetic_dir", default="data/synthetic")
    parser.add_argument("--n_samples", type=int, default=8)
    parser.add_argument("--save", default="results/inpainting_preview.png")
    args = parser.parse_args()

    os.makedirs(os.path.dirname(args.save), exist_ok=True)
    visualize_samples(args.synthetic_dir, args.n_samples, args.save)


if __name__ == "__main__":
    main()
