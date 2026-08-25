"""
Phase 3: PyTorch Dataset 클래스

두 가지 데이터셋을 지원:
  1. SyntheticTumorDataset  — Phase 2에서 생성한 [합성 MRI + GT 마스크] 쌍
  2. BraTSDataset           — 실제 BraTS 2D 슬라이스 (검증/평가용)
"""

import os
import json
import random
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import Dataset
import cv2
from PIL import Image


# ──────────────────────────────────────────────────────────────────────────────
# 공통 증강
# ──────────────────────────────────────────────────────────────────────────────

class Augmentor:
    """2D 의료 영상 증강 (image & mask에 동일 변환 적용)"""

    def __init__(self, p_flip=0.5, p_rotate=0.5, p_noise=0.3, p_contrast=0.3):
        self.p_flip = p_flip
        self.p_rotate = p_rotate
        self.p_noise = p_noise
        self.p_contrast = p_contrast

    def __call__(self, image: np.ndarray, mask: np.ndarray
                 ) -> tuple[np.ndarray, np.ndarray]:
        # 수평 뒤집기
        if random.random() < self.p_flip:
            image = np.fliplr(image).copy()
            mask = np.fliplr(mask).copy()

        # 랜덤 회전 (-30°~30°)
        if random.random() < self.p_rotate:
            angle = random.uniform(-30, 30)
            h, w = image.shape
            M = cv2.getRotationMatrix2D((w / 2, h / 2), angle, 1.0)
            image = cv2.warpAffine(image, M, (w, h), flags=cv2.INTER_LINEAR)
            mask  = cv2.warpAffine(mask.astype(np.float32), M, (w, h),
                                    flags=cv2.INTER_NEAREST).astype(np.uint8)

        # 가우시안 노이즈
        if random.random() < self.p_noise:
            noise = np.random.normal(0, 0.02, image.shape).astype(np.float32)
            image = np.clip(image + noise, 0, 1)

        # 대비 조절 (CLAHE 효과 시뮬레이션)
        if random.random() < self.p_contrast:
            alpha = random.uniform(0.8, 1.2)
            beta  = random.uniform(-0.05, 0.05)
            image = np.clip(alpha * image + beta, 0, 1)

        return image, mask


# ──────────────────────────────────────────────────────────────────────────────
# Dataset 1: 합성 데이터
# ──────────────────────────────────────────────────────────────────────────────

class SyntheticTumorDataset(Dataset):
    """
    Phase 2에서 생성된 합성 뇌종양 MRI 데이터셋.

    디렉토리 구조:
        synthetic_dir/
            images/   *.png  (합성 MRI)
            masks/    *.png  (GT 이진 마스크)
            metadata.json
    """

    def __init__(self,
                 synthetic_dir: str,
                 target_size: tuple = (256, 256),
                 augment: bool = True,
                 split: str = "train",
                 val_ratio: float = 0.15,
                 test_ratio: float = 0.10,
                 seed: int = 42):

        self.target_size = target_size
        self.augment = augment
        self.aug = Augmentor() if augment else None

        meta_path = os.path.join(synthetic_dir, "metadata.json")
        if not os.path.exists(meta_path):
            raise FileNotFoundError(f"metadata.json 없음: {meta_path}")

        with open(meta_path) as f:
            meta = json.load(f)

        # samples 키가 있으면 사용, 없으면 디렉토리에서 직접 탐색
        if "samples" in meta:
            all_samples = [(
                os.path.join(synthetic_dir, s["image"]),
                os.path.join(synthetic_dir, s["mask"]),
            ) for s in meta["samples"]]
        else:
            # fallback: *_synthetic.png 파일 직접 탐색
            import glob
            syn_imgs = sorted(glob.glob(os.path.join(synthetic_dir, '*_synthetic.png')))
            gt_dir   = os.path.join(synthetic_dir, 'gt_masks')
            all_samples = []
            for img_path in syn_imgs:
                stem      = os.path.basename(img_path).replace('_synthetic.png', '')
                mask_path = os.path.join(gt_dir, f'{stem}_gt_mask.png')
                if os.path.exists(mask_path):
                    all_samples.append((img_path, mask_path))

        # 재현 가능한 split
        rng = random.Random(seed)
        rng.shuffle(all_samples)
        n = len(all_samples)
        n_test = max(1, int(n * test_ratio))
        n_val  = max(1, int(n * val_ratio))

        if split == "test":
            self.samples = all_samples[:n_test]
        elif split == "val":
            self.samples = all_samples[n_test:n_test + n_val]
        else:
            self.samples = all_samples[n_test + n_val:]

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int) -> dict:
        img_path, mask_path = self.samples[idx]

        image = np.array(Image.open(img_path).convert("L"), dtype=np.float32) / 255.0
        mask  = np.array(Image.open(mask_path).convert("L"))
        mask  = (mask > 127).astype(np.uint8)

        image = cv2.resize(image, self.target_size[::-1], interpolation=cv2.INTER_LINEAR)
        mask  = cv2.resize(mask,  self.target_size[::-1], interpolation=cv2.INTER_NEAREST)

        if self.aug is not None and self.augment:
            image, mask = self.aug(image, mask)

        image_t = torch.from_numpy(image).unsqueeze(0)    # (1, H, W)
        mask_t  = torch.from_numpy(mask.astype(np.float32)).unsqueeze(0)  # (1, H, W)

        return {"image": image_t, "mask": mask_t, "path": img_path}


# ──────────────────────────────────────────────────────────────────────────────
# Dataset 2: BraTS 실제 데이터 (평가용)
# ──────────────────────────────────────────────────────────────────────────────

class BraTSDataset(Dataset):
    """
    BraTS 2020/2021 데이터셋 2D 슬라이스 버전.

    BraTS 원본은 4채널 3D NIfTI이므로, 먼저 아래 스크립트로 전처리 필요:
        phase4_evaluation/prepare_brats_slices.py

    전처리 후 디렉토리 구조:
        brats_dir/
            images/   *_t1ce.png  (T1 contrast enhanced 슬라이스)
            masks/    *_seg.png   (라벨: 0=배경, 1=괴사, 2=부종, 4=강화종양)
    """

    def __init__(self,
                 brats_dir: str,
                 target_size: tuple = (256, 256),
                 modality: str = "t1ce",
                 augment: bool = False,
                 binary_mask: bool = True):
        """
        binary_mask=True: 모든 종양 라벨을 1로 합침 (전체 종양 분할)
        binary_mask=False: 다중 클래스 (괴사/부종/강화)
        """
        self.target_size = target_size
        self.binary_mask = binary_mask
        self.aug = Augmentor() if augment else None

        img_dir  = os.path.join(brats_dir, "images")
        mask_dir = os.path.join(brats_dir, "masks")

        img_files = sorted(Path(img_dir).glob(f"*_{modality}.png"))
        self.samples = []
        for img_p in img_files:
            stem = img_p.stem.replace(f"_{modality}", "")
            mask_p = os.path.join(mask_dir, f"{stem}_seg.png")
            if os.path.exists(mask_p):
                self.samples.append((str(img_p), mask_p))

        if not self.samples:
            raise FileNotFoundError(
                f"BraTS 슬라이스를 찾을 수 없습니다: {brats_dir}\n"
                "먼저 phase4_evaluation/prepare_brats_slices.py 를 실행하세요."
            )

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int) -> dict:
        img_path, mask_path = self.samples[idx]

        image = np.array(Image.open(img_path).convert("L"), dtype=np.float32) / 255.0
        mask  = np.array(Image.open(mask_path).convert("L"))

        if self.binary_mask:
            mask = (mask > 0).astype(np.uint8)
        else:
            # BraTS 라벨: 1=괴사핵, 2=부종, 4=강화종양 → 0/1/2/3으로 재매핑
            remap = {0: 0, 1: 1, 2: 2, 4: 3}
            mask_new = np.zeros_like(mask)
            for orig, new in remap.items():
                mask_new[mask == orig] = new
            mask = mask_new

        image = cv2.resize(image, self.target_size[::-1], interpolation=cv2.INTER_LINEAR)
        mask  = cv2.resize(mask,  self.target_size[::-1], interpolation=cv2.INTER_NEAREST)

        if self.aug is not None:
            image, mask = self.aug(image, mask)

        image_t = torch.from_numpy(image).unsqueeze(0)
        mask_t  = torch.from_numpy(mask.astype(np.float32)).unsqueeze(0)

        return {"image": image_t, "mask": mask_t, "path": img_path}


# ──────────────────────────────────────────────────────────────────────────────
# 혼합 데이터셋 (합성 + 실제 데이터 합치기)
# ──────────────────────────────────────────────────────────────────────────────

class MixedDataset(Dataset):
    """합성 데이터 + 실제 BraTS 데이터 혼합 (비율 조절 가능)"""

    def __init__(self, synthetic_ds: Dataset, real_ds: Dataset,
                 real_ratio: float = 0.3):
        self.synthetic = synthetic_ds
        self.real = real_ds
        self.real_ratio = real_ratio
        n_real = int(len(synthetic_ds) * real_ratio / (1 - real_ratio))
        self.real_indices = list(range(len(real_ds))) * (n_real // len(real_ds) + 1)
        random.shuffle(self.real_indices)
        self.real_indices = self.real_indices[:n_real]

    def __len__(self) -> int:
        return len(self.synthetic) + len(self.real_indices)

    def __getitem__(self, idx: int) -> dict:
        if idx < len(self.synthetic):
            return self.synthetic[idx]
        else:
            real_idx = self.real_indices[idx - len(self.synthetic)]
            return self.real[real_idx]
