"""
공통 I/O 유틸리티: NIfTI 로드, 슬라이스 저장, 마스크 저장 등
"""
import os
import json
import numpy as np
import nibabel as nib
from pathlib import Path
from PIL import Image
import cv2


def load_nifti(path: str) -> tuple[np.ndarray, np.ndarray]:
    """NIfTI(.nii / .nii.gz) 파일 로드 → (volume, affine)"""
    img = nib.load(path)
    volume = img.get_fdata(dtype=np.float32)
    return volume, img.affine


def save_nifti(volume: np.ndarray, affine: np.ndarray, path: str) -> None:
    os.makedirs(os.path.dirname(path), exist_ok=True)
    nib.save(nib.Nifti1Image(volume, affine), path)


def save_slice_png(arr: np.ndarray, path: str) -> None:
    """float 배열 → 0~255 uint8 PNG 저장"""
    os.makedirs(os.path.dirname(path), exist_ok=True)
    arr_uint8 = (np.clip(arr, 0, 1) * 255).astype(np.uint8)
    Image.fromarray(arr_uint8).save(path)


def save_mask_png(mask: np.ndarray, path: str) -> None:
    """이진 마스크(bool/0-1) → PNG 저장"""
    os.makedirs(os.path.dirname(path), exist_ok=True)
    mask_uint8 = (mask.astype(np.uint8) * 255)
    Image.fromarray(mask_uint8).save(path)


def load_png_as_float(path: str) -> np.ndarray:
    """PNG → [0,1] float32 배열"""
    img = np.array(Image.open(path).convert("L"), dtype=np.float32)
    return img / 255.0


def load_mask_png(path: str) -> np.ndarray:
    """PNG 마스크 → bool 배열"""
    img = np.array(Image.open(path).convert("L"))
    return (img > 127).astype(np.uint8)


def save_json(data: dict, path: str) -> None:
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2, ensure_ascii=False)


def load_json(path: str) -> dict:
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def normalize_volume(volume: np.ndarray, method: str = "z_score") -> np.ndarray:
    """뇌 볼륨 강도 정규화"""
    brain_mask = volume > 0
    if brain_mask.sum() == 0:
        return volume
    if method == "z_score":
        mean = volume[brain_mask].mean()
        std = volume[brain_mask].std() + 1e-8
        norm = (volume - mean) / std
        # sigmoid-like squeeze to [0,1]
        norm = 1.0 / (1.0 + np.exp(-norm / 2.0))
    elif method == "min_max":
        vmin = volume[brain_mask].min()
        vmax = volume[brain_mask].max() + 1e-8
        norm = (volume - vmin) / (vmax - vmin)
        norm = np.clip(norm, 0, 1)
    else:
        raise ValueError(f"Unknown normalization method: {method}")
    norm[~brain_mask] = 0.0
    return norm.astype(np.float32)


def resize_slice(slice_2d: np.ndarray, target_size: tuple) -> np.ndarray:
    """이미지 슬라이스를 target_size로 리사이즈"""
    return cv2.resize(slice_2d, (target_size[1], target_size[0]),
                      interpolation=cv2.INTER_LINEAR)
