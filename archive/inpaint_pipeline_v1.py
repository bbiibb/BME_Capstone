"""
Phase 2: Stable Diffusion Inpainting 파이프라인

정상 뇌 슬라이스 + 종양 마스크 → Stable Diffusion Inpainting → 합성 뇌종양 MRI

핵심 아이디어:
  - 마스크 영역 안에 GBM 종양 특유의 질감(괴사핵, 강화 림, 부종)을 생성
  - Hugging Face diffusers의 StableDiffusionInpaintPipeline 사용
  - LoRA 가중치(finetune_lora.py 출력)를 로드하면 BraTS 질감이 반영됨
  - 생성된 이미지에 후처리를 통해 MRI다운 특성 유지

사용법 (사전학습 모델만):
    python phase2_inpainting/inpaint_pipeline.py \
        --slice_dir data/processed/normal_slices \
        --mask_dir  data/processed/tumor_masks \
        --output_dir data/synthetic \
        --config configs/config.yaml

사용법 (LoRA 적용 — BraTS 질감 학습 후):
    python phase2_inpainting/inpaint_pipeline.py \
        --lora_weights checkpoints/lora_gbm/lora_final \
        --lora_scale   0.9 \
        --output_dir   data/synthetic_lora \
        --config configs/config.yaml
"""

import os
import sys
import argparse
import logging
from pathlib import Path

import numpy as np
import yaml
import torch
import cv2
from PIL import Image
from tqdm import tqdm

sys.path.insert(0, str(Path(__file__).parent.parent))
from utils.io_utils import load_png_as_float, load_mask_png, load_json, save_json

logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)s | %(message)s")
logger = logging.getLogger(__name__)


def load_sd_inpainting_pipeline(model_id: str, device: str,
                                  lora_weights: str = None,
                                  lora_scale: float = 0.9):
    """
    Stable Diffusion Inpainting 파이프라인 로드.

    Args:
        lora_weights: finetune_lora.py 출력 폴더 경로.
                      지정 시 BraTS GBM 질감 LoRA를 UNet에 적용.
        lora_scale:   LoRA 적용 강도 (0.0=미적용, 1.0=완전 적용, 0.9 권장)
    """
    from diffusers import StableDiffusionInpaintPipeline

    logger.info(f"모델 로드 중: {model_id}")
    dtype = torch.float16 if device == "cuda" else torch.float32

    pipe = StableDiffusionInpaintPipeline.from_pretrained(
        model_id,
        torch_dtype=dtype,
        safety_checker=None,
        requires_safety_checker=False,
    )

    # ── LoRA 가중치 적용 ────────────────────────────────────────────────────
    if lora_weights and os.path.exists(lora_weights):
        try:
            from peft import PeftModel
            pipe.unet = PeftModel.from_pretrained(pipe.unet, lora_weights)
            # LoRA 스케일 조정: 1.0이면 완전 적용, 낮추면 사전학습 특성 유지
            for name, module in pipe.unet.named_modules():
                if hasattr(module, "scaling"):
                    module.scaling = {k: lora_scale for k in module.scaling}
            logger.info(f"LoRA 가중치 적용 완료: {lora_weights} (scale={lora_scale})")
        except ImportError:
            logger.warning("peft 미설치 — LoRA 미적용. pip install peft")
        except Exception as e:
            logger.warning(f"LoRA 로드 실패: {e} — 사전학습 모델로 진행")
    elif lora_weights:
        logger.warning(f"LoRA 경로 없음: {lora_weights} — 사전학습 모델로 진행")
        logger.warning("finetune_lora.py 를 먼저 실행하세요.")
    else:
        logger.info("LoRA 미적용 — 사전학습 SD로 진행")

    pipe = pipe.to(device)
    pipe.enable_attention_slicing()

    if device == "cuda":
        try:
            pipe.enable_xformers_memory_efficient_attention()
            logger.info("xformers 메모리 최적화 활성화")
        except Exception:
            logger.info("xformers 미설치 — 기본 attention 사용")

    return pipe


def float_to_pil_rgb(arr: np.ndarray) -> Image.Image:
    """[0,1] float 배열 → PIL RGB 이미지 (SD 입력 형식)"""
    arr_uint8 = (np.clip(arr, 0, 1) * 255).astype(np.uint8)
    # 그레이스케일 → RGB (SD는 RGB 3채널 요구)
    rgb = np.stack([arr_uint8] * 3, axis=-1)
    return Image.fromarray(rgb)


def mask_to_pil(mask: np.ndarray) -> Image.Image:
    """이진 마스크 → PIL 이미지 (흰색=인페인팅 영역)"""
    mask_uint8 = (mask * 255).astype(np.uint8)
    # 마스크 팽창: 경계 부드럽게 처리
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
    mask_dilated = cv2.dilate(mask_uint8, kernel, iterations=2)
    return Image.fromarray(mask_dilated).convert("L")


def postprocess_inpainted(
    inpainted_rgb: Image.Image,
    original_gray: np.ndarray,
    mask: np.ndarray,
    blend_sigma: int = 5
) -> np.ndarray:
    """
    인페인팅 결과 후처리:
    1. RGB → 그레이스케일 변환
    2. 마스크 경계 Gaussian Blending (자연스러운 융합)
    3. 마스크 밖은 원본 유지
    """
    # RGB → 그레이
    inpainted_arr = np.array(inpainted_rgb.convert("L"), dtype=np.float32) / 255.0

    # Gaussian soft mask for blending
    mask_f = mask.astype(np.float32)
    soft_mask = cv2.GaussianBlur(mask_f, (2 * blend_sigma + 1, 2 * blend_sigma + 1), blend_sigma)
    soft_mask = np.clip(soft_mask, 0, 1)

    # Alpha blending
    blended = soft_mask * inpainted_arr + (1 - soft_mask) * original_gray
    return blended.astype(np.float32)


def inpaint_single(pipe,
                    slice_path: str,
                    mask_path: str,
                    cfg_inp: dict,
                    output_path: str) -> bool:
    """슬라이스 + 마스크 쌍을 인페인팅하고 결과 저장. 성공 여부 반환."""
    try:
        original_gray = load_png_as_float(slice_path)
        mask = load_mask_png(mask_path)

        h, w = original_gray.shape
        size = cfg_inp.get("image_size", 512)

        # PIL 변환 & 리사이즈 (SD는 512x512 또는 768x768)
        image_pil = float_to_pil_rgb(
            cv2.resize(original_gray, (size, size), interpolation=cv2.INTER_LINEAR)
        )
        mask_resized = cv2.resize(mask, (size, size), interpolation=cv2.INTER_NEAREST)
        mask_pil = mask_to_pil(mask_resized)

        # Inpainting 실행
        result = pipe(
            prompt=cfg_inp["prompt"],
            negative_prompt=cfg_inp["negative_prompt"],
            image=image_pil,
            mask_image=mask_pil,
            num_inference_steps=cfg_inp["num_inference_steps"],
            guidance_scale=cfg_inp["guidance_scale"],
            strength=cfg_inp["strength"],
            height=size,
            width=size,
        ).images[0]

        # 원본 해상도로 복원 + 후처리
        result_resized = result.resize((w, h), Image.LANCZOS)
        final = postprocess_inpainted(result_resized, original_gray, mask)

        # 저장
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        final_uint8 = (np.clip(final, 0, 1) * 255).astype(np.uint8)
        Image.fromarray(final_uint8).save(output_path)
        return True

    except Exception as e:
        logger.warning(f"인페인팅 실패 [{slice_path}]: {e}")
        return False


def build_pair_list(slice_dir: str, mask_dir: str, mapping: dict) -> list[tuple]:
    """(slice_path, mask_path, stem) 튜플 리스트 생성"""
    pairs = []
    for slice_name, mask_names in mapping.items():
        slice_path = os.path.join(slice_dir, slice_name)
        if not os.path.exists(slice_path):
            continue
        for mask_name in mask_names:
            mask_path = os.path.join(mask_dir, mask_name)
            if not os.path.exists(mask_path):
                continue
            stem = Path(mask_name).stem  # 출력 파일명 기반
            pairs.append((slice_path, mask_path, stem))
    return pairs


def main():
    parser = argparse.ArgumentParser(description="Stable Diffusion Inpainting 파이프라인")
    parser.add_argument("--slice_dir",     default="data/processed/normal_slices")
    parser.add_argument("--mask_dir",      default="data/processed/tumor_masks")
    parser.add_argument("--output_dir",    default="data/synthetic")
    parser.add_argument("--config",        default="configs/config.yaml")
    parser.add_argument("--max_images",    type=int, default=None,
                        help="생성할 최대 이미지 수 (미지정 시 전체)")
    parser.add_argument("--lora_weights",  default=None,
                        help="LoRA 가중치 폴더 (finetune_lora.py 출력). "
                             "미지정 시 사전학습 SD만 사용.")
    parser.add_argument("--lora_scale",    type=float, default=0.9,
                        help="LoRA 적용 강도 (0.0~1.0, 기본 0.9)")
    args = parser.parse_args()

    with open(args.config) as f:
        cfg = yaml.safe_load(f)

    cfg_inp = cfg["inpainting"]
    cfg_inp.setdefault("image_size", 512)

    device = cfg_inp["device"]
    if device == "cuda" and not torch.cuda.is_available():
        logger.warning("CUDA 불가 — MPS 또는 CPU로 전환")
        device = "mps" if torch.backends.mps.is_available() else "cpu"
        cfg_inp["device"] = device
    logger.info(f"사용 디바이스: {device}")

    # 슬라이스-마스크 매핑 로드
    mapping_path = os.path.join(args.mask_dir, "slice_mask_mapping.json")
    if not os.path.exists(mapping_path):
        logger.error(f"매핑 파일 없음: {mapping_path}")
        logger.info("먼저 phase1_preprocessing/generate_tumor_masks.py를 실행하세요.")
        return

    mapping = load_json(mapping_path)
    pairs = build_pair_list(args.slice_dir, args.mask_dir, mapping)

    if not pairs:
        logger.error("처리할 이미지-마스크 쌍이 없습니다.")
        return

    if args.max_images:
        pairs = pairs[:args.max_images]

    logger.info(f"처리할 쌍: {len(pairs)}개")

    # SD 파이프라인 로드 (LoRA 가중치 있으면 자동 적용)
    pipe = load_sd_inpainting_pipeline(
        cfg_inp["model_id"], device,
        lora_weights=args.lora_weights,
        lora_scale=args.lora_scale,
    )

    # 출력 폴더 구조
    img_out = os.path.join(args.output_dir, "images")
    mask_out = os.path.join(args.output_dir, "masks")
    os.makedirs(img_out, exist_ok=True)
    os.makedirs(mask_out, exist_ok=True)

    metadata = []
    success = 0

    for slice_path, mask_path, stem in tqdm(pairs, desc="Inpainting 진행 중"):
        out_img = os.path.join(img_out, f"{stem}_syn.png")
        out_mask = os.path.join(mask_out, f"{stem}_seg.png")

        if inpaint_single(pipe, slice_path, mask_path, cfg_inp, out_img):
            # 마스크도 함께 복사 (분할 학습용 GT)
            import shutil
            shutil.copy2(mask_path, out_mask)
            metadata.append({
                "image": f"images/{stem}_syn.png",
                "mask":  f"masks/{stem}_seg.png",
                "source_slice": os.path.basename(slice_path),
            })
            success += 1

    # 메타데이터 저장
    meta_path = os.path.join(args.output_dir, "metadata.json")
    save_json({"total": success, "samples": metadata}, meta_path)

    logger.info(f"인페인팅 완료: {success}/{len(pairs)}장 성공 → {args.output_dir}")


if __name__ == "__main__":
    main()
