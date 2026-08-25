"""
분할 평가 지표: Dice, IoU, Precision, Recall, Hausdorff Distance
"""
import numpy as np
import torch
from scipy.ndimage import distance_transform_edt


def dice_coefficient(pred: np.ndarray, target: np.ndarray, smooth: float = 1e-6) -> float:
    """
    Dice = 2 * |A ∩ B| / (|A| + |B|)
    pred, target: binary numpy arrays (0 or 1)
    """
    pred = pred.astype(bool).flatten()
    target = target.astype(bool).flatten()
    intersection = (pred & target).sum()
    return (2.0 * intersection + smooth) / (pred.sum() + target.sum() + smooth)


def iou_score(pred: np.ndarray, target: np.ndarray, smooth: float = 1e-6) -> float:
    """IoU = |A ∩ B| / |A ∪ B|"""
    pred = pred.astype(bool).flatten()
    target = target.astype(bool).flatten()
    intersection = (pred & target).sum()
    union = (pred | target).sum()
    return (intersection + smooth) / (union + smooth)


def precision_recall(pred: np.ndarray, target: np.ndarray) -> tuple[float, float]:
    """Precision, Recall 계산"""
    pred = pred.astype(bool).flatten()
    target = target.astype(bool).flatten()
    tp = (pred & target).sum()
    fp = (pred & ~target).sum()
    fn = (~pred & target).sum()
    precision = tp / (tp + fp + 1e-8)
    recall = tp / (tp + fn + 1e-8)
    return float(precision), float(recall)


def hausdorff_distance(pred: np.ndarray, target: np.ndarray) -> float:
    """
    Hausdorff Distance (95th percentile) — 두 경계 간 최대 거리
    pred, target: 2D binary arrays
    """
    pred_b = pred.astype(bool)
    target_b = target.astype(bool)
    if not pred_b.any() or not target_b.any():
        return float("inf")

    dist_pred = distance_transform_edt(~pred_b)
    dist_target = distance_transform_edt(~target_b)

    # 95th percentile Hausdorff
    hd_pred = dist_target[pred_b]
    hd_target = dist_pred[target_b]
    return float(max(np.percentile(hd_pred, 95), np.percentile(hd_target, 95)))


def compute_all_metrics(pred: np.ndarray, target: np.ndarray) -> dict:
    """모든 평가 지표를 한번에 계산"""
    precision, recall = precision_recall(pred, target)
    f1 = 2 * precision * recall / (precision + recall + 1e-8)
    return {
        "dice":       dice_coefficient(pred, target),
        "iou":        iou_score(pred, target),
        "precision":  precision,
        "recall":     recall,
        "f1":         f1,
        "hausdorff":  hausdorff_distance(pred, target),
    }


# ── PyTorch loss 함수 ──────────────────────────────────────────────────────────

class DiceLoss(torch.nn.Module):
    def __init__(self, smooth: float = 1e-6):
        super().__init__()
        self.smooth = smooth

    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        probs = torch.sigmoid(logits)
        probs_f = probs.view(probs.size(0), -1)
        targets_f = targets.view(targets.size(0), -1).float()
        intersection = (probs_f * targets_f).sum(dim=1)
        dice = (2.0 * intersection + self.smooth) / (
            probs_f.sum(dim=1) + targets_f.sum(dim=1) + self.smooth
        )
        return 1.0 - dice.mean()


class DiceBCELoss(torch.nn.Module):
    def __init__(self, dice_weight: float = 0.6, bce_weight: float = 0.4):
        super().__init__()
        self.dice = DiceLoss()
        self.bce = torch.nn.BCEWithLogitsLoss()
        self.dice_w = dice_weight
        self.bce_w = bce_weight

    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        return self.dice_w * self.dice(logits, targets) + \
               self.bce_w * self.bce(logits, targets.float())
