"""
Phase 3: U-Net & Attention U-Net 아키텍처

두 모델을 구현합니다:
  1. UNet         — 표준 U-Net (Ronneberger et al., 2015)
  2. AttentionUNet — Attention Gate를 추가한 확장 버전 (Oktay et al., 2018)

입력:  (B, 1, H, W) — 그레이스케일 MRI 슬라이스
출력: (B, 1, H, W) — 종양 확률 맵 (sigmoid 전 logit)
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


# ──────────────────────────────────────────────────────────────────────────────
# 공통 빌딩 블록
# ──────────────────────────────────────────────────────────────────────────────

class ConvBlock(nn.Module):
    """Conv → BN → ReLU × 2 (U-Net 기본 블록)"""

    def __init__(self, in_ch: int, out_ch: int, dropout: float = 0.0):
        super().__init__()
        layers = [
            nn.Conv2d(in_ch, out_ch, 3, padding=1, bias=False),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_ch, out_ch, 3, padding=1, bias=False),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
        ]
        if dropout > 0:
            layers.append(nn.Dropout2d(dropout))
        self.block = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.block(x)


class DownSample(nn.Module):
    """MaxPool → ConvBlock"""

    def __init__(self, in_ch: int, out_ch: int, dropout: float = 0.0):
        super().__init__()
        self.pool = nn.MaxPool2d(2)
        self.conv = ConvBlock(in_ch, out_ch, dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.conv(self.pool(x))


class UpSample(nn.Module):
    """Bilinear Upsample → Concat skip → ConvBlock"""

    def __init__(self, in_ch: int, skip_ch: int, out_ch: int, dropout: float = 0.0):
        super().__init__()
        self.up   = nn.Upsample(scale_factor=2, mode="bilinear", align_corners=True)
        self.conv = ConvBlock(in_ch + skip_ch, out_ch, dropout)

    def forward(self, x: torch.Tensor, skip: torch.Tensor) -> torch.Tensor:
        x = self.up(x)
        # 해상도가 맞지 않을 경우 패딩
        if x.shape != skip.shape:
            x = F.pad(x, [0, skip.shape[-1] - x.shape[-1],
                          0, skip.shape[-2] - x.shape[-2]])
        return self.conv(torch.cat([x, skip], dim=1))


# ──────────────────────────────────────────────────────────────────────────────
# Attention Gate (Attention U-Net용)
# ──────────────────────────────────────────────────────────────────────────────

class AttentionGate(nn.Module):
    """
    Soft Attention Gate.
    g: 디코더 feature, x: 인코더 skip feature
    출력: x에 attention weight를 곱한 결과
    """

    def __init__(self, g_ch: int, x_ch: int, inter_ch: int):
        super().__init__()
        self.W_g = nn.Sequential(
            nn.Conv2d(g_ch, inter_ch, 1, bias=False),
            nn.BatchNorm2d(inter_ch),
        )
        self.W_x = nn.Sequential(
            nn.Conv2d(x_ch, inter_ch, 1, bias=False),
            nn.BatchNorm2d(inter_ch),
        )
        self.psi = nn.Sequential(
            nn.Conv2d(inter_ch, 1, 1, bias=False),
            nn.BatchNorm2d(1),
            nn.Sigmoid(),
        )
        self.relu = nn.ReLU(inplace=True)

    def forward(self, g: torch.Tensor, x: torch.Tensor) -> torch.Tensor:
        g1 = self.W_g(g)
        x1 = self.W_x(x)
        # 해상도 맞추기
        if g1.shape[-2:] != x1.shape[-2:]:
            g1 = F.interpolate(g1, size=x1.shape[-2:], mode="bilinear", align_corners=True)
        psi = self.relu(g1 + x1)
        psi = self.psi(psi)
        return x * psi


# ──────────────────────────────────────────────────────────────────────────────
# Model 1: 표준 U-Net
# ──────────────────────────────────────────────────────────────────────────────

class UNet(nn.Module):
    """
    Standard U-Net for brain tumor segmentation.
    features: [64, 128, 256, 512] — 각 레벨의 채널 수
    """

    def __init__(self,
                 in_channels: int = 1,
                 out_channels: int = 1,
                 features: list[int] = None,
                 dropout: float = 0.2):
        super().__init__()
        if features is None:
            features = [64, 128, 256, 512]

        # Encoder
        self.enc1 = ConvBlock(in_channels, features[0])
        self.enc2 = DownSample(features[0], features[1], dropout)
        self.enc3 = DownSample(features[1], features[2], dropout)
        self.enc4 = DownSample(features[2], features[3], dropout)

        # Bottleneck
        self.bottleneck = DownSample(features[3], features[3] * 2, dropout)

        # Decoder
        self.dec4 = UpSample(features[3] * 2, features[3], features[3], dropout)
        self.dec3 = UpSample(features[3], features[2], features[2], dropout)
        self.dec2 = UpSample(features[2], features[1], features[1], dropout)
        self.dec1 = UpSample(features[1], features[0], features[0], dropout)

        # Output
        self.head = nn.Conv2d(features[0], out_channels, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        s1 = self.enc1(x)
        s2 = self.enc2(s1)
        s3 = self.enc3(s2)
        s4 = self.enc4(s3)
        b  = self.bottleneck(s4)
        d4 = self.dec4(b,  s4)
        d3 = self.dec3(d4, s3)
        d2 = self.dec2(d3, s2)
        d1 = self.dec1(d2, s1)
        return self.head(d1)


# ──────────────────────────────────────────────────────────────────────────────
# Model 2: Attention U-Net
# ──────────────────────────────────────────────────────────────────────────────

class AttentionUNet(nn.Module):
    """
    Attention U-Net — skip connection에 Attention Gate 추가.
    종양처럼 작고 국소적인 병변 분할에 유리.
    """

    def __init__(self,
                 in_channels: int = 1,
                 out_channels: int = 1,
                 features: list[int] = None,
                 dropout: float = 0.2):
        super().__init__()
        if features is None:
            features = [64, 128, 256, 512]

        # Encoder
        self.enc1 = ConvBlock(in_channels, features[0])
        self.enc2 = DownSample(features[0], features[1], dropout)
        self.enc3 = DownSample(features[1], features[2], dropout)
        self.enc4 = DownSample(features[2], features[3], dropout)

        # Bottleneck
        self.bottleneck = DownSample(features[3], features[3] * 2, dropout)

        # Attention Gates
        self.att4 = AttentionGate(features[3] * 2, features[3], features[3] // 2)
        self.att3 = AttentionGate(features[3],      features[2], features[2] // 2)
        self.att2 = AttentionGate(features[2],      features[1], features[1] // 2)
        self.att1 = AttentionGate(features[1],      features[0], features[0] // 2)

        # Decoder
        self.dec4 = UpSample(features[3] * 2, features[3], features[3], dropout)
        self.dec3 = UpSample(features[3],      features[2], features[2], dropout)
        self.dec2 = UpSample(features[2],      features[1], features[1], dropout)
        self.dec1 = UpSample(features[1],      features[0], features[0], dropout)

        # Output
        self.head = nn.Conv2d(features[0], out_channels, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        s1 = self.enc1(x)
        s2 = self.enc2(s1)
        s3 = self.enc3(s2)
        s4 = self.enc4(s3)
        b  = self.bottleneck(s4)

        # Attention gate: g=디코더 feature, x=인코더 skip
        s4_att = self.att4(b,  s4)
        d4 = self.dec4(b, s4_att)

        s3_att = self.att3(d4, s3)
        d3 = self.dec3(d4, s3_att)

        s2_att = self.att2(d3, s2)
        d2 = self.dec2(d3, s2_att)

        s1_att = self.att1(d2, s1)
        d1 = self.dec1(d2, s1_att)

        return self.head(d1)


# ──────────────────────────────────────────────────────────────────────────────
# 모델 팩토리
# ──────────────────────────────────────────────────────────────────────────────

def build_model(cfg: dict) -> nn.Module:
    seg_cfg = cfg["segmentation"]
    model_type = seg_cfg.get("model", "attention_unet").lower()

    kwargs = dict(
        in_channels  = seg_cfg["in_channels"],
        out_channels = seg_cfg["out_channels"],
        features     = seg_cfg["features"],
        dropout      = seg_cfg["dropout"],
    )

    if model_type == "unet":
        model = UNet(**kwargs)
    elif model_type == "attention_unet":
        model = AttentionUNet(**kwargs)
    else:
        raise ValueError(f"Unknown model type: {model_type}")

    total_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"모델: {model_type} | 파라미터: {total_params:,}")
    return model


if __name__ == "__main__":
    # 빠른 동작 확인
    model = AttentionUNet(in_channels=1, out_channels=1, features=[32, 64, 128, 256])
    x = torch.randn(2, 1, 256, 256)
    out = model(x)
    print(f"입력: {x.shape} → 출력: {out.shape}")  # (2, 1, 256, 256) 기대
