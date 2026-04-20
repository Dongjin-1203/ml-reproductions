"""
model.py
--------
CIFAR-10 분류를 위한 ViT 및 CNN baseline 모델 정의.

notebooks/model.ipynb 의 구현을 그대로 옮긴 파일이다.
노트북 코드를 수정할 경우 이 파일도 함께 동기화할 것.

클래스 목록:
  - PatchEmbedding          : 이미지를 패치 시퀀스로 변환
  - MultiHeadSelfAttention  : Scaled Dot-Product Attention
  - TransformerEncoderBlock : Pre-LN Transformer 블록
  - ViT                     : Vision Transformer 전체 모델
  - SimpleCNN               : CNN baseline (4-block ConvNet)
"""

from __future__ import annotations

import torch
import torch.nn as nn


# ─────────────────────────────────────────────────────────────────────────────
# Patch Embedding
# 논문 Section 3.1 — Equation (1)
# ─────────────────────────────────────────────────────────────────────────────

class PatchEmbedding(nn.Module):
    """이미지를 N개의 패치로 분할하고 D차원으로 선형 투영한다.

    Conv2d(kernel=patch_size, stride=patch_size)로 분할과 투영을 동시에 수행한다.

    Args:
        patch_size:  패치 한 변의 크기 P.
        in_channels: 입력 이미지 채널 수 C.
        embed_dim:   출력 임베딩 차원 D.
    """

    def __init__(
        self,
        patch_size: int = 16,
        in_channels: int = 3,
        embed_dim: int = 768,
    ) -> None:
        super().__init__()
        self.proj = nn.Conv2d(
            in_channels=in_channels,
            out_channels=embed_dim,
            kernel_size=patch_size,
            stride=patch_size,
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: (B, C, H, W)
        Returns:
            (B, N, D)  where N = (H/P) * (W/P)
        """
        x = self.proj(x)       # (B, D, H/P, W/P)
        x = x.flatten(2)       # (B, D, N)
        x = x.transpose(1, 2)  # (B, N, D)
        return x


# ─────────────────────────────────────────────────────────────────────────────
# Multi-Head Self-Attention
# 논문 Section 3.1 — Equation (2)
# ─────────────────────────────────────────────────────────────────────────────

class MultiHeadSelfAttention(nn.Module):
    """Scaled Dot-Product Multi-Head Self-Attention.

    Args:
        embed_dim: 입력/출력 차원 D.
        num_heads: Attention 헤드 수 k. embed_dim은 num_heads로 나누어 떨어져야 한다.
        dropout:   Attention weight에 적용하는 dropout 비율.
    """

    def __init__(
        self,
        embed_dim: int = 256,
        num_heads: int = 8,
        dropout: float = 0.1,
    ) -> None:
        super().__init__()
        assert embed_dim % num_heads == 0, "embed_dim must be divisible by num_heads"
        self.num_heads = num_heads
        self.embed_dim = embed_dim
        self.d_k = embed_dim // num_heads

        self.W_q = nn.Linear(embed_dim, embed_dim)
        self.W_k = nn.Linear(embed_dim, embed_dim)
        self.W_v = nn.Linear(embed_dim, embed_dim)
        self.W_o = nn.Linear(embed_dim, embed_dim)
        self.dropout = nn.Dropout(dropout)

    def _split_heads(self, x: torch.Tensor) -> torch.Tensor:
        """(B, seq, D) → (B, heads, seq, d_k)"""
        B, seq_len, _ = x.shape
        x = x.reshape(B, seq_len, self.num_heads, self.d_k)
        return x.transpose(1, 2)

    def _scaled_dot_product(
        self,
        q: torch.Tensor,
        k: torch.Tensor,
        v: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        scores = torch.matmul(q, k.transpose(-2, -1)) / (self.d_k ** 0.5)
        attn_weights = torch.softmax(scores, dim=-1)
        attn_weights = self.dropout(attn_weights)
        attn_output = torch.matmul(attn_weights, v)
        return attn_output, attn_weights

    def forward(
        self, x: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Args:
            x: (B, seq, D)
        Returns:
            attn_out:     (B, seq, D)
            attn_weights: (B, heads, seq, seq)
        """
        B, seq_len, _ = x.shape

        q = self._split_heads(self.W_q(x))  # (B, heads, seq, d_k)
        k = self._split_heads(self.W_k(x))
        v = self._split_heads(self.W_v(x))

        attn_out, attn_weights = self._scaled_dot_product(q, k, v)

        attn_out = attn_out.transpose(1, 2).reshape(B, seq_len, self.embed_dim)
        attn_out = self.W_o(attn_out)
        return attn_out, attn_weights


# ─────────────────────────────────────────────────────────────────────────────
# Transformer Encoder Block
# 논문 Section 3.1 — Equations (2)(3), Pre-LayerNorm 구조
# ─────────────────────────────────────────────────────────────────────────────

class TransformerEncoderBlock(nn.Module):
    """Pre-LN Transformer Encoder 블록.

    z'_l = MSA(LN(z_{l-1})) + z_{l-1}
    z_l  = MLP(LN(z'_l))   + z'_l

    Args:
        embed_dim: 입력/출력 차원 D.
        num_heads: MSA 헤드 수.
        mlp_dim:   MLP 내부 확장 차원.
        dropout:   MSA 및 MLP dropout 비율.
    """

    def __init__(
        self,
        embed_dim: int = 256,
        num_heads: int = 8,
        mlp_dim: int = 512,
        dropout: float = 0.1,
    ) -> None:
        super().__init__()
        self.norm1 = nn.LayerNorm(embed_dim)
        self.norm2 = nn.LayerNorm(embed_dim)
        self.attn = MultiHeadSelfAttention(embed_dim, num_heads, dropout)
        self.mlp = nn.Sequential(
            nn.Linear(embed_dim, mlp_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(mlp_dim, embed_dim),
            nn.Dropout(dropout),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: (B, seq, D)
        Returns:
            (B, seq, D)
        """
        attn_out, _ = self.attn(self.norm1(x))
        x = x + attn_out
        x = x + self.mlp(self.norm2(x))
        return x


# ─────────────────────────────────────────────────────────────────────────────
# ViT
# 논문 Figure 1 및 Section 3.1
# ─────────────────────────────────────────────────────────────────────────────

class ViT(nn.Module):
    """Vision Transformer (ViT) for image classification.

    PatchEmbedding → CLS token → Positional Encoding → Dropout
    → TransformerEncoderBlock × L → LayerNorm → MLP Head

    config.yaml 파라미터 매핑:
        dataset.image_size  → img_size
        vit.patch_size      → patch_size
        dataset.num_classes → num_classes
        vit.hidden_dim      → embed_dim   (논문의 D)
        vit.num_heads       → num_heads
        vit.num_layers      → num_layers
        vit.mlp_dim         → mlp_dim
        vit.dropout         → dropout

    Args:
        img_size:    입력 이미지 한 변의 크기 (CIFAR-10 기준 32).
        patch_size:  패치 크기 P (CIFAR-10 기준 4).
        in_channels: 입력 채널 수 (RGB = 3).
        num_classes: 분류 클래스 수.
        embed_dim:   Transformer hidden 차원 D.
        num_heads:   MSA 헤드 수.
        num_layers:  Transformer Encoder 블록 반복 수 L.
        mlp_dim:     MLP 내부 확장 차원.
        dropout:     Dropout 비율.
    """

    def __init__(
        self,
        img_size: int = 32,
        patch_size: int = 4,
        in_channels: int = 3,
        num_classes: int = 10,
        embed_dim: int = 256,
        num_heads: int = 8,
        num_layers: int = 6,
        mlp_dim: int = 512,
        dropout: float = 0.1,
    ) -> None:
        super().__init__()
        self.num_patches = (img_size // patch_size) ** 2

        self.patch_embedding = PatchEmbedding(patch_size, in_channels, embed_dim)
        self.cls_token = nn.Parameter(torch.zeros(1, 1, embed_dim))
        self.pos_embedding = nn.Parameter(torch.zeros(1, self.num_patches + 1, embed_dim))
        self.dropout = nn.Dropout(dropout)

        self.encoder_blocks = nn.ModuleList([
            TransformerEncoderBlock(embed_dim, num_heads, mlp_dim, dropout)
            for _ in range(num_layers)
        ])

        self.norm = nn.LayerNorm(embed_dim)
        self.mlp_head = nn.Linear(embed_dim, num_classes)

        self._init_weights()

    def _init_weights(self) -> None:
        nn.init.trunc_normal_(self.cls_token, std=0.02)
        nn.init.trunc_normal_(self.pos_embedding, std=0.02)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: (B, C, H, W)
        Returns:
            logits: (B, num_classes)
        """
        B = x.shape[0]

        x = self.patch_embedding(x)                         # (B, N, D)
        cls_token = self.cls_token.expand(B, -1, -1)        # (B, 1, D)
        x = torch.cat([cls_token, x], dim=1)                # (B, N+1, D)
        x = self.dropout(x + self.pos_embedding)

        for block in self.encoder_blocks:
            x = block(x)

        x = self.norm(x)
        return self.mlp_head(x[:, 0])                       # CLS token → (B, num_classes)


# ─────────────────────────────────────────────────────────────────────────────
# CNN Baseline
# ─────────────────────────────────────────────────────────────────────────────

class SimpleCNN(nn.Module):
    """4-block ConvNet baseline for CIFAR-10.

    Conv Block × 4 → Flatten → FC × 3

    채널 구성: 3 → 32 → 64 → 128 → 256
    MaxPool(2) × 4 → feature map: 32 → 16 → 8 → 4 → 2
    flatten_size: 256 * 2 * 2 = 1024

    Args:
        num_classes: 분류 클래스 수.
    """

    def __init__(self, num_classes: int = 10) -> None:
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(3, 32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2, 2),

            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2, 2),

            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2, 2),

            nn.Conv2d(128, 256, kernel_size=3, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2, 2),
        )
        self.classifier = nn.Sequential(
            nn.Dropout(0.5),
            nn.Linear(256 * 2 * 2, 512),
            nn.ReLU(inplace=True),
            nn.Dropout(0.3),
            nn.Linear(512, 256),
            nn.ReLU(inplace=True),
            nn.Linear(256, num_classes),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: (B, C, H, W)
        Returns:
            logits: (B, num_classes)
        """
        x = self.features(x)
        x = x.view(x.size(0), -1)
        return self.classifier(x)
