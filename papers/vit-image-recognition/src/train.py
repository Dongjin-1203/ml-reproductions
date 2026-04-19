"""
train.py
--------
CNN baseline 및 ViT 모델을 공통 학습 루프로 학습하는 스크립트.

config.yaml에서 하이퍼파라미터를 로드하고, MLflow로 실험을 추적한다.
val_accuracy 기준 best 모델을 체크포인트로 저장한다.

실행 예시:
    python src/train.py --config configs/config.yaml --model vit
    python src/train.py --config configs/config.yaml --model cnn
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path
from typing import Tuple

import mlflow
import torch
import torch.nn as nn
import yaml
from torch.optim import AdamW, Adam, SGD
from torch.optim.lr_scheduler import CosineAnnealingLR, StepLR
from torch.utils.data import DataLoader

# 노트북에서 구현한 모델을 import (구현 완료 후 경로 수정)
# from notebooks.model import ViT, CNNBaseline  # 대신 아래와 같이 사용 가능
# 현재는 model.py가 없으므로 런타임에서 직접 import하도록 안내 주석 처리


def load_config(config_path: str | Path) -> dict:
    with open(config_path, "r") as f:
        return yaml.safe_load(f)


# ─────────────────────────────────────────────────────────────────────────────
# Optimizer / Scheduler 팩토리
# ─────────────────────────────────────────────────────────────────────────────

def build_optimizer(
    model: nn.Module,
    cfg: dict,
) -> torch.optim.Optimizer:
    """config의 optimizer 설정에 따라 optimizer를 생성한다."""
    lr = cfg["training"]["learning_rate"]
    wd = cfg["training"]["weight_decay"]
    name = cfg["training"]["optimizer"].lower()

    if name == "adamw":
        return AdamW(model.parameters(), lr=lr, weight_decay=wd)
    elif name == "adam":
        return Adam(model.parameters(), lr=lr, weight_decay=wd)
    elif name == "sgd":
        return SGD(model.parameters(), lr=lr, momentum=0.9, weight_decay=wd)
    else:
        raise ValueError(f"Unsupported optimizer: {name}")


def build_scheduler(
    optimizer: torch.optim.Optimizer,
    cfg: dict,
) -> torch.optim.lr_scheduler._LRScheduler | None:
    """config의 scheduler 설정에 따라 LR scheduler를 생성한다."""
    name   = cfg["training"]["scheduler"].lower()
    epochs = cfg["training"]["epochs"]

    if name == "cosine":
        return CosineAnnealingLR(optimizer, T_max=epochs)
    elif name == "step":
        return StepLR(optimizer, step_size=epochs // 3, gamma=0.1)
    elif name == "none":
        return None
    else:
        raise ValueError(f"Unsupported scheduler: {name}")


# ─────────────────────────────────────────────────────────────────────────────
# 단일 epoch 학습 / 평가
# ─────────────────────────────────────────────────────────────────────────────

def train_one_epoch(
    model: nn.Module,
    loader: DataLoader,
    criterion: nn.Module,
    optimizer: torch.optim.Optimizer,
    device: torch.device,
) -> Tuple[float, float]:
    """한 epoch 학습 후 (avg_loss, accuracy)를 반환한다."""
    model.train()
    total_loss, correct, total = 0.0, 0, 0

    for images, labels in loader:
        images, labels = images.to(device), labels.to(device)

        optimizer.zero_grad()
        logits = model(images)
        loss   = criterion(logits, labels)
        loss.backward()
        optimizer.step()

        total_loss += loss.item() * images.size(0)
        preds       = logits.argmax(dim=1)
        correct    += (preds == labels).sum().item()
        total      += images.size(0)

    return total_loss / total, correct / total


@torch.no_grad()
def evaluate(
    model: nn.Module,
    loader: DataLoader,
    criterion: nn.Module,
    device: torch.device,
) -> Tuple[float, float]:
    """평가 루프. (avg_loss, accuracy)를 반환한다."""
    model.eval()
    total_loss, correct, total = 0.0, 0, 0

    for images, labels in loader:
        images, labels = images.to(device), labels.to(device)
        logits  = model(images)
        loss    = criterion(logits, labels)

        total_loss += loss.item() * images.size(0)
        preds       = logits.argmax(dim=1)
        correct    += (preds == labels).sum().item()
        total      += images.size(0)

    return total_loss / total, correct / total


# ─────────────────────────────────────────────────────────────────────────────
# 공통 학습 루프
# ─────────────────────────────────────────────────────────────────────────────

def train(
    model: nn.Module,
    train_loader: DataLoader,
    val_loader: DataLoader,
    cfg: dict,
    model_name: str,
    device: torch.device,
) -> None:
    """CNN 또는 ViT를 학습하는 공통 루프.

    MLflow에 loss, accuracy, epoch, learning_rate를 로깅하고,
    val_accuracy 기준 best model을 체크포인트로 저장한다.

    Args:
        model:        학습할 nn.Module.
        train_loader: 학습 DataLoader.
        val_loader:   검증 DataLoader.
        cfg:          config.yaml 내용을 담은 dict.
        model_name:   "vit" 또는 "cnn" — MLflow run name과 체크포인트 파일명에 사용.
        device:       학습 디바이스.
    """
    epochs        = cfg["training"]["epochs"]
    warmup_epochs = cfg["training"].get("warmup_epochs", 0)
    ckpt_dir      = Path(cfg["checkpoint"]["dir"])
    ckpt_dir.mkdir(parents=True, exist_ok=True)

    mlflow_cfg  = cfg["mlflow"]
    run_name    = mlflow_cfg[f"run_name_{model_name}"]

    optimizer  = build_optimizer(model, cfg)
    scheduler  = build_scheduler(optimizer, cfg)
    criterion  = nn.CrossEntropyLoss()

    model.to(device)

    mlflow.set_tracking_uri(mlflow_cfg["tracking_uri"])
    mlflow.set_experiment(mlflow_cfg["experiment_name"])

    best_val_acc = 0.0

    with mlflow.start_run(run_name=run_name):
        # 하이퍼파라미터 한 번에 로깅
        mlflow.log_params({
            "model":          model_name,
            "epochs":         epochs,
            "batch_size":     cfg["training"]["batch_size"],
            "learning_rate":  cfg["training"]["learning_rate"],
            "optimizer":      cfg["training"]["optimizer"],
            "scheduler":      cfg["training"]["scheduler"],
            "weight_decay":   cfg["training"]["weight_decay"],
        })

        for epoch in range(1, epochs + 1):
            # Warmup: linear LR ramp-up
            if epoch <= warmup_epochs:
                warmup_lr = cfg["training"]["learning_rate"] * epoch / warmup_epochs
                for pg in optimizer.param_groups:
                    pg["lr"] = warmup_lr

            epoch_start = time.time()

            train_loss, train_acc = train_one_epoch(
                model, train_loader, criterion, optimizer, device
            )
            val_loss, val_acc = evaluate(model, val_loader, criterion, device)

            current_lr = optimizer.param_groups[0]["lr"]
            epoch_time = time.time() - epoch_start

            # MLflow 메트릭 로깅
            mlflow.log_metrics(
                {
                    "train_loss":     train_loss,
                    "train_accuracy": train_acc,
                    "val_loss":       val_loss,
                    "val_accuracy":   val_acc,
                    "learning_rate":  current_lr,
                    "epoch_time_sec": epoch_time,
                },
                step=epoch,
            )

            print(
                f"[{model_name.upper()}] Epoch {epoch:3d}/{epochs} | "
                f"train_loss={train_loss:.4f} train_acc={train_acc:.4f} | "
                f"val_loss={val_loss:.4f} val_acc={val_acc:.4f} | "
                f"lr={current_lr:.2e}"
            )

            # Best model 체크포인트 저장
            if val_acc > best_val_acc:
                best_val_acc = val_acc
                ckpt_path = ckpt_dir / f"best_{model_name}.pt"
                torch.save(
                    {
                        "epoch":      epoch,
                        "model_state": model.state_dict(),
                        "val_accuracy": val_acc,
                    },
                    ckpt_path,
                )
                mlflow.log_artifact(str(ckpt_path))
                print(f"  ✓ Saved best checkpoint (val_acc={val_acc:.4f})")

            # Scheduler step (warmup 이후부터)
            if scheduler is not None and epoch > warmup_epochs:
                scheduler.step()

        mlflow.log_metric("best_val_accuracy", best_val_acc)
        print(f"\n[{model_name.upper()}] Training done. Best val_acc={best_val_acc:.4f}")


# ─────────────────────────────────────────────────────────────────────────────
# 진입점
# ─────────────────────────────────────────────────────────────────────────────

def main() -> None:
    parser = argparse.ArgumentParser(description="Train ViT or CNN on CIFAR-10")
    parser.add_argument(
        "--config",
        type=str,
        default="configs/config.yaml",
        help="config.yaml 경로 (papers/vit-image-recognition 기준 상대 경로)",
    )
    parser.add_argument(
        "--model",
        type=str,
        choices=["vit", "cnn"],
        required=True,
        help="학습할 모델 선택: vit | cnn",
    )
    args = parser.parse_args()

    cfg    = load_config(args.config)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    # 데이터 로더
    from dataset import get_dataloaders
    train_loader, val_loader, _ = get_dataloaders(args.config)

    # 모델 초기화
    # ─────────────────────────────────────────────────────────
    # 아래 import는 notebooks/model.ipynb 구현 완료 후 적절한
    # 모듈 경로로 수정하세요.
    # 예시:
    #   from models import ViT, CNNBaseline
    # ─────────────────────────────────────────────────────────
    ds_cfg = cfg["dataset"]
    if args.model == "vit":
        vc = cfg["vit"]
        # model = ViT(
        #     image_size=ds_cfg["image_size"],
        #     patch_size=vc["patch_size"],
        #     num_classes=ds_cfg["num_classes"],
        #     hidden_dim=vc["hidden_dim"],
        #     num_layers=vc["num_layers"],
        #     num_heads=vc["num_heads"],
        #     mlp_dim=vc["mlp_dim"],
        #     dropout=vc["dropout"],
        #     emb_dropout=vc["emb_dropout"],
        #     pool=vc["pool"],
        # )
        raise NotImplementedError(
            "ViT 모델 구현 후 위 주석을 해제하고 import를 추가하세요."
        )
    else:
        cc = cfg["cnn"]
        # model = CNNBaseline(
        #     num_classes=ds_cfg["num_classes"],
        #     num_conv_blocks=cc["num_conv_blocks"],
        #     base_channels=cc["base_channels"],
        #     channel_multiplier=cc["channel_multiplier"],
        #     kernel_size=cc["kernel_size"],
        #     pool_size=cc["pool_size"],
        #     fc_hidden_dim=cc["fc_hidden_dim"],
        #     dropout=cc["dropout"],
        # )
        raise NotImplementedError(
            "CNNBaseline 모델 구현 후 위 주석을 해제하고 import를 추가하세요."
        )

    train(model, train_loader, val_loader, cfg, args.model, device)


if __name__ == "__main__":
    main()
