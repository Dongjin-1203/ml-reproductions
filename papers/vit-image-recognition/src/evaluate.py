"""
evaluate.py
-----------
ViT / CNN 모델 평가 유틸리티 모듈.

아래 4가지 평가 기능을 제공한다:
  1. learning_curves  — train/val loss, accuracy per epoch 반환
  2. predict_single   — 이미지 1장 입력 → 예측 클래스 + confidence 반환
  3. count_parameters — 모델별 total / trainable parameter 수 반환
  4. measure_speed    — 배치 단위 inference time 측정 (ms/batch, ms/image)
"""

from __future__ import annotations

import time
from typing import Dict, List, Tuple

import mlflow
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torchvision import transforms
from PIL import Image


# ─────────────────────────────────────────────────────────────────────────────
# 1. 학습 곡선
# ─────────────────────────────────────────────────────────────────────────────

def learning_curves(run_id: str, tracking_uri: str = "./mlruns") -> Dict[str, List[float]]:
    """MLflow run에서 train/val loss·accuracy per epoch을 불러온다.

    Args:
        run_id:       MLflow run ID.
        tracking_uri: MLflow tracking server URI (기본: 로컬 ./mlruns).

    Returns:
        {
            "train_loss":     [epoch1_loss, epoch2_loss, ...],
            "val_loss":       [...],
            "train_accuracy": [...],
            "val_accuracy":   [...],
        }
    """
    mlflow.set_tracking_uri(tracking_uri)
    client = mlflow.tracking.MlflowClient()

    metric_keys = ["train_loss", "val_loss", "train_accuracy", "val_accuracy"]
    result: Dict[str, List[float]] = {}

    for key in metric_keys:
        history = client.get_metric_history(run_id, key)
        result[key] = [m.value for m in sorted(history, key=lambda m: m.step)]

    return result


# ─────────────────────────────────────────────────────────────────────────────
# 2. 단일 이미지 추론
# ─────────────────────────────────────────────────────────────────────────────

CIFAR10_CLASSES = [
    "airplane", "automobile", "bird", "cat", "deer",
    "dog", "frog", "horse", "ship", "truck",
]

_EVAL_TRANSFORM = transforms.Compose([
    transforms.Resize((32, 32)),
    transforms.ToTensor(),
    transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2470, 0.2435, 0.2616)),
])


@torch.no_grad()
def predict_single(
    model: nn.Module,
    image: Image.Image,
    device: torch.device,
    class_names: List[str] = CIFAR10_CLASSES,
) -> Tuple[str, float]:
    """PIL 이미지 1장을 받아 예측 클래스명과 confidence를 반환한다.

    Args:
        model:       추론할 nn.Module (eval 모드 권장).
        image:       PIL.Image 입력.
        device:      추론 디바이스.
        class_names: 클래스 이름 목록 (순서는 학습 데이터셋과 동일해야 함).

    Returns:
        (predicted_class_name, confidence)  — confidence는 0~1 사이 float.
    """
    model.eval()
    tensor = _EVAL_TRANSFORM(image).unsqueeze(0).to(device)  # (1, C, H, W)
    logits = model(tensor)                                    # (1, num_classes)
    probs  = torch.softmax(logits, dim=1)
    conf, pred_idx = probs.max(dim=1)
    return class_names[pred_idx.item()], conf.item()


@torch.no_grad()
def predict_all_classes(
    model: nn.Module,
    image: Image.Image,
    device: torch.device,
    class_names: List[str] = CIFAR10_CLASSES,
) -> Dict[str, float]:
    """모든 클래스에 대한 confidence를 dict로 반환한다.

    Args:
        model:       추론할 nn.Module.
        image:       PIL.Image 입력.
        device:      추론 디바이스.
        class_names: 클래스 이름 목록.

    Returns:
        {"airplane": 0.02, "automobile": 0.85, ...}
    """
    model.eval()
    tensor = _EVAL_TRANSFORM(image).unsqueeze(0).to(device)
    logits = model(tensor)
    probs  = torch.softmax(logits, dim=1).squeeze(0)  # (num_classes,)
    return {cls: probs[i].item() for i, cls in enumerate(class_names)}


# ─────────────────────────────────────────────────────────────────────────────
# 3. 파라미터 수
# ─────────────────────────────────────────────────────────────────────────────

def count_parameters(model: nn.Module) -> Dict[str, int]:
    """모델의 total / trainable parameter 수를 반환한다.

    Args:
        model: nn.Module.

    Returns:
        {"total": int, "trainable": int}
    """
    total      = sum(p.numel() for p in model.parameters())
    trainable  = sum(p.numel() for p in model.parameters() if p.requires_grad)
    return {"total": total, "trainable": trainable}


# ─────────────────────────────────────────────────────────────────────────────
# 4. 추론 속도 측정
# ─────────────────────────────────────────────────────────────────────────────

@torch.no_grad()
def measure_speed(
    model: nn.Module,
    loader: DataLoader,
    device: torch.device,
    num_batches: int = 50,
    warmup_batches: int = 5,
) -> Dict[str, float]:
    """배치 단위 inference time을 측정한다.

    GPU 사용 시 CUDA 이벤트로 정밀 측정하고,
    CPU 사용 시 time.perf_counter를 사용한다.

    Args:
        model:          추론할 nn.Module.
        loader:         DataLoader (배치 크기 정보 포함).
        device:         추론 디바이스.
        num_batches:    측정에 사용할 배치 수.
        warmup_batches: 측정 전 워밍업 배치 수 (결과에 미포함).

    Returns:
        {
            "ms_per_batch": float,  # 배치당 평균 inference time (ms)
            "ms_per_image": float,  # 이미지당 평균 inference time (ms)
            "batch_size":   int,
        }
    """
    model.eval()
    model.to(device)
    use_cuda = device.type == "cuda"

    times: List[float] = []
    batch_size = loader.batch_size or 1

    data_iter = iter(loader)

    # Warmup
    for _ in range(warmup_batches):
        try:
            images, _ = next(data_iter)
        except StopIteration:
            data_iter = iter(loader)
            images, _ = next(data_iter)
        images = images.to(device)
        model(images)

    if use_cuda:
        torch.cuda.synchronize()

    # 측정
    for i in range(num_batches):
        try:
            images, _ = next(data_iter)
        except StopIteration:
            data_iter = iter(loader)
            images, _ = next(data_iter)
        images = images.to(device)

        if use_cuda:
            start_event = torch.cuda.Event(enable_timing=True)
            end_event   = torch.cuda.Event(enable_timing=True)
            start_event.record()
            model(images)
            end_event.record()
            torch.cuda.synchronize()
            elapsed_ms = start_event.elapsed_time(end_event)
        else:
            t0 = time.perf_counter()
            model(images)
            elapsed_ms = (time.perf_counter() - t0) * 1000.0

        times.append(elapsed_ms)

    avg_ms_per_batch = sum(times) / len(times)
    avg_ms_per_image = avg_ms_per_batch / batch_size

    return {
        "ms_per_batch": avg_ms_per_batch,
        "ms_per_image": avg_ms_per_image,
        "batch_size":   batch_size,
    }
