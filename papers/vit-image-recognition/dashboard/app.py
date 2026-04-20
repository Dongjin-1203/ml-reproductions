"""
app.py
------
ViT vs CNN CIFAR-10 비교 Streamlit 대시보드.

4개 탭으로 구성:
  Tab 1 [학습 곡선]     : MLflow에서 run 선택 → Loss/Accuracy 그래프 (CNN vs ViT 동시 표시)
  Tab 2 [추론]          : 이미지 업로드 → CNN/ViT 각각 예측 결과 + confidence bar 표시
  Tab 3 [파라미터 비교] : 두 모델 파라미터 수 비교 bar chart
  Tab 4 [속도 비교]     : inference time 비교 bar chart

실행:
    streamlit run dashboard/app.py
"""

from __future__ import annotations

import sys
from pathlib import Path

import mlflow
import plotly.graph_objects as go
import streamlit as st
import torch
import yaml
from PIL import Image

# src/ 모듈 경로 추가 (dashboard/ 기준 상대 경로)
ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(ROOT / "src"))

from evaluate import (
    CIFAR10_CLASSES,
    count_parameters,
    learning_curves,
    measure_speed,
    predict_all_classes,
    predict_single,
)

# ─────────────────────────────────────────────────────────────────────────────
# 설정 로드
# ─────────────────────────────────────────────────────────────────────────────

CONFIG_PATH = ROOT / "configs" / "config.yaml"


@st.cache_data
def load_config() -> dict:
    with open(CONFIG_PATH, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


# ─────────────────────────────────────────────────────────────────────────────
# MLflow 유틸
# ─────────────────────────────────────────────────────────────────────────────

@st.cache_data
def get_mlflow_runs(tracking_uri: str, experiment_name: str) -> list[dict]:
    """MLflow에서 experiment의 모든 run을 불러온다."""
    mlflow.set_tracking_uri(tracking_uri)
    client = mlflow.tracking.MlflowClient()
    experiment = client.get_experiment_by_name(experiment_name)
    if experiment is None:
        return []
    runs = client.search_runs(
        experiment_ids=[experiment.experiment_id],
        order_by=["start_time DESC"],
    )
    return [
        {"run_id": r.info.run_id, "run_name": r.data.tags.get("mlflow.runName", r.info.run_id)}
        for r in runs
    ]


# ─────────────────────────────────────────────────────────────────────────────
# 모델 로드 유틸
# ─────────────────────────────────────────────────────────────────────────────

@st.cache_resource
def load_model(model_type: str, cfg: dict) -> torch.nn.Module | None:
    """체크포인트에서 모델을 로드한다.

    모델 클래스(ViT, CNNBaseline)는 notebooks/model.ipynb 구현 완료 후
    아래 import 주석을 해제하고 경로를 수정하세요.
    """
    # from models import ViT, CNNBaseline  # 구현 완료 후 import 추가
    ckpt_path = ROOT / cfg["checkpoint"]["dir"] / f"best_{model_type}.pt"
    if not ckpt_path.exists():
        return None

    ds_cfg = cfg["dataset"]
    device = torch.device("cpu")  # 대시보드는 CPU 추론으로 충분

    if model_type == "vit":
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
        return None  # 구현 후 위 코드로 교체
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
        return None  # 구현 후 위 코드로 교체

    ckpt = torch.load(ckpt_path, map_location=device)
    model.load_state_dict(ckpt["model_state"])
    model.eval()
    return model


# ─────────────────────────────────────────────────────────────────────────────
# Plotly 헬퍼
# ─────────────────────────────────────────────────────────────────────────────

def make_line_chart(
    data: dict[str, list[float]],
    title: str,
    y_label: str,
) -> go.Figure:
    fig = go.Figure()
    for name, values in data.items():
        fig.add_trace(go.Scatter(
            y=values,
            x=list(range(1, len(values) + 1)),
            mode="lines+markers",
            name=name,
        ))
    fig.update_layout(
        title=title,
        xaxis_title="Epoch",
        yaxis_title=y_label,
        hovermode="x unified",
    )
    return fig


def make_bar_chart(
    categories: list[str],
    values: list[float],
    title: str,
    y_label: str,
    colors: list[str] | None = None,
) -> go.Figure:
    fig = go.Figure(go.Bar(
        x=categories,
        y=values,
        marker_color=colors or ["#636EFA", "#EF553B"],
        text=[f"{v:,.0f}" if v > 1000 else f"{v:.2f}" for v in values],
        textposition="outside",
    ))
    fig.update_layout(title=title, yaxis_title=y_label)
    return fig


# ─────────────────────────────────────────────────────────────────────────────
# 앱 메인
# ─────────────────────────────────────────────────────────────────────────────

def main() -> None:
    st.set_page_config(
        page_title="ViT vs CNN — CIFAR-10",
        page_icon="🔬",
        layout="wide",
    )
    st.title("ViT vs CNN — CIFAR-10 비교 대시보드")
    st.caption("논문: An Image is Worth 16x16 Words (Dosovitskiy et al., ICLR 2021)")

    cfg          = load_config()
    tracking_uri = str(ROOT / cfg["mlflow"]["tracking_uri"])
    exp_name     = cfg["mlflow"]["experiment_name"]

    tab1, tab2, tab3, tab4 = st.tabs([
        "📈 학습 곡선",
        "🔍 추론",
        "📊 파라미터 비교",
        "⚡ 속도 비교",
    ])

    # ── Tab 1: 학습 곡선 ──────────────────────────────────────────────────────
    with tab1:
        st.subheader("학습 곡선 — Loss & Accuracy")
        runs = get_mlflow_runs(tracking_uri, exp_name)

        if not runs:
            st.info("MLflow run이 없습니다. 먼저 학습을 실행하세요.")
        else:
            run_options = {r["run_name"]: r["run_id"] for r in runs}
            selected_names = st.multiselect(
                "비교할 run 선택 (CNN, ViT 각각 선택 권장)",
                options=list(run_options.keys()),
                default=list(run_options.keys())[:2],
            )

            if selected_names:
                loss_data, acc_data = {}, {}
                for name in selected_names:
                    curves = learning_curves(run_options[name], tracking_uri)
                    loss_data[f"{name} — train"] = curves.get("train_loss", [])
                    loss_data[f"{name} — val"]   = curves.get("val_loss", [])
                    acc_data[f"{name} — train"]  = curves.get("train_accuracy", [])
                    acc_data[f"{name} — val"]    = curves.get("val_accuracy", [])

                col1, col2 = st.columns(2)
                with col1:
                    st.plotly_chart(
                        make_line_chart(loss_data, "Loss per Epoch", "Loss"),
                        use_container_width=True,
                    )
                with col2:
                    st.plotly_chart(
                        make_line_chart(acc_data, "Accuracy per Epoch", "Accuracy"),
                        use_container_width=True,
                    )

    # ── Tab 2: 추론 ───────────────────────────────────────────────────────────
    with tab2:
        st.subheader("이미지 업로드 → CNN / ViT 예측")
        uploaded = st.file_uploader(
            "이미지 파일 업로드 (jpg, png, ...)",
            type=["jpg", "jpeg", "png", "bmp"],
        )

        if uploaded is not None:
            image = Image.open(uploaded).convert("RGB")
            st.image(image, caption="업로드된 이미지", width=200)

            device     = torch.device("cpu")
            vit_model  = load_model("vit", cfg)
            cnn_model  = load_model("cnn", cfg)

            col1, col2 = st.columns(2)

            for col, model, model_name in [
                (col1, cnn_model, "CNN"),
                (col2, vit_model, "ViT"),
            ]:
                with col:
                    st.markdown(f"#### {model_name}")
                    if model is None:
                        st.warning(f"{model_name} 체크포인트를 찾을 수 없습니다.")
                    else:
                        pred_class, confidence = predict_single(model, image, device)
                        all_confs = predict_all_classes(model, image, device)

                        st.metric("예측 클래스", pred_class)
                        st.metric("Confidence", f"{confidence:.2%}")

                        sorted_confs = sorted(all_confs.items(), key=lambda x: x[1], reverse=True)
                        fig = go.Figure(go.Bar(
                            x=[c for c, _ in sorted_confs],
                            y=[v for _, v in sorted_confs],
                            marker_color=["#EF553B" if c == pred_class else "#636EFA"
                                          for c, _ in sorted_confs],
                        ))
                        fig.update_layout(
                            title=f"{model_name} Confidence",
                            xaxis_title="Class",
                            yaxis_title="Probability",
                            yaxis_range=[0, 1],
                        )
                        st.plotly_chart(fig, use_container_width=True)

    # ── Tab 3: 파라미터 비교 ─────────────────────────────────────────────────
    with tab3:
        st.subheader("모델 파라미터 수 비교")

        vit_model = load_model("vit", cfg)
        cnn_model = load_model("cnn", cfg)

        param_data = {}
        for model, name in [(cnn_model, "CNN"), (vit_model, "ViT")]:
            if model is not None:
                counts = count_parameters(model)
                param_data[name] = counts

        if not param_data:
            st.info("모델 체크포인트가 없습니다. 학습 완료 후 확인하세요.")
        else:
            models  = list(param_data.keys())
            totals  = [param_data[m]["total"] for m in models]
            trains  = [param_data[m]["trainable"] for m in models]

            col1, col2 = st.columns(2)
            with col1:
                st.plotly_chart(
                    make_bar_chart(models, totals, "Total Parameters", "Count"),
                    use_container_width=True,
                )
            with col2:
                st.plotly_chart(
                    make_bar_chart(models, trains, "Trainable Parameters", "Count"),
                    use_container_width=True,
                )

            st.dataframe(
                {
                    "Model":      models,
                    "Total":      [f"{t:,}" for t in totals],
                    "Trainable":  [f"{t:,}" for t in trains],
                }
            )

    # ── Tab 4: 속도 비교 ─────────────────────────────────────────────────────
    with tab4:
        st.subheader("Inference Time 비교")

        vit_model = load_model("vit", cfg)
        cnn_model = load_model("cnn", cfg)

        speed_data = {}
        device     = torch.device("cpu")

        if st.button("속도 측정 시작"):
            from dataset import get_dataloaders
            _, _, test_loader = get_dataloaders(CONFIG_PATH)

            for model, name in [(cnn_model, "CNN"), (vit_model, "ViT")]:
                if model is not None:
                    with st.spinner(f"{name} 측정 중..."):
                        result = measure_speed(model, test_loader, device, num_batches=30)
                        speed_data[name] = result

            if not speed_data:
                st.info("모델 체크포인트가 없습니다. 학습 완료 후 확인하세요.")
            else:
                models         = list(speed_data.keys())
                ms_per_batch   = [speed_data[m]["ms_per_batch"] for m in models]
                ms_per_image   = [speed_data[m]["ms_per_image"] for m in models]

                col1, col2 = st.columns(2)
                with col1:
                    st.plotly_chart(
                        make_bar_chart(models, ms_per_batch, "Inference Time (ms/batch)", "ms"),
                        use_container_width=True,
                    )
                with col2:
                    st.plotly_chart(
                        make_bar_chart(models, ms_per_image, "Inference Time (ms/image)", "ms"),
                        use_container_width=True,
                    )

                st.dataframe({
                    "Model":         models,
                    "ms/batch":      [f"{v:.2f}" for v in ms_per_batch],
                    "ms/image":      [f"{v:.3f}" for v in ms_per_image],
                    "batch_size":    [speed_data[m]["batch_size"] for m in models],
                })


if __name__ == "__main__":
    main()
