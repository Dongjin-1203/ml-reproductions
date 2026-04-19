# ml-reproductions

> AI/ML 논문을 읽고, 직접 구현하며 이해를 검증하는 레포지토리입니다.  
> 각 구현체는 직접 작성한 모델 노트북 + 학습 코드 + Streamlit 대시보드를 포함합니다.

---

## 📌 목적

- 논문의 핵심 아이디어를 코드로 직접 재현하며 깊이 있는 이해
- 재현 과정에서 얻은 인사이트와 실험 결과를 시각적으로 서빙
- AI 엔지니어링 역량 누적 및 포트폴리오 정리

---

## 📂 레포지토리 구조

```
ml-reproductions/
│
├── README.md
│
├── papers/                            # 논문별 구현 디렉토리
│   └── {paper-slug}/                  # 예: attention-is-all-you-need
│       ├── README.md                  # 논문 링크, 요약, 구현 포인트, 실험 결과
│       ├── configs/
│       │   └── config.yaml            # 하이퍼파라미터, 경로 등
│       ├── notebooks/
│       │   └── model.ipynb            # 모델 구현 (직접 작성 / 가이드 주석 기반)
│       ├── src/
│       │   ├── train.py               # 학습 루프
│       │   ├── dataset.py             # 데이터 로딩 / 전처리
│       │   ├── evaluate.py            # 평가 지표
│       │   └── utils.py               # 로컬 유틸
│       ├── dashboard/
│       │   └── app.py                 # Streamlit 대시보드
│       └── requirements.txt
│
└── shared/                            # 모델 분류별 공통 모듈
    ├── transformer/
    │   ├── attention.py               # Multi-head attention, positional encoding 등
    │   └── layers.py                  # Feed-forward, layer norm 등
    ├── diffusion/
    │   ├── noise_scheduler.py         # DDPM, DDIM 스케줄러
    │   └── unet_blocks.py             # U-Net 구성 블록
    ├── generative/
    │   ├── vae_base.py                # VAE encoder/decoder 베이스
    │   └── gan_base.py                # GAN generator/discriminator 베이스
    └── common/
        ├── metrics.py                 # 공통 평가 지표
        └── visualize.py               # 공통 시각화 유틸
```

---

## 📋 구현 목록

| # | 논문 | 분류 | 연도 | 데이터셋 | 구현 상태 | 대시보드 |
|---|------|------|------|----------|-----------|---------|
| 1 | [AN IMAGE IS WORTH 16X16 WORDS: TRANSFORMERS FOR IMAGE RECOGNITION AT SCALE](papers/vit-image-recognition/) | ViT | 2021 | CIFAR-10 | 🔄 진행 중료 | [🚀 실행](#) |
| 2 | [Learning Transferable Visual Models From Natural Language Supervision](papers/CLIP/) | Multi-modal | 2021 | - | 📅 예정 | - |

> 상태: ✅ 완료 / 🔄 진행 중 / 📅 예정

---

## 🚀 빠른 시작

```bash
git clone https://github.com/Dong-1203/ml-reproductions.git
cd papers/{paper-slug}
pip install -r requirements.txt
```

### 모델 구현

각 논문의 `notebooks/model.ipynb` 에서 직접 구현합니다.  
노트북에는 논문의 수식/구조에 따른 가이드 주석이 포함되어 있습니다.

### 학습 실행

```bash
python src/train.py --config configs/config.yaml
```

### Streamlit 대시보드

```bash
streamlit run dashboard/app.py
```

---

## 📄 논문별 README 구성 가이드

각 `papers/{paper-slug}/README.md` 는 아래 구조를 따릅니다:

```
## 논문 정보
- 제목, 저자, 발표 학회/연도
- 📄 논문 링크: https://arxiv.org/abs/...

## 핵심 아이디어
- 풀려는 문제
- 제안 방법의 핵심 메커니즘 (수식 포함 가능)

## 구현 포인트
- notebooks/model.ipynb 에서 주의깊게 구현한 부분
- 공식 구현과 다르게 선택한 부분 및 이유
- shared/ 에서 가져다 쓴 모듈

## 실험 결과
- 논문 원본 결과 vs 재현 결과 비교 표

## 배운 점 / 인사이트
```

---

## 📐 공통 모듈 (`shared/`) 사용 규칙

- 동일 분류의 논문 2개 이상에서 재사용되는 코드만 `shared/`로 이동
- 처음에는 각 논문 `src/utils.py`에 로컬로 작성 → 중복 발생 시 `shared/`로 리팩터링
- `shared/` 모듈은 논문 구현에 의존하지 않아야 함 (단방향 의존성 유지)

---

## 🛠️ 기술 스택

- **언어**: Python 3.10+
- **딥러닝**: PyTorch
- **실험 추적**: MLflow
- **자동 스케줄링**: Airflow (필요 시)
- **서빙**: Streamlit
- **모델 구현**: Jupyter Notebook

---

## 📝 관련 포스팅

구현 과정은 [Velog](https://velog.io/@hambur1203) 에 정리하고 있습니다.

---

## 📜 라이선스

MIT License