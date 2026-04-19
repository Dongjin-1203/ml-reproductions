# An Image is Worth 16x16 Words: Transformers for Image Recognition at Scale

## 논문 정보

- **제목**: An Image is Worth 16x16 Words: Transformers for Image Recognition at Scale
- **저자**: Dosovitskiy et al. (Google Brain)
- **발표**: ICLR 2021
- **📄 논문 링크**: https://arxiv.org/abs/2010.11929

---

## 핵심 아이디어

### 풀려는 문제

CNN 없이 순수 Transformer 구조만으로 이미지 분류를 수행할 수 있는가?  
NLP에서 검증된 Transformer의 확장성(scalability)을 vision 태스크에 적용한다.

### 제안 방법의 핵심 메커니즘

#### 1. Patch Embedding (Section 3.1)

이미지를 고정 크기의 패치(P×P)로 분할한 뒤 각 패치를 1D 벡터로 flatten하여 선형 투영한다.

```
이미지 x ∈ R^(H×W×C)
→ N개의 패치, N = HW / P²
→ 각 패치 x_p ∈ R^(P²·C)
→ 선형 투영 E ∈ R^(P²·C × D) 로 임베딩
```

#### 2. Multi-Head Self-Attention (MSA, Section 3.1)

표준 Transformer Encoder의 MSA를 그대로 적용.  
`[CLS]` 토큰을 시퀀스 앞에 붙여 최종 분류에 사용한다.

```
z_0 = [x_class; x_p¹E; x_p²E; ...; x_pᴺE] + E_pos

MSA(Q, K, V) = softmax(QKᵀ / √d_k) · V
```

#### 3. MLP Head (Section 3.1)

Transformer Encoder 출력의 `[CLS]` 토큰 위치 벡터만 추출하여 MLP로 분류.

```
y = MLP(LayerNorm(z_L^0))
```

---

## 구현 포인트

- `notebooks/model.ipynb` 에서 Patch Embedding, Positional Encoding, Transformer Encoder Block, MLP Head를 순서대로 직접 구현
- CIFAR-10(32×32)에 맞게 `patch_size=4` 사용 (논문은 ImageNet 기준 16×16)
- CNN baseline은 외부 모델(ResNet 등) 없이 간단한 ConvNet으로 직접 설계
- `shared/`는 동일 분류 논문이 2개 이상일 때만 이동 예정 → 현재는 `src/` 로컬에 유지
- 모든 하이퍼파라미터는 `configs/config.yaml`에서 로드

---

## 실험 결과

### 환경

- 데이터셋: CIFAR-10
- 비교: CNN (baseline) vs ViT

### 결과 테이블

| 모델 | Test Accuracy | Parameters | Inference Time (ms/img) | Epochs |
|------|:-------------:|:----------:|:-----------------------:|:------:|
| CNN (baseline) | - | - | - | - |
| ViT | - | - | - | - |

> 추후 실험 완료 후 채울 것

---

## 배운 점 / 인사이트

> 구현 및 실험 완료 후 작성 예정
