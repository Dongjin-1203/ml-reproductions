# CLAUDE.md — ml-reproductions 프로젝트 규칙

Claude Code가 이 레포지토리에서 작업할 때 항상 따라야 하는 규칙 파일입니다.

---

## 레포지토리 목적

AI/ML 논문을 읽고 직접 구현하며 이해를 검증하는 레포지토리.  
각 구현체는 모델 노트북 + 학습 코드 + Streamlit 대시보드로 구성된다.

---

## 디렉토리 구조 규칙

- 논문 구현은 `papers/{paper-slug}/` 단위로 완전히 격리한다.
- 논문 간 공통 모듈은 `shared/{분류}/` 에 위치한다.
  - 분류: `transformer/`, `diffusion/`, `generative/`, `common/`
- **`shared/` 이동 기준**: 동일 분류 논문 **2개 이상**에서 재사용될 때만 이동.  
  처음에는 각 논문 `src/` 에 로컬로 작성하고, 중복 발생 시 리팩터링한다.
- 모든 경로는 `papers/{paper-slug}/` 기준 상대 경로로 작성한다.

---

## 파일 생성 규칙

- **`notebooks/model.ipynb`**: 구현 가이드 주석만 작성. **코드 작성 금지.**  
  모델 구현(PatchEmbedding, Attention 등)은 사용자가 직접 채운다.
- **하드코딩 금지**: 모든 설정값(하이퍼파라미터, 경로 등)은 `configs/config.yaml` 에서 로드한다.
- **논문 PDF 포함 금지**: README에 arxiv 링크만 명시한다.
- 모든 파일 상단에 해당 파일의 역할을 **docstring** 으로 명시한다.
- **타입 힌트 작성 필수** (함수 인자 및 반환값).

---

## 기술 스택

| 역할 | 도구 |
|------|------|
| 언어 | Python 3.10+ |
| 딥러닝 | PyTorch — `nn.Module` 상속 구조 유지 |
| 실험 추적 | MLflow — `train.py` 에 트래킹 코드 필수 포함 |
| 자동 스케줄링 | Airflow (필요 시) |
| 서빙 | Streamlit |

---

## Git 규칙

- **작업 브랜치**: `paper/{slug}` 형식으로 생성한다.
- **`main` 직접 push 금지**. PR을 통해서만 머지한다.
- **커밋 컨벤션**: `{prefix}({slug}): 메시지`

  | prefix | 용도 |
  |--------|------|
  | `feat` | 새 기능 추가 (모델, 학습 코드 등) |
  | `fix` | 버그 수정 |
  | `docs` | README, 주석 수정 |
  | `refactor` | 코드 리팩터링 |
  | `exp` | 실험 결과 추가 / 하이퍼파라미터 변경 |

  > 예: `feat(vit): add patch embedding module`

- **`.gitignore` 필수 항목**: `mlruns/`, `data/`, `*.pt`, `*.pth`, `*.ckpt`

---

## 커밋 단위

파일 성격별로 커밋을 분리한다. 권장 순서:

1. `docs` — README, .gitignore 등 문서/설정
2. `feat` — `shared/` 공통 모듈
3. `feat({slug})` — `papers/{slug}/` 논문 구현 파일
