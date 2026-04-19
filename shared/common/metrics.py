"""
shared/common/metrics.py
--------------------------
논문 분류에 상관없이 공통으로 사용하는 평가 지표 유틸리티.

현재는 구조만 예약되어 있으며, 동일 분류의 논문이 2개 이상 구현되었을 때
각 논문 src/에서 중복되는 코드를 이 모듈로 리팩터링한다.

포함 예정:
  - top_k_accuracy
  - confusion_matrix
  - precision_recall_f1
"""
