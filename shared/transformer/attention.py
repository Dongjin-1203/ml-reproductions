"""
shared/transformer/attention.py
--------------------------------
Transformer 계열 논문에서 공통으로 사용하는 Attention 모듈.

현재는 구조만 예약되어 있으며, 동일 분류의 논문이 2개 이상 구현되었을 때
각 논문 src/에서 중복되는 코드를 이 모듈로 리팩터링한다.

포함 예정:
  - MultiHeadSelfAttention
  - MultiHeadCrossAttention
  - PositionalEncoding (sinusoidal / learnable)
"""
