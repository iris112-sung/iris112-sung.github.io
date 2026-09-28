---
title: "[논문리뷰] FlashAttention-2:Faster Attention with Better Parallelism and Work Partitioning"
last_modified_at: 2026-08-10
categories: [논문리뷰]
tags: [LLM, Paper]
use_math: true
classes: wide
notion_source: "https://app.notion.com/p/3b43ba5c82c98066a42ac508efbd7cdd"
---

{% raw %}
> 작성 중인 학습 노트입니다. 현재까지 작성한 원문을 옮겼으며 이후 보완할 예정입니다.

## 초록
 FlashAttention은 런타임과 메모리 사용량이 이차적으로 증가하여 병목이 발생하던 기존 Attention 레이어의 런타임 속도 향상과 메모리 절감을 달성하였지만 여전히 GEMM(최적화된 행렬 곱셈) 연산만큼 빠르지는 않으며 이론적으로 최대 FLOP/s의 25\~40% 수준에 도달 하는데에 그친다. 본 논문에서는 이러한 비효율성이 GPU의 서로 다른 스레드 블록과 워프간의 최적화되지 않은 작업 분할로 인해 발생하며, 이로 인한 낮은 점유율이나 불필요한 공유 메모리 읽기/쓰기가 발생한다는 점을 관찰하였다. 따라서 이러한 문제를 해결하기 위해 더 나은 작업 분할을 적용한 FlashAttention-2를 제안한다. 본 논문에서는 행렬곱셈이 아닌 FLOPs의 수를 줄이고, Attention 계산을 여러 스레드 블록에 걸쳐 병렬화하여 점유율을 높이며, 각 스레드 블록 내에서 워프 간에 작업을 분배하여 공유 메모리를 통한 통신을 줄였다.
## 서론
{% endraw %}
