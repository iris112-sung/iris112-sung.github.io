---
title: "[논문리뷰] Flashattention-3: Fast and Accurate Attention with Asynchrony and Low-precision"
last_modified_at: 2026-08-09
categories: [논문리뷰]
tags: [LLM, Paper]
use_math: true
classes: wide
notion_source: "https://app.notion.com/p/39c3ba5c82c980d1a138c1706dadbb52"
---

{% raw %}
> 작성 중인 학습 노트입니다. 현재까지 작성한 원문을 옮겼으며 이후 보완할 예정입니다.

Attention은 LLM과 Long-context 어플리케이션에서 병목을 일으키는 주 원인이다. Flashattention은 그러한 병목을 줄이기 위해 메모리 읽기/쓰기를 최소화 하여 GPU에서 어텐션 속도를 증가시킨다. 하지만 이 방식은 최근 하드웨어의 새로운 기능을 완전히 사용하지 못한다.Flashattention-2는 H-100에서 35%의 활용률을 달성하는데에 그쳤다. 따라서 본 논문의 연구에선 Hopper GPU에서 어텐션 속도를 높이기 위한 주된 3가지 방법을 소개한다. 첫번째는 워프 전문화(warp-specialization)을 통해 전체 연산과 데이터으 이동을 중첩시킨다. 두번째론
{% endraw %}
