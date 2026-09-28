---
title: "[논문리뷰] Preserving LLM Capabilities through Calibration Data Curation: From Analysis to Optimization"
last_modified_at: 2026-08-18
categories: [논문리뷰]
tags: [LLM, Paper]
use_math: true
classes: wide
notion_source: "https://app.notion.com/p/3c03ba5c82c98097b079e9a8c178bfd7"
---

{% raw %}
> 작성 중인 학습 노트입니다. 현재까지 작성한 원문을 옮겼으며 이후 보완할 예정입니다.

## 서론
Post-training compression(사후 학습 압축)은 LLM의 규모를 축소하고 효율적인 추론을 촉진하기 위해 널리 사용되는 접근 방식이다. pruning 및 quantization을 포함한 다양한 압축 방법에서, 보정 데이터(calibration data)는 가중치 중요도와 활성화 동적 범위(activation dynamic ranges)에 대한 정보를 제공함으로써 중요한 역할을 한다. 본 논문에서는 수학 문제 해결 및 코드 생성과 같은 고수준의 복잡한 추론 능력에 대한 보정 데이터의 영향을 탐구한다. 이러한 보정데이터는 근본적인 메커니즘에서 활성화 공간에서의 대표성과 다양성이 영향을 끼친다는 것을 발견했다. 따라서 이러한 관찰과 분석을 바탕으로 보정 데이터 큐레이션 프레임워크를 제안하며, 이를 통해 기존 사후 학습 압축 방법이 중요한 LLM 능력을 보존하는 성능을 향상시킨다
{% endraw %}
