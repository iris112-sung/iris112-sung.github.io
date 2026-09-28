---
title: "[논문리뷰] Flashattention : fast and memory-efficient exact attention with io-awareness"
last_modified_at: 2026-08-09
categories: [논문리뷰]
tags: [LLM, Paper]
use_math: true
classes: wide
notion_source: "https://app.notion.com/p/3a33ba5c82c9806b8522f82f65e2e508"
---

{% raw %}
## 초록
 Transformer는 시퀀스 길이가 길어질 수록 속도가 느려지고 메모리 소모가 커진다. 본 논문에서는 기존의 attention 알고리즘이 gpu 메모리 계층간의 읽기 및 쓰기 작업을 고려하는 IO-aware 설계가 결여되어있다는 점을 지적한다.이러한 문제를 타일링(tiling)을 통해 gpu의 HBM(High Bandwidth Memory)과 온칩 SRAM 간의 메모리 읽기/쓰기 횟수를 줄이는 IO-aware을 고려한 정확한 attention 알고리즘인 Flashattention을 제안한다. Flashattention의 I/O 복잡도를 분석한 결과, 표준 attention보다 HBM 접근 횟수가 적으며 다양한 SRAM 크기 범위에서 최적임을 확인했다. Flashattention은 기존 베이스라인보다 빠르게 Transformer를 학습시킨다. 시퀀스 길이가 512인 BERT-large에서는 MLPert 1.1 학습속도 기록 대비 15%의 시간 절약을 보였으며 시퀀스 길이가 1K인 GPT-2에서는 3배, 시퀀스 길이가 1K-4K인 Long-Range Arena에서는 2.4배의 속도 향상을 보였다. 
## 서론
 Transformer 모델은 자연어 처리 및 이미지 분류와 같은 응용 분야에서 가장 널리 사용되는 아키텍처로 자리 잡았다. 하지만 핵심 모듈인 self-attention의 시간 및 메모리 복잡도가 시퀀스 길이의 제곱에 비례하기 때문에 더 긴 컨텍스트를 처리하도록 만드는 것은 여전히 어렵다. 따라서 이러한 attention을 더 빠르고 메모리 효율적으로 만드는 것이 Transformer 모델이 긴 시퀀스에서 겪는 런타임 및 메모리 문제를 해결하는 데 도움이 될 수 있는지는 중요한 문제이다. 

 많은 근사 attention 방법들이 attention의 연산 및 메모리 요구사항을 줄이는 것을 목표로 해왔다. sparse-approximation부터 low-rank approximation 까지 다양한 조합이 이르렀다. 이러한 방법들은 연산 요구 사항을 시퀀스 길이에 대해 선형적으로 또는 거의 선형으로 줄이지만, 많은 경우 표준 attention 대비 실제 wall-clock time 시간 기준의 속도 향상을 보여주지 못하여 널리 채택되지 못했다. 

 본 논문은 기존의 attention 알고리즘이 IO-aware가 되어야 한다는 원칙을 준수하고 있지 않다고 주장한다. 즉, 빠르고 느린 메모리의 여러 계층간의 읽기 및 쓰기 작업을 신중하게 고려해야 한다는 것이다. 현대 gpu에서는 연산 속도가 메모리 속도를 앞질렀으며 Transforme의 대부분의 연산은 메모리 접근에 의해 병목 현상이 발생한다. 데이터 읽기 및 쓰기가 전체 실행 시간의 상당 부분을 차지하는 유사한 메모리 집약적 연산에서 IO-aware 알고리즘은 매우 중요하며, 이는 데이터베이스 조인, 이미지처리, 수치 선형 대수 및 기타 분야 등이 해당된다. 그러나 PyTorch 나 TensorFlow와 같이 딥러닝을 위한 일반적인 Python 인터페이스는 메모리 접근에 대한 세밀한 제어를 허용하지 않는다.

![Flashattention : fast and memory-efficient exact attention with io-awareness](/assets/reviews/flashattention-1.png)
 본 논문에서는 훨씬 적은 메모리 접근으로 정확한 어텐션을 계산하는 새로운 어텐션 알고리즘인 FlashAttention을 제안한다. 이 아키텍쳐의 주된 목표는 HBM으로 어텐션 행렬을 읽고 쓰는 것을 피하는 것이다. 이를 위해선 위 두가지를 이행하야한다.
1. 전체 입력에 접근하지 않고 소프트 맥스 리덕션을 계산한다. 2. 역전파를 위해 큰 중간 어텐션 행렬을 저장하지 않아야 한다. 
이걸 해결하기 위해 2가지 기술을 사용한다. 
1. 어텐션 계산을 재구성하여 입력을 블록으로 나누고 입력 블록에 대해 여러 번 패스를 수행함으로써 타일링 한다. 

2. 순전파에서 소프트맥스 정규화 계수를 저장하여 역전파 시 온칩(on-chip)에서 어텐션을 빠르게 재계산한다
이는 HBM에서 중간 어텐션 행렬을 읽는 표준 방식보다 빠르다. 본 논문에선 CUDA로 FlashAttention을 구현하여 메모리 접근에 대한 세밀한 제어를 달성하고 모든 어텐션을 하나의 gpu 커널로 융합하였다. 재계산으로 인해 FLOPs가 증가함에도 불구하고, FlashAttention은 HBM 접근량이 대폭 감소한 덕분에 표준 어텐션보다 더 빠르게 실행되며(그림 1 오른쪽 참조), 시퀀스 길이에 선형적인 더 적은 메모리를 사용한다. 
 IO 복잡도를 비교해보자면 기존의 어텐션의 복잡도는 Ω(Nd+N²)으로 N ⨉ N 크기의 어텐션 행렬 S와 P를 모두 HBM에 저장하고 읽어야 한다. 따라서 행렬의 크기는 N²이되고 이것 때문에 시퀀스 길이가 길어질수록 메모리 비용이 급격히 증가한다. 반면 FlashAttention의 복잡도는 O(N²d²M⁻¹)으로 시퀀스의 길이를 SRAM의 크기인 M으로 나누는 형태이기에 SRAM의 용량이 커질수록 타일링하여 HBM접근 횟수가 줄어든다. 이는 그림2에서 보이듯 표준 어텐션보다 최대 9배 적은 수치다.
 또한 어떠한 임의의 정확한 어텐션의 알고리즘도 HBM에 접근하는 복잡도가 O(N²d²M⁻¹)보다 더 나아질 수 없다는 하한(lower bound)을 제시한다.
 본 논문에서는 FlashAttention이 메모리 접근 오버헤드 문제를 극복함으로써 근사 어텐션 알고리즘의 잠재력의 실현하는 유용한 기본요소로 활용될 수 있음을 보여준다. 개념 증명으로서, 본 논문에선 FlashAttention보다 2\~4배 빠르고 최대 64k의 시퀀스 길이까지 확장 가능한 희소 어텐션 알고리즘인 block-sparse FlashAttention을 구현한다. 또한 이러한 기본 요소를 기반으로 더 쉽게 개발 하기 위해 FlashAttention을 오픈 소스로 공개한다. 
 또한 FlashAttention은 더 빠른 모델 학습과 더 높은 품질의 모델, Transformer를 더 높은 품질의 모델로써 변환한다는 것과 어텐션을 벤치마킹 할 수 있다는 장점을 가진다.
## 배경
#### 2.1 하드웨어 성능
 본 논문에서는 GPU에 초점을 맞춘다. 그림 1의 왼쪽과 같은 GPU 메모리 계층 구조는 크기와 속도가 다른 여러 형태의 메모리로 구성되며, 메모리 크기와 속도가 반비례한다. 예를들어, A100 GPU는 1.5-2.0TB/s 대역폭의 40-80GB 고대역폭 메모리(HBM)와, 약 19TB/s 정도인 대역폭을 가진 108개의 스트리밍 멀티프로세서 각각에 192KB의 온칩 SRAM을 갖추고 있다. 온칩 SRAM은 HBM보다 대략 10배가량 빠르지만 크기는 훨씬 작다. 연산 속도가 메모리 속도에 비해 빨라짐에 따라 연산은 점점 더 메모리(HBM) 접근에 의해 병목 현상이 발생한다. 따라서 빠른 SRAM을 활용하는 것이 더욱 중요해졌다.
 또한 GPU는 커널을 실행하기 위해 방대한 수의 스레드를 사용한다. 각 커널은 HBM에서 레지스터와 SRAM으로 입력을 로드하고, 연산을 수행한 다음, 결과를 HBM에 쓴다.
 성능 특성에서도 필요성이 들어나는데 연산과 메모리 접근의 균형에 따라 연산은 연산중심 또는 메모리 중심으로 분류 될 수 있다. 이는 1바이트의 데이터를 메모리에서 읽어올 때, GPU가 얼마나 많은 연산을 수행할 수 있는지를 나타낸다. 이 수치가 높을수록 연산 장치가 쉬지 않고 일할 수 있는 연산중심(Compute-bound) 작업에 가깝고, 낮을수록 메모리에서 데이터를 가져오느라 연산 장치가 기다리는 메모리 중심(Memory-bound)작업이 된다.
 결국 표준적인 Attention에서는 sum, softmax, batch norm, layer norm등과 같은 계산시 메모리 중심 작업이 되므로 이러한 메모리 병목을 줄이기 위해 타일링과 커널 퓨전이 필요하다.
 여기서 커널 퓨전은 메모리 중심 연산을 가속화하는 가장 일반적인 접근 방식이다. 동일한 입력에 여러 연산이 적용되는 경우, 각 연산마다 여러 번 로드하는 대신 입력을 HBM애서 한 번만 로드할 수 있다. 컴파일러는 많은 요소별 연산을 자동으로 퓨전할 수 있다. 하지만 모델 학습 과정에서는 역전파를 위해 중간 값들을 여전히 HBM에 기록해야 하므로, 단순한 커널 퓨전의 효율성이 떨어진다.
#### 2.2 표준 어텐션 구현
 표준적인 어텐션 연산은 다음과 같은 3단계로 구성된다.
1. 입력 행렬인 쿼리(Q)와 키(K)를 곱하여 시퀀스 길이(N)의 제곱에 비례하는 크기를 가진 스코어 행렬(S)을 만든다.
	$$
	S=QK ^{T}∈R ^{N×N}
	$$
2. S의 각 행에 소프트맥스 함수를 적용하여 어텐션 가중치 행렬(P)을 얻는다.
$$
P=softmax(S)∈R^{N×N}
$$
1. 어텐션 가중치 P를 값(V) 행렬과 곱하여 최종 출력(O)을 생성한다.
$$
O=PV∈R^{N×d}
$$
 표준 방식에선 중간 결과물인 S와 P를 GPU의 고속 메모리인 SRAM이 아닌, 용량은 크지만 속도가 느린 HBM에 모두 저장한다. 따라서 N²의 행렬은 시퀀스 길이인 N이 커질수록 메모리 사용량이 기하급수적으로 늘게되어 병목이 생긴다. 이 문제는 어텐션 행렬에 적용되는 다른 요소별 연산들인 마스킹이나 드롭아웃에 의해 더욱 악화된다. 

#### 3. FlashAttention: 알고리즘, 분석 및 확장
 본 논문에선 역전파를 위해 큰 중간 행렬을 저장하지 않고 더 적은 HBM 읽기/쓰기로 정확한 어텐션을 계산하는 방법을 보여준다. 이는 메모리 효율적이면서 실제 실행 시간도 더 빠른 어텐션 알고리즘을 산출한다. 또한 FlashAttention을 블록 희소(block-sparse) 어텐션으로 처리하도록 확장함으로써 유용한 기본 연산으로 활용될 수 있음을 보여준다. 설명의 편의를 위해 여기선 순전파에 집중한다.
#### 3.1 타일링과 재계산을 이용한 효율적인 어텐션 알고리즘
 이 연구의 목표는 HBM 접근량을 줄이는것이다.  이 연구에서는 N에 대해 준이차적인 HBM 접근으로 정확한 어텐션을 계산하는 기술적 과제를 극복하기 위해 두 가지 확립된 기법(타일링,재계산)을 적용한다(이는 알고리즘 1에서 설명하며 자세한 알고리즘 공식들은 본문을 참고하길 바람). 핵심 아이디어는 입력 Q,K,V를 블록으로 나누고, 느린 HBM에서 빠른 SRAM으로 로드한 다음, 해당 블록들에 대한 어텐션 출력을 계산하는 것이다.
 각 블록의 출력은 올바른 정규화 계수로 스케일링한 후 합산함으로써, 최종적으로 정확한 결과를 얻는다. 
 타일링(Tiling). 어텐션을 블록 단위로 계산한다. Softmax는 K의 열들을 서로 결합하므로, 스케일링을 사용하여 큰 Softmax를 분해한다.
 재계산(Recomputation). 본 연구의 목표 중 하나는 역전파를 위해 O(N²)의 중간 값을 저장하지 않는 것이다. 역전파는 일반적으로 Q,K,V에 대한 기울기를 계산하기 위해 행렬 S,P를 필요로 한다. 그러나 출력 O와 softmax 정규화 통계량(m,l)을 저장함으로써, SRAM에 있는 Q,K,V 블록으로부터 어텐션 행렬 S와 P를 역전파 과정에서 쉽게 재계산할 수 있다. 이는 선택적 기울기 체크포인팅(selective gradient checkpointing)의 한 형태로 볼 수 있다. 하지만 이는 메모리를 아끼지만 전체 학습 속도는 느려지는 상충관계를 겪어야 했다. 하지만 FlashAttention의 재계산 방식은 더 많은 FLOPs(연산횟수)를 사용하더라도 HBM 접근을 줄임으로써 역전파 속도를 높인다.
#### 3.2 분석: FlashAttention의 IO복잡도
 본 논문은 FlashAttention의 IO복잡도를 분석하여 표준 어텐션 대비 HBM 접근이  크게 감소함을 보여준다. 
 또한 앞서 말했듯 어떠한 정확한 어텐션 알고리즘도 모든 SRAM 크기에 걸쳐 HBM 접근 횟수를 점근적으로 개선할 수 없음을 증명하는 하한을 제시하며 이를 부록 C에서 증명한다.
 본 논문에서는 HBM 접근 횟수가 어텐션 실행 시간을 결정하는 주요 요인임을 검증한다. 그림2 왼쪽에서 FlashAttention이 표준 어텐션보다 더 높은 FLOP(부동소수점 연산 횟수) 수를 가짐에도 불구하고, 훨씬 적은 HBM 접근 횟수로 인해 훨씬 빠른 런타임을 보임을 확인할 수 있다. 그림 2의 중간에서는 FlashAttention의 블록 크기 $`B_{c}`$를 변화시켜 HBM 접근 횟수의 차이를 유도하고 순방향 패스의 런타임을 측정한다. 블록크기가 증가함에 따라 HBM 접근 횟수는 감소하며, 런타임도 감소한다
#### 3.3 확장: 블록 희소 (Block-Sparse) FlashAttention
 본 논문은 FlashAttention을 근사 어텐션으로 확장하여 블록 희소 FlashAttention을 제안하며, 이 알고리즘의 IO 복잡도는 희소성에 비례하는 계수만큼 FlashAttention보다 작다.
## 실험
 본 논문에서는 Transformer 모델 학습에 FlashAttention을 사용하는 것의 영향을 평가한다. 학습 시간과 모델 정확도를 검증하고, 어텐션 런타임 및 메모리 벤치마크 결과를 보고한다.
![Flashattention : fast and memory-efficient exact attention with io-awareness](/assets/reviews/flashattention-2.png)
 학습 속도: FlashAttention은 BERT에 대한 MLPert 1.1 속도 기록을 15% 능가하며, GPT-2 속도를 HuggingFace 및 Megatron 대비 향상시키고, 표준 Transformer FlashAttention은 Long-Range Arena(LRA) 벤치마크의 속도를 2배 향상시킨다.
 품질: FlashAttention은 컨텍스트 길이가 4K인 GPT-2를 Megatron이 컨텍스트 길이 1K로 학습시키는 것보다 더 빠르게 학습시키면서도 0.7 더 나은 퍼플렉서티를 달성했다. 또한 긴 문서 분류 작업에서 시퀀스 길이를 늘려 성능을 6.4포인트 끌어올렸으며 기존의 Transformer 모델들이 메모리 문제로 실패하거나 무작위 수준의 성능을 보였던 난이도 높은 작업에서 최초로 성공적인 결과를 얻었다. 
 어텐션 벤치마킹: FlashAttention의 메모리 사용량이 확장되며 일반적인 시퀀스 길이(최대 2K)에서 표준 어텐션보다 최대 3배 더 효율적이다. 또한 블록 희소 FlashAttention의 런타임이 기존의 모든 근사 어텐션 베이스라인보다 빠르다.
####  4.1 FlashAttention을 통한 더 빠른 모델
![Flashattention : fast and memory-efficient exact attention with io-awareness](/assets/reviews/flashattention-3.png)
 BERT-large에서는 Nvidia의 기존 학습속도보다 15% 더 빠른 학습 시간을 달성하였다. 또한 GPT-2에서는 HuggingFace 구현체 대비 최대 3.5배, Megatron-LM 대비 1.7배 더 빠른 속도를 보였으며, LRA에서는 표준 어텐션대비 2.4배 속도 향상을 기록하였다.
![Flashattention : fast and memory-efficient exact attention with io-awareness](/assets/reviews/flashattention-4.png)
#### 4.2 더 긴 시퀀스를 통한 더 나은 모델
 FlashAttention의 런타임 및 메모리 효율성 덕분에 GPT-2의 컨텍스트 길이를 4배 늘리면서도 Megatron-LM의 최적화된 구현보다 더 빠르게 실행할 수 있다. 아래는 긴 문서 분류 작업에서의 성능 향상이다.
![Flashattention : fast and memory-efficient exact attention with io-awareness](/assets/reviews/flashattention-5.png)
![Flashattention : fast and memory-efficient exact attention with io-awareness](/assets/reviews/flashattention-6.png)
 왼쪽은 순방향 패스(forward pass) + 역방향 패스(backward pass) 의 런타임이며 오른쪽은 어텐션 메모리 사용량이다.
![Flashattention : fast and memory-efficient exact attention with io-awareness](/assets/reviews/flashattention-7.png)
 FlashAttention을 사용한 다양한 시퀀스 길이에 따른 긴 문서 성능이다
![Flashattention : fast and memory-efficient exact attention with io-awareness](/assets/reviews/flashattention-8.png)
 Path-X 및 Path-256에서 무작위 성능을 상회하는 결과를 달성한 지표이다.
#### 4.3 어텐션 벤치마킹
![Flashattention : fast and memory-efficient exact attention with io-awareness](/assets/reviews/flashattention-9.png)
 본 논문에서는 FlashAttention과 블록 희소 FlashAttention의 런타임 및 메모리 사용량을 다양한 어텐션 베이스라인과 비교 및 측정한다. 본문에서는 베이스라인 일부만을 보고하며 부록 E에서 세부사항을 확인할 수 있다.
## 한계점 및 향후 연구 방향
 본 논문에서의 한계점은 IO-aware 어텐션 구현을 구축하기 위해선 새로운 어텐션 구현마다 새로운 CUDA 커널을 작성해야 한다는 것이다. 이는 PyTorch보다 상당히 낮은 수준의 언어로 어텐션 알고리즘을 작성해야 하며, 상당한 엔지니어링 노력이 필요하다. 또한 구현이 GPU 아키텍처 간에 호환되지 않을 수 있다. 이러한 한계점들은 이미지 처리 분야의 Halide와 같은 노력과 유사하게, 고수준 언어로 어텐션 알고리즘을 작성하고 이를 CUDA의 IO-aware 구현으로 컴파일하는 것을 지원하는 방법의 필요성을 시사한다.
본 논문에서는 앞으로 IO-aware 접근 방식이 어텐션을 넘어 확장될 수 있다고 믿는다. 어텐션은 Transformer에서 가장 메모리 집약적인 연산이지만, 딥 네트워크의 모든 레이어는 GPU HBM에 접근한다. 따라서 이러한 연구가 추가 모듈의 IO-aware 구현에 영감을 주기 바란다. 또한 이러한 IO-aware 구현은 단일 GPU에서 어텐션을 계산하는데 있어 물리적 범위 내에서 최적이다. 하지만 어텐션 연산은 여러 GPU에 걸쳐 병렬화될 수 있다. 여러 GPU를 사용하면 GPU 간 데이터 전송을 고려해야 하므로 IO 분석에 추가적인 계층이 더해진다. 이 방향으로의 향후 연구에 본 논문의 연구가 영감을 주길 바란다.
{% endraw %}
