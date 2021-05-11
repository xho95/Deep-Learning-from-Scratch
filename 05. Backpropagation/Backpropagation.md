# 5. 오차 역전파법 

* 오차 역전파법 (backpropagation) : 가중치 매개 변수의 기울기를 효율적으로 계산

- 오차 역전파법을 이해하는 방법
    1. 수식을 통한 것
    2. 계산 그래프로 이해하는 것 : 안드레 카패시 (Andreij Karpathy), 페이페이 리 (Fei-Fei Li) - Stanford cs231n

## 5.1 계산 그래프

* 계산 그래프 (computational graph) : 계산 과정을 그래프로 나타낸 것

- 그래프 자료 구조 : 노드 (node) 와 에지 (edge) 로 표현

### 5.1.1 계산 그래프로 풀다

* 계산 그래프는 계산 과정을 '노드' 와 '화살표 (에지)' 로 표현
    1. 노드 : 연산 내용을 담은 원으로 표기
    2. 에지 : 계산 결과를 에지 위에 표기

- [그림 5-1]
- [그림 5-2]
- [그림 5-3]

* 계산 그래프를 이용한 문제 풀이
    1. 계산 그래프를 구성
    2. 그래프 왼쪽에서 오른쪽으로 계산을 진행함 : 순전파 (forward propagation)

- 역전파 (backpropagation) : 계산을 오른쪽에서 왼쪽으로 진행 - 미분 계산

### 5.1.2 국소 계산

* 계산 그래프의 특징 : '국소적 계산' 을 전파함으로써 최종 결과를 얻음
* 국소 계산 : 자신과 직접 관계된 작은 범위 - 전체와는 상관없이 자신과 관계된 정보만으로 다음 결과를 출력할 수 있음

- [그림 5-4]

* 각 노드는 자신과 관련한 계산 외에는 신경 쓸 것이 없음

### 5.1.3 왜 계산 그래프로 푸는가?

* 계산 그래프의 이점
    1. 국소 계산
    2. 중간 계산 결과를 저장할 수 있음
    3. '역전파' 를 통하여 '미분' 을 효율적으로 계산할 수 있음 - 가장 큰 이유

- [그림 5-5]

* 역전파는 순전파와는 반대 방향의 '굵은 에지 (화살표)' 로 그림
* 예제에서 역전파는 오른쪽에서 왼쪽으로 '미분 값' 을 전달
* 미분을 구할 때 저장했던 '중간 미분 결과' 를 활용 가능 

## 5.2 연쇄 법칙

* 국소적 미분을 전달하는 원리 : '연쇄 법칙 (chain rule)' 을 따름
* '연쇄 법칙' 은 결국 '계산 그래프 상의 역전파' 와 똑같음

### 5.2.1 계산 그래프의 역전파

* `y = f(x)` 역전파

- [그림 5-6]

* 역전파의 계산 절차 : '신호 `E`' 에 노드의 국소 미분 `∂y/∂x` 를 곱한 후 다음 노드로 전달
* 국소 미분 : 순전파 `y = f(x)` 계산의 미분
* 이 방식을 사용하면 '미분 값' 을 효율적으로 구할 수 있음

### 5.2.2 연쇄법칙이란?

* 합성 함수 : 여러 함수로 구성된 함수

- [식 5-1]

* 연쇄 법칙 : 함성 함수의 미분은 합성 함수를 구성하는 각 함수의 미분의 곱으로 나타낼 수 있다 - 참고 자료 : [합성 함수의 미분법](https://m.blog.naver.com/PostView.nhn?blogId=at3650&logNo=40138304205&proxyReferer=https:%2F%2Fwww.google.com%2F)

- [식 5-4] <img src="https://latex.codecogs.com/svg.latex?\frac{\partial&space;z}{\partial&space;x}=\frac{\partial&space;z}{\partial&space;t}\frac{\partial&space;t}{\partial&space;x}" title="\frac{\partial&space;z}{\partial&space;x}=\frac{\partial&space;z}{\partial&space;t}\frac{\partial&space;t}{\partial&space;x}" />

### 5.2.3 연쇄 법칙과 계산 그래프 

* [그림 5-6]

* [사진 1]

### 5.3 역전파 

* 앞에서 '계산 그래프의 역전파' 가 '연쇄 법칙' 에 따라 진행함을 보임
* 역전파의 구조를 설명함

### 5.3.1 덧셈 노드의 역전파 

* `z = x + y`

- [식 5-5]
- <img src="https://latex.codecogs.com/svg.latex?\frac{\partial&space;z}{\partial&space;x}=1" title="\frac{\partial&space;z}{\partial&space;x}=1" />
- <img src="https://latex.codecogs.com/svg.latex?\frac{\partial&space;z}{\partial&space;y}=1" title="\frac{\partial&space;z}{\partial&space;y}=1" />

* [그림 5-9]

- 덧셈 노드의 역전파는 입력된 값을 그대로 다음 노드로 보냄

### 5.3.2 곱셈 노드의 역전파 

* `z = xy`

- [식 5-6]
- <img src="https://latex.codecogs.com/svg.latex?\frac{\partial&space;z}{\partial&space;x}=y" title="\frac{\partial&space;z}{\partial&space;x}=y" />
- <img src="https://latex.codecogs.com/svg.latex?\frac{\partial&space;z}{\partial&space;y}=x" title="\frac{\partial&space;z}{\partial&space;y}=x" />

* [그림 5-12]

- 곱셈 노드의 역전파는 입력된 값에 순전파 입력 신호들을 '서로 바꾼 값' 을 곱하여 다음 노드로 보냄
- 곱셈의 역전파에는 순방향 입력 신호의 값이 필요함 : 곱셈 노드 구현시에는 순전파의 입력 신호를 저장함

### 5.3.3 사과 쇼핑의 예

* [사진 2]

## 5.4. 단순한 계층 구현하기

* 곱셈 노드 : MulLayer, 덧셈 노드 : AddLayer

### 5.4.1 곱셈 계층 (Multiply Layer)

* 모든 계층은 `forward()` 와 `backward()` 를 가지도록 구현함

- 곱셈 계층의 구현

* 코드는 `LayerNaive.py` 참고 : 설명은 코드 참고

- [그림 5-16] : `BuyApple.py` 코드 살펴보고, 실행
- 결과는 [그림 5-16] 과 일치

### 5.4.2 덧셈 계층 (Add Layer)

- 덧셈 계층의 구현

* 코드는 `LayerNaive.py` 참고 : 설명은 코드 참고

- [그림 5-17] : `BuyAppleOrange.py` 코드 살펴보고, 실행
- 결과는 [그림 5-17] 과 일치

## 5.5 활성화 함수 계층 구현하기 (Implementing an Activation Function Layer)

* 신경망을 구성하는 계층을 각각의 클래스로 구현
* 우선 `ReLu` 와 `Sigmoid` 구현

### 5.5.1 ReLU Layer

* ReLU 식

> <img src="https://latex.codecogs.com/svg.latex?y=\begin{cases}0\left(&space;x>0\right)&space;\\&space;1\left(&space;x\leq&space;0\right)&space;\end{cases}" title="y=\begin{cases}0\left(&space;x>0\right)&space;\\&space;1\left(&space;x\leq 0\right)&space;\end{cases}" />
>
> 식 5-7

* x 에 대한 y 미분 

> <img src="https://latex.codecogs.com/svg.latex?\frac{\partial&space;y}{\partial&space;x}=\begin{cases}0\left(&space;x>0\right)&space;\\&space;1\left(&space;x\leq&space;0\right)&space;\end{cases}" title="\frac{\partial&space;y}{\partial&space;x}=\begin{cases}0\left(&space;x>0\right)&space;\\&space;1\left(&space;x\leq 0\right)&space;\end{cases}" />
>
> 식 5-8

- [그림 5-18]

* ReLU 구현 : `Layers.py`

- ReLU 실습 : `Mask.py` - 역전파 시 `mask` 가 `True` 인 원소를 `0` 으로 설정

### 5.5.2 Sigmoid Layer 

* Sigmoid 함수 식

> <img src="https://latex.codecogs.com/gif.latex?y=\frac{1}{1&plus;\exp(-x)&space;}" title="y=\frac{1}{1+\exp(-x) }" />
> 
> 식 5-9

- [그림 5-19]

* `x` 와 `+` 노드 외에도 `exp` 노드와 `/` 노드가 새롭게 등장함

**1 단계**

* `/` 노드를 미분하면 `dy/dx` 는 `-y^2` 이 됨 

- 참고 자료 : [다항 함수 미분](https://bhsmath.tistory.com/177), [Derivatives of polynomials](https://mathinsight.org/derivatives_polynomials_refresher)

**2 단계**

* `+` 노드를 미분하면 `1`

**3 단계**

* `exp` 노드를 미분하면 `exp` 가 됨 (그 자신)

**4 단계**

* `x` 노드를 미분하면 순전파 시의 값을 서로 바꿔 곱하는 것이 됨 : `-1` 을 곱함

- [그림 5-21]

* 결국, Sigmoid 계층의 역전파는 순전파의 출력인 `y` 만으로 계산할 수 있음

<img src="https://latex.codecogs.com/gif.latex?\frac{\partial&space;L}{\partial&space;x}=\frac{\partial&space;L}{\partial&space;y}y^2\exp(-x)=\frac{\partial&space;L}{\partial&space;y}y(1-y)" title="\frac{\partial&space;L}{\partial&space;x}=\frac{\partial&space;L}{\partial&space;y}y^2\exp(-x)=\frac{\partial&space;L}{\partial&space;y}y(1-y)" />

* Sigmoid 계층 구현 : `Layers.py` - 순전파 출력을 `out` 에 저장하여 역전파 때 사용함

## 5.6 Affine 게층 구현하기 (Implementing a Softmax (Affine) Layer)

### 5.6.1 Affine 계층

* 신경망의 순전파에서는 가중치 신호의 총합을 계산하기 때문에 행렬의 내적을 사용함
* `Dot.py` 실행

- Affine 계층 : 신경망의 순전파에서 수행하는 행렬의 내적은 기하학의 '어파인 (Affine) 변환' 에 해당
- [사진 3]

* 행렬을 사용한 역전파도 행렬 원소마다 전해가면 '스칼라 값' 을 사용한 계산 그래프와 같음

- [식 5-13]

* 행렬 미분 : 전치 행렬 사용 - 참고 자료 : [Matrix calculus](https://en.wikipedia.org/wiki/Matrix_calculus)
* 행렬 미분에서는 '계산 그래프 각 변수' 의 '형상 (shape)' 에 주의해야 함 - 행렬 미분 : 원래의 행렬과 형상이 같음

- [사진 4]

### 5.6.2 배치용 Affine 계층

- [사진 4] 활용

* 기존과 다른 부분 : 입력 X 의 형상이 '(N, 2)' 임
* 편향 계산의 주의 사항 : 순전파 때 편향 덧셈은 `X W` 에 대한 편향이 각 데이터에 더해짐
* `Batch.py` 실행

- 따라서 편향의 역전파는 모든 데이터에 대한 미분을 데이터마다 더해서 구함
- 그래서 `np.sum()` 에서 (데이터를 단위로 한 축인) '0 번째 축 (`axis=0`)' 에 대해서 총합을 구함

* Affine 구현 : `Layers.py`

### 5.6.3 Softmax_with_Loss Layer 

> 이 Softmax 층이 4 장에서 말한 출력에 사용하는 Softmax 층임

* [그림 5-29] : Softmax 계층은 `log` 를 제외하면 이전에 구한 것들의 조함임

- [그림 5-30] : 역전파의 결과가 `(y1 - t1, y2 - t2, y3 - t3)` 라는 간단한 형태임
- `(y1 - t1, y2 - t2, y3 - t3)` 라는 결과는 Softmax 계층의 출력과 정답 레이블의 차이임
- 이는 신경망의 현재 출력과 정답 레이블의 오차를 그대로 드러냄

> 'Softmax 함수' 와 '교체 엔트로피 오차' 를 사용하면 결과가 간단한 형태로 나옴
>
> 마찬가지로, '항등 함수' 와 '평균 제곱 오차' 를 사용하면 결과가 간단한 형태로 나옴

* Softmax-with-Loss 계층 구현 : `Layers.py` - 책과 GitHub 코드가 다름!!

## 5.7 오차 역전파법 구현하기

* 지금까지 구현한 계층을 조합하면 신경망을 구축할 수 있음

### 5.7.1 신경망 학습의 전체 그림

* 신경망 학습 : '신경망' 에는 적응 가능한 가중치와 편향이 있고, 훈련 데이터에 적응하도록 이를 조정하는 과정을 '학습' 이라 함

- 신경망 학습 순서
    1. 미니 배치 : 훈련 데이터 중 일부를 무작위로 가져옴. 이렇게 선별한 데이터를 '미니 배치' 라 하며, 목표는 '미니 배치' 의 손실 함수 값을 줄이는 것
    2. 기울기 산출 : 미니 배치의 '손실 함수' 값을 줄이기 위해 각 가중치 매개 변수의 기울기를 구함. 기울기는 손실 함수의 값을 가장 작게하는 방향임
    3. 매개 변수 갱신 : 가중치 매개 변수를 기울기 방향으로 아주 조금 갱신함
    4. 반복 : 1~3 을 반복함

* 오차 역전파법은 두 번째 단계인 '기울기 산출' 에서 등장
* 앞 장에서 사용한 '수치 미분' 은 구현은 쉽지만 계산이 오래 걸림
* '오차 역전파법' 을 사용하면 기울기를 효율적으로 빠르게 구할 수 있음

### 5.7.1 오차 역전파법을 적용한 신경망 구현하기

* 구현 : `TwoLayerNet2.py` - 책의 코드에 오류 있음

- 신경망 계층을 `OrderedDict` 에 저장함
    1. 순전파 시에는 추가한 계층 순서대로 `forward()` 호출
    2. 역전파 시에는 계층을 반대 순서로 호출

* 신경망 구성 요소를 계층으로 구현했기 때문에 쉽게 구현 가능
* 더 깊은 신경망을 만들고 싶으면, 계층을 더 추가하기만 하면 됨

### 5.7.3 오차 역전파법으로 구한 기울기 검증하기

* 수치 미분의 결과와 오차 역전파법의 결과를 비교하여 오차 역전파법을 제대로 구현했는지 검증함
* `GradientCheck.py` 실행

- 각 가중치 매개 변수 차이의 절대값을 구한 다음, 이를 평균한 오차를 구함
- 실행 결과가 '0' 에 가까우면 구현이 잘된 것임

### 5.7.4 오차 역전파법을 사용한 학습 구현하기

* `TrainNueralNet.py` 실행

- 결과가 '1' 에 가까워지면 학습이 잘되고 있는 것임

## 5.8 Summary

