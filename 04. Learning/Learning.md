# 4. Nueral Net Learning 

* 학습 : 훈련 데이터로부터 가중치 매개 변수의 최적값을 자동으로 획득하는 것
* 손실 함수 : 신경망이 학습할 수 있도록 해주는 지표

- 학습의 목표 : 손실 함수의 결과값을 가장 작게 만드는 가중치 매개 변수를 찾는 것
- 경사법 : 함수의 기울기를 활용하여 손실 함수의 값을 가급적 작게 만드는 기법

## 4.1 데이터에서 학습한다!

* 데이터에서 학습한다 : 가중치 매개 변수의 값을 데이터를 보고 자동으로 결정한다는 뜻
* 이번 장
    1. 신경망 학습에 대한 설명 : 데이터부터 매개 변수의 값을 정하는 방법
    2. 파이썬으로 MNIST 데이터셋의 손글씨 숫자를 학습하는 코드 구현

### 4.1.1 데이터 주도 학습

* 데이터 : 기계 학습의 중심 - 데이터에서 답을 찾고 패턴을 발견하고 이야기를 만듦
* 기계 학습 : 사람의 개입을 최소화하고 수집한 데이터로부터 패턴을 찾으려고 시도
* 신경망과 딥러닝 : 기존 기계 학습보다 더 사람의 개입을 배제할 수 있는 특성을 가짐

- 구체적인 문제 : 손글씨 숫자 '5' 인식
    1. 사람이 직접 알고리즘 고안 : 특징짓는 규칙 찾기 어려움
    2. 기존 기계 학습 : (SIFT 등의) 특징을 추출하고 특징의 패턴을 (SVM 같은) 기계 학습 기술로 학습 - 특징은 사람이 설계
    3. 딥러닝 : 신경망으로 이미지를 있는 그대로 학습 - 특징도 기계가 스스로 학습

* 신경망의 이점 : 모든 문제를 같은 방식으로 해결 - 주어진 입력 데이터로 '종단간 (end-to-end)' 학습

### 4.1.2 훈련 데이터와 시험 데이터

* 기계 학습 문제 : 데이터를 '훈련 (training) 데이터' 와 '시험 (test) 데이터' 로 나누어서 학습과 실험 수행
* 훈련 데이터만으로 학습하여 최적의 매개 변수를 찾은 후, 시험 데이터를 사용하여 훈련한 모델의 실력을 평가

- 범용 능력 : 아직 보지 못한 데이터로도 문제를 올바르게 풀어내는 능력 - 기계 학습의 최종 목표
- 범용 능력을 평가하기 위해 '훈련 데이터' 와 '시험 데이터' 를 분리함

* 데이터셋 하나로만 매개 변수의 학습과 평가를 수행하면 올바른 평가가 될 수 없음
* 오버 피팅 (overfitting) : 한 데이터셋에만 지나치게 최적화된 상태

## 4.2 손실 함수

* 신경망 학습은 현재 상태를 하나의 지표로 표현하여, 그 지표를 가장 좋게 만들어주는 가중치 매개 변수 값을 탐색함
* 손실 함수 (loss function) : 신경망 학습에서 사용하는 지표 (비용 함수 (cost function) 라고도 함) 
* 보통 손실 함수로 '평균 제곱 오차' 와 '교차 엔트로피 오차' 를 사용함

### 4.2.1 평균 제곱 오차

* 평균 제곱 오차 (MSE; Mean Squared Error) : 가장 많이 쓰이는 손실 함수

><img src="https://latex.codecogs.com/gif.latex?E=\frac{1}{2}\sum_k&space;(y_k&space;-&space;t_k)^2" title="E=\frac{1}{2}\sum_k (y_k - t_k)^2" />
>
> * <img src="https://latex.codecogs.com/gif.latex?y_k" title="y_k" /> : 신경망의 출력
> * <img src="https://latex.codecogs.com/gif.latex?t_k" title="t_k" /> : 정답 레이블
> * <img src="https://latex.codecogs.com/gif.latex?k" title="k" /> : 데이터의 차원 수
 
```python
y = [0.1, 0.05, 0.6, 0.0, 0.05, 0.1, 0.0, 0.1, 0.0, 0.0]
t = [0, 0, 1, 0, 0, 0, 0, 0, 0, 0]
```

* 신경망 출력 'y' 는 소프트맥스 함수의 출력 : 확률로 해석 가능
* 원-핫 인코딩 : 정답 레이블 't' 에서 한 원소만 '1' 이고 그 외는 '0' 으로 나타내는 표기법 

- 평균 제곱 오차 : 각 원소의 출력과 정답의 차 `(y_k - t_k)` 를 제곱한 후, 총합을 구함
- 평균 제곱 오차의 구현

```python
def mean_squared_error(y, t):
    return 0.5 * np.sum((y-t)**2)
```

* 함수의 실제 사용 예

```python
t = [0, 0, 1, 0, 0, 0, 0, 0, 0, 0]

y = [0.1, 0.05, 0.6, 0.0, 0.05, 0.1, 0.0, 0.1, 0.0, 0.0]
mean_squared_error(np.array(y), np.array(t))

y = [0.1, 0.05, 0.1, 0.0, 0.05, 0.1, 0.0, 0.6, 0.0, 0.0]
mean_squared_error(np.array(y), np.array(t))

$ python LossFunction.py
0.09750000000000003
0.5975
```

* 손실 함수 출력은 첫 번째가 더 작으며 정답 레이블과의 오차도 작음
* 평균 제곱 오차를 기준으로는 첫 번째 추정 결과가 정답에 더 가까운 것으로 판단함

### 4.2.2. 교차 엔트로피 오차 

* 교차 엔트로피 오차 (CEE; Cross Entropy Error)

> <img src="https://latex.codecogs.com/gif.latex?E=-\sum_k&space;t_k&space;\log&space;{y_k}" title="E=-\sum_k t_k \log {y_k}" />
>
> * <img src="https://latex.codecogs.com/gif.latex?y_k" title="y_k" /> : 신경망의 출력
> * <img src="https://latex.codecogs.com/gif.latex?t_k" title="t_k" /> : 정답 레이블

* <img src="https://latex.codecogs.com/gif.latex?t_k" title="t_k" /> 는 정답에 해당하는 원소만 '1' 이고 나머지는 '0' 임 (원-핫 인코딩)
* 즉, 실질적으로 위 식은 정답일 때의 <img src="https://latex.codecogs.com/gif.latex?y_k" title="y_k" /> 의 자연 로그를 계산하는 식이 됨
* 교차 엔트로피 오차는 정답일 때의 출력이 전체 값을 정함
* 교차 엔트로피의 구현

```python
def cross_entropy_error(y, t:
    delta = 1e-7
    return -np.sum(t * np.log(y+delta))
```

* 아주 작은 값을 더해서 입력이 '0' 이 되지 않도록 함

- 함수의 사용 예

```python
t = [0, 0, 1, 0, 0, 0, 0, 0, 0, 0]

y = [0.1, 0.05, 0.6, 0.0, 0.05, 0.1, 0.0, 0.1, 0.0, 0.0]

print(cross_entropy_error(np.array(y), np.array(t)))

y = [0.1, 0.05, 0.1, 0.0, 0.05, 0.1, 0.0, 0.6, 0.0, 0.0]

print(cross_entropy_error(np.array(y), np.array(t)))

$ python LossFunction.py
0.510825457099338
2.302584092994546
```

* 값은 다르지만, 첫 번째 추정이 정답을 가능성이 높다는, 판단은 평균 제곱 오차과 일치함

### 4.2.3 미니 배치 학습

* 기계 학습을 하려면, 모든 훈련 데이터를 대상으로 '손실 함수 값' 을 구해야 함
* 모든 훈련 데이터에 대한 손실 함수의 합을 구하는 방법

> <img src="https://latex.codecogs.com/gif.latex?E=-\frac{1}{N}&space;\sum_n&space;\sum_k&space;t_{nk}&space;\log&space;{y_{nk}}" title="E=-\frac{1}{N} \sum_n \sum_k t_{nk} \log {y_{nk}}" />
>
> * <img src="https://latex.codecogs.com/gif.latex?N" title="N" /> : 데이터의 개수
> * <img src="https://latex.codecogs.com/gif.latex?t_{nk}" title="t_{nk}" /> : n 번째 데이터의 k 차원 째의 값

* 수식이 복잡해 보이지만, 데이터 하나에 대한 손실 함수를 단순히 N 개의 데이터로 확장한 것
* 'N' 으로 나누어 정규화 : 'N' 으로 나눔으로써 '평균 손실 함수' 를 구함

- 많은 데이터를 대상으로 일일이 손실 함수를 계산하는 것은 비현실적임 : 데이터를 추려서 '근사치' 로 사용
- 미니 배치 (mini-batch) : 신경망 학습에서 학습할 훈련 데이터로 고른 일부의 데이터
- 미니 배치 학습 : 무작위로 뽑은 '미니 배치' 를 사용하여 학습하는 것

* 미니 배치 학습 구현

- MNIST 데이터셋을 읽어오는 코드

```python
import sys, os
sys.path.append(os.pardir)

import numpy as np 
from dataset.mnist import load_mnist 

(x_train, t_train), (x_test, t_test) = \
    load_mnist(normalize=True, one_hot_label=True)

print(x_train.shape)    # (60000, 784)
print(t_train.shape)    # (60000, 10)
```

* 이 훈련 데이터에서 무작위로 10 장을 선택하는 코드

```python
train_size = x_train.shape[0]
batch_size = 10 
batch_mask = np.random.choice(train_size, batch_size)

x_batch = x_train[batch_mask]
t_batch = t_train[batch_mask]
```

* `np.random.choice()` 는 지정한 범위의 수에서 무작위로 원하는 개수만 선택함
* 위 함수가 출력한 배열을 색인으로 하여 '미니 배치' 를 뽑아내면 됨

### 4.2.4 (미니 배치로) 교차 엔트로피 오차 구현하기

* 미니 배치 같은 '배치 데이터' 를 처리할 수 있는 '교차 엔트로피 오차' 구현
* 데이터가 하나인 경우와 배치로 묶여 입력되는 경우 모두를 처리할 수 있도록 구현

```python
def cross_entropy_error(y, t):
    if y.ndim == 1:
        t = t.reshape(1, t.size)
        y = y.reshape(1, y.size)

    batch_size = y.shape[0]
    return -np.sum(t * np.log(y)) / batch_size
```

* 정답 레이블이 '원-핫 인코딩' 이 아니라 '2', '7' 등의 '숫자 레이블' 로 주어졌을 때의 교차 엔트로피 구현

```python
def cross_entropy_error_with_label(y, t):
    if y.ndim == 1:
        t = t.reshape(1, t.size)
        y = y.reshape(1, y.size)

    batch_size = y.shape[0]
    return -np.sum(np.log(y[np.arrange(batch_size), t])) / batch_size   # 정확하게는 잘 모르겠음
```

* '원-핫 인코딩' 일 때, 't' 가 '0' 인 원소는 '교차 엔트로피 오차' 도 '0' 이므로 계산을 무시할 수 있음
* 그래서 `t * np.log(y)` 를 `np.log(y[np.arrange(batch_size), t])` 로 바꿔줌 : `y[0, 2]`, `y[1, 7]`, `...` 등만 계산함

### 4.2.5 왜 '손실 함수' 를 설정하는가?

* 신경망 학습에서는 최적의 매개 변수를 탐색할 때, 손실 함수의 값을 가능한 작게 하는 매개 변수를 찾음
* 이 때, 매개 변수의 미분을 계산하고, 그 미분 값을 단서로 매개 변수의 값을 갱신하는 것을 반복

- '손실 함수' 는 '미분 가능' 이 핵심 : 그래서 '정확도' 같은 것은 사용할 수 없음

## 4.3 수치 미분

* 경사법 : 기울기 값을 기준으로 갱신함
* 미분 복습

### 4.3.1 미분

* 미분 : 한순간의 변화량

<img src="https://latex.codecogs.com/gif.latex?\frac{df(x)}{dx}&space;=&space;\lim_{h&space;\rightarrow&space;0}&space;\frac{f(x&plus;h)-f(x)}{h}" title="\frac{df(x)}{dx} = \lim_{h \rightarrow 0} \frac{f(x+h)-f(x)}{h}" />

* 함수 미분을 파이썬으로 구현 : 안좋은 구현

```python
def numerical_diff(f, x):
    h = 10e-50
    return (f(x+h) - f(x)) / h
```

* 개선점 : 2 가지
    1. 반올림 오차 (rounding error) 문제 : `10e-50` 같은 너무 작은 값 이용 - `1e-4` 정도가 좋다고 알려짐
    2. 함수 f 의 차분 : 오차를 줄이기 위해 '중심 차분', 또는 '중앙 차분' 이용

- 개선점을 적용한 수치 미분 구현

```python
def numerical_diff(f, x):
    h = 1e-4    # 0.0001
    return (f(x+h) - f(x-h)) / (2*h)
```

> '수치 미분' 외에 '해석적 해' 를 구할 수도 있으나 항상되는 것은 아님

### 4.3.2 수치 미분의 예

* 간단한 '2차 함수' 의 미분

<img src="https://latex.codecogs.com/gif.latex?y=0.01x^2&space;&plus;&space;0.1x" title="y=0.01x^2 + 0.1x" />

* 파이썬으로 수식 구현

```python
def function_1(x):
    return 0.01*x**2 + 0.1*x
```

* 함수 그리기 : 'macOS' 에서는 안됨

```python
import numpy as np
import matplotlib.pylab as plt 

x = np.arange(0.0, 20.0, 0.1)
y = function_1(x)

plt.xlabel("x")
plt.ylabel("f(x)")
plt.plot(x, y)
plt.show()
```

* 'x' 가 '5', '10' 일 때의 미분 계산

```python
print(numerical_diff(function_1, 5))    # 0.1999999999990898
print(numerical_diff(function_1, 10))   # 0.2999999999986347
```

* 해석적 해는 각각 '0.2', '0.3' 인데, 오차가 매우 작음을 알 수 있음

### 4.3.3 편미분

* '2 변수 함수' 의 미분

<img src="https://latex.codecogs.com/gif.latex?f(x_0,&space;x_1)&space;=&space;x_0^2&space;&plus;&space;x_1^2" title="f(x_0, x_1) = x_0^2 + x_1^2" />

* 파이썬으로 수식 구현

```python
def function_2(x):
    return x[0]**2 + x[1]**2
```

* 변수가 2 개 이므로, 미분을 구할 때, '어느 변수에 대한 미분' 인지를 구별해야 함
* 편미분 : 변수가 여러 개인 함수에 대한 미분

- 문제 1 : <img src="https://latex.codecogs.com/gif.latex?x_0" title="x_0" /> = 3, <img src="https://latex.codecogs.com/gif.latex?x_1" title="x_1" /> = 4 일 때, <img src="https://latex.codecogs.com/gif.latex?x_0" title="x_0" /> 에 대한 편미분 <img src="https://latex.codecogs.com/gif.latex?\frac{\partial&space;f}{\partial&space;x_0}" title="\frac{\partial f}{\partial x_0}" /> 구하기

```python
def function_tmp1(x0):
    return x0*x0 + 4.0**2.0

print(numerical_diff(function_tmp1, 3.0))   # 6.00000000000378
```

- 문제 2 : <img src="https://latex.codecogs.com/gif.latex?x_0" title="x_0" /> = 3, <img src="https://latex.codecogs.com/gif.latex?x_1" title="x_1" /> = 4 일 때, <img src="https://latex.codecogs.com/gif.latex?x_1" title="x_1" /> 에 대한 편미분 <img src="https://latex.codecogs.com/gif.latex?\frac{\partial&space;f}{\partial&space;x_1}" title="\frac{\partial f}{\partial x_1}" /> 구하기

```python
def function_tmp2(x1):
    return 3.0**2.0 + x1**x1

print(numerical_diff(function_tmp2, 4.0))   # 7.999999999999119
```

* 문제들을 '1 변수 함수' 로 정의하여, 그 함수에 대한 미분을 구하는 형태로 구현함

- 편미분은 '1 변수 함수' 와 마찬가지로 특정 장소의 기울기를 구함
- 단, 여러 변수 중 목표 변수 하나에 초점을 맞추고 다른 변수는 값을 고정함
- 목표 변수를 제외한 나머지를 특정 값에 고정하는 새로운 함수를 정의함
- 새로 정의한 함수에 '수치 미분 함수' 를 적용하여 편미분을 구함

## 4.4 기울기

* <img src="https://latex.codecogs.com/gif.latex?x_0" title="x_0" /> 과 <img src="https://latex.codecogs.com/gif.latex?x_1" title="x_1" /> 의 편미분을 동시에 계산하고 싶음
* 기울기 (gradient) : <img src="https://latex.codecogs.com/svg.latex?\left&space;(&space;\frac{\partial&space;f}{\partial&space;x_0},&space;\frac{\partial&space;f}{\partial&space;x_1}&space;\right&space;)" title="\left ( \frac{\partial f}{\partial x_0}, \frac{\partial f}{\partial x_1} \right )" /> 처럼 모든 변수의 편미분을 벡터로 정리한 것

```python
# code
```

* 기울기가 가리키는 쪽은 각 장소에서 함수의 출력 값을 가장 줄이는 방향

### 4.4.1 경사법 / 경사 하강법

* 기계 학습 : 학습 단계에서, 손실 함수가 최소가 되는, 최적의 매개 변수를 찾음, 일반적인 경우, 손실 함수는 매우 복잡함
* 경사법 : 기울기를 활용하여 함수의 최소값을 찾으려는 것, 기울기가 가리키는 곳이 함수의 최소값이라는 보장은 없음
* 기울어진 방향이 꼭 최소값을 가리키지는 않으나, 그 방향으로 가야 함수의 값을 줄일 수 있음, 기울기 정보를 단서로 나갈 방향을 정함

- 경사법 (Gradient method)
    1. 현 위치에서 기울어진 방향으로 일정 거리만큼 이동
    2. 이동한 곳에서도 기울기를 구함
    3. 그 방향으로 일정 거리만큼 이동하는 일을 반복

* 경사 하강법 (gradient descent method) == 경사 상승법 (gradient ascent method)

- 학습률 (learning rate) : 한 번의 학습으로 얼만큼 학습해야 할지, 즉 매개 변수 값을 얼마나 갱신할지를 정하는 것

* 학습률 값은 '0.01' 이나 '0.001' 등으로 미리 정해둠 
* 이 값이 너무 크거나 작으면 '좋은 장소' 를 찾아갈 수 없음






