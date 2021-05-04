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

* 파이썬 구현

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

* 우선, 변수가 하나인 함수를 정의한 후, 그 함수에 대한 미분을 구하는 형태로 구현함
* '문제 1' 의 경우, 'x1 = 4' 로 고정된 함수를 정의하고, 변수가 `x0' 하나 뿐인 함수에 대해 수치 미분 함수를 적용함
* 결과는 '해석적 미분' 과 거의 같음

- 편미분은 '1 변수 함수' 와 마찬가지로 기울기를 구하지만, 목표 변수에 초점을 맞추고 다른 변수의 값은 고정함
- 예제는, 목표 변수 외의 나머지를 특정 값으로 고정하기 위해 새 함수를 정의하고, '수치 미분 함수' 를 적용하여 편미분을 구함

## 4.4 기울기

* 앞의 예는 <img src="https://latex.codecogs.com/svg.latex?x_0" title="x_0" /> 와 <img src="https://latex.codecogs.com/svg.latex?x_1" title="x_1" /> 의 편미분을 변수별로 따로 계산함
* <img src="https://latex.codecogs.com/svg.latex?x_0" title="x_0" /> 와 <img src="https://latex.codecogs.com/svg.latex?x_1" title="x_1" /> 의 편미분을 동시에 계산하고 싶다면?
* 구불거림 (gradient) : <img src="https://latex.codecogs.com/svg.latex?\left&space;(&space;\frac{\partial&space;f}{\partial&space;x_0},&space;\frac{\partial&space;f}{\partial&space;x_1}&space;\right&space;)" title="\left ( \frac{\partial f}{\partial x_0}, \frac{\partial f}{\partial x_1} \right )" /> 처럼 모든 변수의 편미분을 벡터로 정리한 것
* '구불거림' 의 구현

```python
def numeric_gradient(f, x):
    h = 1e-4                    # 0.0001
    grad = np.zeros_like(x)     # grad has the same shape with x

    for index in range(x.size):
        tempValue = x[index]

        x[index] = tempValue + h        # f(x + h)
        fxh1 = f(x)

        x[index] = tempValue - h        # f(x - h)
        fxh2 = f(x)

        grad[index] = (fxh1 - fxh2) / (2 * h)
        x[index] = tempValue            # restore
    
    return grad       
```

* 구현은 복잡해 보이지만, 동작은 '1 변수' 일 때의 '수치 미분' 과 거의 같음
* numpy 배열 x 의 각 원소에 대한 수치 미분을 구함
* 아래는 세 점 (3, 4), (0, 2), (3, 0) 에서의 기울기를 구한 것임

```python
numeric_gradient(function_2, np.array([3.0, 4.0]))  # [6. 8.]
numeric_gradient(function_2, np.array([0.0, 2.0]))  # [0. 4.]
numeric_gradient(function_2, np.array([3.0, 0.0]))  # [6. 0.]
```

* [그림 4-9] : 기울기 결과에 마이너스를 붙인 벡터 그림
* 기울기는 함수의 '최소값' 을 가리키는 것처럼 보이지만, 반드시 그런 것은 아님
* 기울기는 각 장소에서 함수의 출력 값을 가장 줄이는 방향을 가리킴

### 4.4.1 경사법 / 경사 하강법

* 기계 학습 : 학습 단계에서, 손실 함수가 최소가 되는, 최적의 매개 변수를 찾아야 하나, 일반적으로 손실 함수는 매우 복잡함
* 경사법 : 기울기를 잘 이용하여 함수의 최소값을 찾으려는 방법, 기울기가 가리키는 곳이 함수의 최소값이라는 보장은 없음
* 기울어진 방향이 꼭 최소값을 가리키지는 않으나, 그 방향으로 가야 함수의 값을 줄일 수 있음, 기울기 정보를 단서로 나갈 방향을 정함

- 경사법 (Gradient method)
    1. 현 위치에서 기울어진 방향으로 일정 거리만큼 이동
    2. 이동한 곳에서도 기울기를 구함
    3. 그 방향으로 일정 거리만큼 이동하는 일을 반복

* 경사 하강법 (gradient descent method) vs 경사 상승법 (gradient ascent method) : 본질상 중요하진 않음

- 경사법 수식 : 학습률 (learning rate) - 갱신하는 양

<img src="https://latex.codecogs.com/svg.latex?{x_0&space;=&space;x_0&space;-&space;\eta&space;\frac{\partial&space;f}{\partial&space;x_0}}&space;\\&space;{x_1&space;=&space;xI&space;-&space;\eta&space;\frac{\partial&space;f}{\partial&space;x_1}}" title="{x_0 = x_0 - \eta \frac{\partial f}{\partial x_0}} \\ {x_1 = xI - \eta \frac{\partial f}{\partial x_1}}" />

* 학습률 : 한 번의 학습으로 얼만큼 학습해야 할지, 즉, 매개 변수 값을 얼마나 갱신할지 정하는 것
* 위 식은 1회에 해당하는 갱신이며, 이를 반복하여, 서서히 함수의 값을 줄임
* 학습률 값은 ('0.01', '0.001' 등) 미리 정해둠 : 값이 너무 크거나 작으면 '좋은 장소' 를 찾아갈 수 없음
* 신경망 학습 : 학습률 값을 변경하면서 올바르게 학습하고 있는지 확인하며 진행함

- 경사 하강법 구현

```python
def gradient_descent(f, init_x, lr=0.01, step_num=100):
    x = init_x 

    for i in range(step_num):
        grad = numeric_gradient(f, x)
        x -= lr * grad
    return x
```

* `f` : 최적화 하려는 함수
* `init_x` : 초기값
* `lr` : 학습률 (learning rate)
* `step_num` : 경사법 반복 횟수

- 함수의 기울기는 `numeric_gradient(f, x)` 으로 구하고, 이 기울기에 학습률을 곱한 값으로 갱신하는 처리를 `step_num` 번 반복함
- 이 함수를 사용하면 '함수의 극소값' 또는 '최소값' 을 구할 수 있음

* 경사법의 사용 예

```python
def function_2(x):
    return x[0]**2 + x[1]**2

init_x = np.array([-3.0, 4.0])
gradient_descent(function_2, init_x=init_x, lr=0.1, step_num=100)

# [-6.11110793e-10  8.14814391e-10]
```

* 초기값을 `(-3.0, 4.0)` 으로 설정한 후 경사법으로 최소값을 탐색함
* 최종 결과는 `(-6.1e-10, 8.1e-10)` 으로, 거의 `(0, 0)` 에 가까움
* 실제 최소값은 `(0, 0)` 으로, '경사법' 이 거의 정확한 결과를 구함

- [그림 4-10] : 경사법을 사용한 갱신 과정

* 학습률이 너무 크거나 작은 사용 예

```python
init_x = np.array([-3.0, 4.0])
gradient_descent(function_2, init_x=init_x, lr=10.0, step_num=100)

# [-2.58983747e+13 -1.29524862e+12]         too large lr

init_x = np.array([-3.0, 4.0])
gradient_descent(function_2, init_x=init_x, lr=1e-10, step_num=100)

# [-2.99999994  3.99999992]                 too small lr
```

* 학습률이 너무 크면 큰 값으로 '발산' 함
* 반대로 너무 작으면 거의 갱신되지 않은 채로 끝남

- 학습률 같은 매개 변수를 '하이퍼 매개 변수 (hyper parameter)' 라고 함 : 신경망 매개 변수와는 성질이 다름
- 신경망 매개 변수 (가중치) : 훈련 데이터와 학습 알고리즘으로 자동으로 획득되는 매개 변수
- 하이퍼 매개 변수 (학습률) : 사람이 직접 설정함 - 여러 후보 중에서 시험을 통해 최적값을 찾아야 함

### 4.4.2 신경망에서의 기울기

* 신경망 학습에서도 '기울기' 를 구해야 함 : 기울기 - 가중치 매개 변수에 대한 손실 함수의 기울기
* 가중치가 <img src="https://latex.codecogs.com/svg.latex?W" title="W" /> 고, 손실 함수가 <img src="https://latex.codecogs.com/svg.latex?L" title="L" /> 인, '2x3' 형상의 신경망에 대한 기울기 수식

<img src="https://latex.codecogs.com/svg.latex?W&space;=&space;\begin{bmatrix}&space;w_{11}&space;&&space;w_{21}&space;&&space;w_{31}&space;\\&space;w_{12}&space;&&space;w_{22}&space;&&space;w_{32}&space;\end{bmatrix}&space;\\&space;\frac{\partial&space;L}{\partial&space;W}=\begin{bmatrix}&space;\frac{\partial&space;L}{\partial&space;W_{11}}&space;&&space;\frac{\partial&space;L}{\partial&space;W_{21}}&space;&&space;\frac{\partial&space;L}{\partial&space;W_{31}}&space;\\&space;\frac{\partial&space;L}{\partial&space;W_{12}}&space;&&space;\frac{\partial&space;L}{\partial&space;W_{22}}&space;&&space;\frac{\partial&space;L}{\partial&space;W_{32}}&space;\end{bmatrix}" title="W = \begin{bmatrix} w_{11} & w_{21} & w_{31} \\ w_{12} & w_{22} & w_{32} \end{bmatrix} \\ \frac{\partial L}{\partial W}=\begin{bmatrix} \frac{\partial L}{\partial W_{11}} & \frac{\partial L}{\partial W_{21}} & \frac{\partial L}{\partial W_{31}} \\ \frac{\partial L}{\partial W_{12}} & \frac{\partial L}{\partial W_{22}} & \frac{\partial L}{\partial W_{32}} \end{bmatrix}" />

- 기울기 :  <img src="https://latex.codecogs.com/svg.latex?\frac{\partial&space;L}{\partial&space;W}" title="W" /> - <img src="https://latex.codecogs.com/svg.latex?\frac{\partial&space;L}{\partial&space;W}" title="W" /> 의 각 원소는 각각의 원소에 대한 편미분임
- 중요한 것은  <img src="https://latex.codecogs.com/svg.latex?\frac{\partial&space;L}{\partial&space;W}" title="W" /> 의 형상이 <img src="https://latex.codecogs.com/svg.latex?W" title="W" /> 와 같다는 것임

* 간단한 신경망의 기울기를 구하는 코드 구현

```python
import sys, os
sys.path.append(os.pardir)
sys.path.append("../03. Nueral Net")

from NueralNet import softmax
from MiniBatchLossFunction import cross_entropy_error
from NumericGradient import numeric_gradient 

import numpy as np

class simpleNet:
    def __init__(self):
        self.W = np.random.randn(2,3)       # normal distribution
    
    def predict(self, x):
        return np.dot(x, self.W)

    def loss(self, x, t):
        z = self.predict(x)
        y = softmax(z)
        loss = cross_entropy_error(y, t)

        return loss
```

* 기존에 정의한 `softmax`, `cross_entropy_error`, `numeric_gradient` 메소드를 이용함
* `simpleNet` 클래스
    1. '2x3' 형상의 가중치 매개 변수로 초기화됨
    2. `predict(x)` : 예측 수행, `loss(x, t)` : 손실 함수의 값 구함
    3. `x` : 입력 데이터, `t` : 정답 레이블

- `simpleNet` 실험 결과

```python
net = simpleNet()
print(net.W)
# [[-0.73546853 -0.90485202 -0.22409225]
#  [ 0.19518637  0.10183601  1.89255414]]

x = np.array([0.6, 0.9])
p = net.predict(x)
print(p)
# [-0.26561338 -0.4512588   1.56884337]

print(np.argmax(p))
# 2

t = np.array([0, 0, 1])
print(net.loss(x, t))
# 0.25645618872166714
```

* `numerical_gradient(f, x)` 를 사용하여 기울기를 구할 수 있음 

```python
def f(W):
    return net.loss(x, t)

dW = numerical_gradient(f, net.W)
print(dW)
# [[ 0.07414456  0.0615821  -0.13572666]
#  [ 0.11121684  0.09237315 -0.20359   ]]
```

* `numerical_gradient(f, x)`
    - 다차원 배열의 가중치 매개 변수 `W` 를 처리할 수 있도록 새로 만든 것임
    - `numerical_gradient(f, x)` 에서, `f` : 함수, `x` : 함수 `f` 의 인자
    - `new.W` 를 인자로 받아서 손실 함수를 계산하는 새로운 함수 `f(W)` 를 정의함
    - 이 새로운 함수를 `numerical_gradient(f, x)` 에 넘김

* `dW` 는 `numerical_gradient(f, x)` 의 결과 : '2x3' 형상의 2차원 배열 
* 결과를 보면 한 번에 갱신되는 양에는 'w32' 가 가장 크게 기여함을 할 수 있음 (절대값이 가장 큼)
* 파이썬은 간단한 함수를 다음처럼 'lambda' 로 구현할 수 있음

```python
f = lambda w: net.loss(x, t)
dW = numerical_gradient(f, net.W)
```

* 신경망의 기울기를 구한 다음에는 경사법에 따라 가중치 매개 변수를 갱신하면 됨
* 다음 절은 2층 신경망을 대상으로 학습 과정 전체를 구현함

## 4.5 학습 알고리즘 구현하기

* 신경망 학습 : '신경망' 에는 적응 가능한 가중치와 편향이 있고, 훈련 데이터에 적응하도록 이를 조정하는 과정을 '학습' 이라 함
* 신경망 학습 과정
    1. 미니 배치 : 훈련 데이터 중 일부를 무작위로 가져옴. 이렇게 선별한 데이터를 '미니 배치' 라 하며, 목표는 '미니 배치' 의 손실 함수 값을 줄이는 것
    2. 기울기 산출 : 미니 배치의 '손실 함수' 값을 줄이기 위해 각 가중치 매개 변수의 기울기를 구함. 기울기는 손실 함수의 값을 가장 작게하는 방향임
    3. 매개 변수 갱신 : 가중치 매개 변수를 기울기 방향으로 아주 조금 갱신함
    4. 반복 : 1~3 을 반복함

- 위는 '경사 하강법' 으로 매개 변수를 갱신하는 방법
- 확률적 경사 하강법 (SGD; stochastic gradient descent) : 확률적으로 무작위로 골라낸 '미니 배치' 에 대해 수행하는 경사 하강법

* 손글씨 숫자를 학습하는 신경망 구현 : 2층 신경망을 대상으로 MNIST 데이터셋을 사용하여 학습함

### 4.5.1 2층 신경망 클래스 구현하기

* 2층 신경망을 하나의 클래스로 구현

```python
import sys, os
sys.path.append(os.pardir)
sys.path.append("../03. Nueral Net")

from NueralNet import softmax
from MiniBatchLossFunction import cross_entropy_error
from NumericGradient import numerical_gradient 

import numpy as np

class TwoLayerNet: 
    def __init__(self, inputSize, hiddenSize, outputSize, weightInitStd = 0.01):
        # initialize the weights
        self.params = {}
        self.parmas['W1'] = weightInitStd * np.random.randn(inputSize, hiddenSize)
        self.params['b1'] = np.zeros(hiddenSize)
        self.params['W2'] = weightInitStd * np.random.randn(hiddenSize, outputSize)
        self.params['b2'] = np.zeros(outputSize)

    def predict(self, x):
        W1, W2 = self.params['W1'], self.params['W2']
        b1, b2 = self.params['b1'], self.params['b2']

        a1 = np.dot(x, W1) + b1
        z1 = sigmoid(a1)
        a2 = np.dot(z1, W2) + b2
        y = softmax(a2)

        return y
    
    # x: input, t: solution label
    def loss(self, x, t):
        y = self.predict(x)

        return cross_entropy_error(y, t)

    def accuracy(self, x, t):
        y = self.predict(x)
        y = np.argmax(y, axis=1)
        t = np.argmax(t, axis=1)

        accuracy = np.sum(y == t) / float(x.shape[0])
        return accuracy

    def numerical_gradient(self, x, t):
        lossW = lambda W: self.loss(x, t)

        grads = {}
        grads['W1'] = numerical_gradient(lossW, self.params['W1'])
        grads['b1'] = numerical_gradient(lossW, self.params['b1'])
        grads['W2'] = numerical_gradient(lossW, self.params['W2'])
        grads['b2'] = numerical_gradient(lossW, self.params['b2'])

        return grads
```

* TwoLayerNet 클래스가 사용하는 변수

변수 | 설명 |--- 
---|---|---
params | 신경망의 매개 변수를 보관하는 딕셔너리 \\ params['W1'] 은 첫 번째 층의 가중치, params['b1'] 은 첫 번째 층의 편향 \\ params['W2'] 은 두 번째 층의 가중치, params['b2'] 은 두 번째 층의 편향 | 

* TwoLayerNet 클래스의 메소드






