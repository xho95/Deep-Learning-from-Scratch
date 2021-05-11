## 5.4. Implementing a Simple Layer

### 5.4.1 Multiply Layer

* 모든 계층은 `forward()` 와 `backward()` 를 가지도록 구현함

- 곱셈 계층의 구현

### 5.4.2 Add Layer

## 5.5 Implementing an Activation Function Layer

### 5.5.1 ReLU Layer

### 5.5.2 Sigmoid Layer 

* `x` 와 `+` 노드 외에도 `exp` 노드와 `/` 노드가 새롭게 등장함

**1 단계**

* `/` 노드를 미분하면 `dy/dx` 는 `-y^2` 이 됨

**2 단계**

* `+` 노드를 미분하면 `1`

**3 단계**

* `exp` 노드를 미분하면 `exp` 가 됨 (지수 함수)

**4 단계**

* `x` 노드를 미분하면 순전파 시의 값을 서로 바꿔 곱하는 것이 됨

- 결국, Sigmoid 계층의 역전파는 순전파의 출력인 `y` 만으로 계산할 수 있음

`dL/dy` = `dL/dy y (1-y)`


## 5.6 Implementing a Softmax (Affine) Layer

### 5.6.1 Affine Layer 

* 신경망의 순전파에서는 가중치 신호의 총합을 계산하기 때문에 행렬의 내적을 사용함

- 신경망의 순전파에서 수행하는 행렬의 내적은 기하학의 '어파인 (Affine) 변환' 에 해당함. 그래서 이 계층을 'Affine 계층' 이라함


### 5.6.2 Batch Affine Layer 

* 순전파 때의 편향 덧셈은 `X W` 에 대한 편향이 각 데이터에 더해짐

- 편향의 역전파는 모든 데이터에 대한 미분을 데이터마다 더해서 구함
- 그래서 `np.sum()` 에서 (데이터를 단위로 한 축인) '0 번째 축 (`axis=0`)' 에 대해서 총합을 구하는 것임

### 5.6.3 Softmax_with_Loss Layer 

> 이 Softmax 층이 4 장에서 말한 출력에 사용하는 Softmax 층임

## 5.7 Implementing an Backpropagation

* 지금까지 구현한 계층을 조합하면 신경망을 구축할 수 있음

### 5.7.1 NueralNet Learning

* 신경망 학습 : '신경망' 에는 적응 가능한 가중치와 편향이 있고, 훈련 데이터에 적응하도록 이를 조정하는 과정을 '학습' 이라 함

- 신경망 학습 순서
    1. 미니 배치 : 훈련 데이터 중 일부를 무작위로 가져옴. 이렇게 선별한 데이터를 '미니 배치' 라 하며, 목표는 '미니 배치' 의 손실 함수 값을 줄이는 것
    2. 기울기 산출 : 미니 배치의 '손실 함수' 값을 줄이기 위해 각 가중치 매개 변수의 기울기를 구함. 기울기는 손실 함수의 값을 가장 작게하는 방향임
    3. 매개 변수 갱신 : 가중치 매개 변수를 기울기 방향으로 아주 조금 갱신함
    4. 반복 : 1~3 을 반복함

* 오차 역전파법은 두 번째 단계인 '기울기 산출' 에서 등장
* 앞 장에서 사용한 '수치 미분' 은 구현은 쉽지만 계산이 오래 걸림
* '오차 역전파법' 을 사용하면 기울기를 효율적으로 빠르게 구할 수 있음

### 5.7.1 Implementing an NueralNet by using an Backpropagation

### 5.7.3 Checking out a Gradient from Backpropagation

### 5.7.4 Implementing a Learning with Backpropagation

## 5.8 Summary

