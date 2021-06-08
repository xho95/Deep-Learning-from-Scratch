# 6. 학습 관련 기술들

* 신경망 학습에서 중요한 주제들
    1. 가중치 매개 변수의 최적값을 탐색하는 최적화 방법
    2. 가중치 매개 변수 초기값
    3. 하이퍼 파라미터 설정 방법
    4. 오버피팅 대응책 : 정규화 방법 - 가중치 감소, 드롭아웃
    5. 배치 정규화
* 이 장의 기법을 이용하면 신경망 학습의 효율과 정확도를 높일 수 있음

## 6.1 매개 변수 갱신

* 신경망 학습의 목적 : 손실 함수의 값을 가능한 낮추는 매개 변수를 찾는 것 - 최적 매개 변수를 찾는 문제
* 최적화 (Optimization) : 최적 매개 변수를 찾는 문제를 푸는 것
* 신경망 최적화는 아주 어려운 문제

- 확률적 경사 하강법 (SGD)
    - 매개 변수의 기울기인 미분을 이용하여 최적 매개 변수의 값을 찾음
    - 매개 변수의 기울기를 구하고, 기울어진 방향으로 매개 변수 값을 갱신하는 것을 반복하여 최적 매개 변수로 다가감
    - 매개 변수를 무작정 찾는 것보다는 똑똑한 방법임

* SGD 의 단점을 알아보고 다른 최적화 기법을 소개함

### 6.1.1 모험가 이야기

* 최적 매개 변수를 탐색하는 일은 지도도 없이 눈을 가리고 가장 낮은 골짜기를 찾는 것과 같음

### 6.1.2 확률적 경사 하강법 (SGD)

* SGD 복습

- [식 5-4] <img src="https://latex.codecogs.com/svg.latex?W\leftarrow&space;W-\eta&space;\frac{\partial&space;L}{\partial&space;W}" title="W\leftarrow W-\eta \frac{\partial L}{\partial W}" />
- <img src="https://latex.codecogs.com/svg.latex?\frac{\partial&space;L}{\partial&space;W}" title="{\partial L}{\partial W}" /> : W 에 대한 손실 함수의 기울기
- <img src="https://latex.codecogs.com/svg.latex?\eta" title="\eta" /> : 학습률 - 실제로는 0.01 이나 0.001 과 같은 값을 미리 정해서 사용함

* SGD 는 기울어진 방향으로 일정 거리만 가겠다는 단순한 방법

### 6.1.3 SGD 의 단점

### 6.1.4 모멘텀

### 6.1.5 AdaGrad

### 6.1.6 Adam

### 6.1.7 어느 갱신 방법을 이용할 것인가?

### 6.1.8 MNIST 데이터셋으로 본 갱신 방법 비교

## 6.2 가중치의 초기값

### 6.2.1 초기값을 0으로 하면?

### 6.2.2 은닉층의 활성화값 분포

### 6.2.3 ReLU 를 사용할 때의 가중치 초기값

### 6.2.4 MNIST 데이터셋으로 본 가중치 초기값 비교

## 6.3 배치 정규화

### 6.3.1 배치 정규화 알고리즘

### 6.3.2 배치 정규화의 효과

## 6.4 바른 학습을 위해

* 오버피팅 (Overfitting) : 신경망이 훈련 데이터에만 지나치게 적응되어 그 외 데이터에는 제대로 대응하지 못하는 상태

- 기계 학습의 목표
    - 범용 성능 :  훈련 데이터에는 없는 데이터가 주어져도 바르게 식별하는 모델이 바람직함
    - 복잡하고 표현력이 높은 모델을 만들수록, 오버피팅을 억제하는 기술이 중요함

### 6.4.1 오버피팅

* 오버피팅이 일어나는 두 경우
    1. 매개 변수가 많고 표현력이 높은 모델
    2. 훈련 데이터가 적음

- 일부러 오버피팅을 일으켜봄
    1. 7-층 네트워크 사용하여 네트워크 복잡성을 높임 : 각 층의 뉴런은 100개, 활성 함수는 ReLU 사용
    2. 60,000 개의 MNIST 훈련 데이터 중 300 개만 사용

* 전체 코드 중에서 데이터를 읽는 부분 (전체 코드는 Jupytor 파일 참고)

```python
(x_train, t_train), (x_test, t_test) = load_mnist(normalize=True)

x_train = x_train[:300]     # 학습 데이터 수를 줄임
t_train = t_train[:300]
```

* 전체 코드 중에서 훈련을 수행하는 부분 

```python
network = MultiLayerNet(input_size=784, hidden_size_list=[100, 100, 100, 100, 100, 100], output_size=10)
optimizer = SGD(lr=0.01)    # 학습률이 0.01 인 SGD 로 매개 변수 갱신

max_epochs = 201
train_size = x_train.shape[0]
batch_size = 100

train_loss_list = []
train_acc_list = []
test_acc_list = []

iter_per_epoch = max(train_size / batch_size, 1)
epoch_cnt = 0

for i in range(1000000000):
    batch_mask = np.random.choice(train_size, batch_size)
    x_batch = x_train[batch_mask]
    t_batch = t_train[batch_mask]

    grads = network.gradient(x_batch, t_batch)
    optimizer.update(network.params, grads)

    if i % iter_per_epoch == 0:
        train_acc = network.accuracy(x_train, t_train)
        test_acc = network.accuracy(x_test, t_test)
        train_acc_list.append(train_acc)
        test_acc_list.append(test_acc)

        print("epoch:" + str(epoch_cnt) + ", train acc:" + str(train_acc) + ", test acc:" + str(test_acc))

        epoch_cnt += 1
        if epoch_cnt >= max_epochs:
            break
```

* 에폭마다 모든 훈련 데이터와 모든 시험 데이터 각각에 대한 정확도 산출
* `train_acc_list` 와 `test_acc_list` 에 에폭 단위의 정확도 저장

- 에폭 단위
    - 모든 훈련 데이터를 한 번씩 본 단위 (1 에폭 : 전체 훈련 데이터를 한 번 사용해본 상황)
    - 딥러닝에서의 에폭은 학습의 횟수를 의미함 [배치(batch)와 에포크(epoch)란?](https://bskyvision.com/803)

* [결과 Graph] : Jupytor 결과 참고
    1. 훈련 데이터를 사용하여 측정한 정확도는 100 에폭을 지나는 무렵 (실제로는 126) 부터 거의 100 %임
    2. 하지만 시험 데이터를 사용한 정확도와는 큰 차이가 있음
    3. 정확도가 크게 벌어지는 것은 훈련 데이터에만 적응한 결과임
    4. 훈련 때 사용하지 않은 범용 (시험) 데이터에는 제대로 대응하지 못하는 것을 확인할 수 있음

### 6.4.2 가중치 감소

* 가중치 감소 (Weight Decay)
    - 오버 피팅 억제용으로 예전부터 많이 이용함
    - 학습 과정에서 큰 가중치에 대해서는 그에 상응하는 큰 패널티를 부여하여 오버피팅을 억제하는 방법
    - 원래 오버피팅은 가중치 매개 변수의 값이 커서 발생하는 경우가 많기 때문임

* 신경망 학습의 목적 : 손실 함수의 값을 줄이는 것
    - 가중치의 제곱 (L2 법칙) 을 손실 함수에 더하면 가중치가 커지는 것을 억제할 수 있음
    - 가중치를 <img src="https://latex.codecogs.com/svg.latex?W" title="W" /> 라 하면 L2 법칙에 따른 가중치 감소는 <img src="https://latex.codecogs.com/gif.latex?\frac{1}{2}&space;\lambda&space;W^{2}" title="\frac{1}{2} \lambda W^{2}" /> 이며, 이 값을 손실 함수에 더함
    - <img src="https://latex.codecogs.com/gif.latex?\lambda" title="\lambda" /> : '정류화 (Regularization)' 의 세기를 조절하는 하이퍼 파라미터 - <img src="https://latex.codecogs.com/gif.latex?\lambda" title="\lambda" /> 가 클 수록 큰 가중치에 대한 패널티가 커짐
    - <img src="https://latex.codecogs.com/gif.latex?\frac{1}{2}&space;\lambda&space;W^{2}" title="\frac{1}{2} \lambda W^{2}" /> 앞의 <img src="https://latex.codecogs.com/gif.latex?\frac{1}{2}" title="\frac{1}{2}" /> 은 <img src="https://latex.codecogs.com/gif.latex?\frac{1}{2}&space;\lambda&space;W^{2}" title="\frac{1}{2} \lambda W^{2}" /> 의 미분 결과인 <img src="https://latex.codecogs.com/gif.latex?\lambda&space;W" title="\lambda W" /> 를 조정하는 역할을 하는 상수

* 가중치 감소는 모든 가중치 각각의 손실 함수에 <img src="https://latex.codecogs.com/gif.latex?\frac{1}{2}&space;\lambda&space;W^{2}" title="\frac{1}{2} \lambda W^{2}" /> 을 더함
* 따라서 가중치의 기울기 계산에서는 오차 역전파법의 결과에 '정류화' 항을 미분한 <img src="https://latex.codecogs.com/gif.latex?\lambda&space;W" title="\lambda W" /> 을 더함

* 책에서는 이렇게 말하고 있지만, '정류화 (Regularization)' 는 결국 가중치 중에 튀는 값을 평탄화하는 기능에 더 가깝지 않을까함

> [L2 법칙](https://koreapy.tistory.com/530) : 결국 Norm 에서 온 개념 - L2 는 일반적인 거리 개념, [L2 Regularization 의 이해](https://light-tree.tistory.com/125) - 데이터에서 튀는 값을 평탄화하는 효과

- '<img src="https://latex.codecogs.com/gif.latex?\lambda" title="\lambda" /> = 0.1' 이라는 가중치 감소를 적용한 결과

- [결과 Graph] : Jupytor 결과 참고
    1. 훈련 데이터에서의 정확도와 시험 데이터에서의 정확도는 아직 차이가 있으나, 가중치 감소를 써서 이전보다 차이가 줄어듦
    2. 오버피팅이 억제된 것을 확인 가능
    3. 이전과 달리 훈련 데이터에 대한 정확도가 100% 에 도달하지 못함

### 6.4.3 드롭아웃

* 가중치 감소 
    - 간단하기 구현할 수 있고 오버피팅을 어느 정도 억제할 수 있음
    - 신경망 모델이 복잡해지면 가중치 감소만으로는 대응하기 어려움

* 드롭아웃 (Dropout)
    - 뉴런을 임의로 삭제하면서 학습하는 방법 : 훈련 때 은닉층의 뉴런을 무작위로 골라 삭제함
    - 삭제된 뉴런은 신호를 전달하지 않음
    - 훈련 때는 데이터를 흘릴 때마다 삭제할 뉴런을 무작위로 선태갛고, 시험 때는 모든 뉴런에 신호를 전달함
    - 단, 시험 때는 각 뉴런의 출력에 훈련 때 삭제한 비율을 곱하여 출력함

* 드롭아웃 구현 : Jupytor 참고 
    - 순전파를 담당하는 forward 메소드에서 훈련 때만 잘 계산해두면, 시험 때는 단순히 데이터를 흘리기만 하면 됨
    - 삭제한 비율은 곱하지 않아도 됨 : 실제 딥러닝 프레임웍들도 비율을 곱하지 않음

```python
class Dropout:
    """
    http://arxiv.org/abs/1207.0580
    """
    def __init__(self, dropout_ratio=0.5):
        self.dropout_ratio = dropout_ratio
        self.mask = None

    def forward(self, x, train_flg=True):
        if train_flg:
            self.mask = np.random.rand(*x.shape) > self.dropout_ratio
            return x * self.mask
        else:
            return x * (1.0 - self.dropout_ratio)

    def backward(self, dout):
        return dout * self.mask
```

* 드롭아웃의 핵심
    - 훈련 시에는 순전파 때마다 `self.mask` 에 삭제할 뉴런을 `False` 로 표시한다는 것임
    - `self.mask` 는 `x` 와 형상이 같은 배열을 무작위로 생성해서, `dropout_ratio` 보다 큰 원소만 `True` 로 설정함
    - 역전파 때의 동작은 `ReLU` 와 같음
    - 즉, 순전파 때 신호를 통과시키는 뉴런은 역전파 때도 신호를 통과하고, 순전파 때 통과시키지 않은 뉴런은 역전파 때도 신호를 차단함

* 드롭아웃의 결과 : Jupytor 참고
    - 실험은 앞과 같이 7-층 네트워크를 사용함
    - 드롭아웃을 적용하면 훈련 데이터와 시험 데이터에 대한 정확도 차이가 줄어듦
    - 훈련 데이터에 대한 정확도도 100% 도달하지 않음
    - 드롭아웃을 사용하면 표현력을 높이면서도 오버피팅을 억제할 수 있음

> 앙상블 학습 (ensemble learning)
>   - 개별적으로 학습시킨 여러 모델의 출력을 평균 내어 추론하는 방식 - 기계 학습에서 자주 사용됨
>   - 비슷한 신경망을 여러 개 준비해서 따로 학습시키고 출력을 평균내는 방식 
>   - 앙상블 학습을 수행하면 신경망의 정확도가 몇% 개선됨
>   - 앙상블 학습은 드롭아웃과 밀접함
>   - 드롭아웃이 학습할 때 뉴런을 무작위로 삭제하는 행위가 다른 모델을 학습하는 것과 유사함
>   - 추론 때는 뉴런의 출력에 삭제한 비율을 곱함으로써 앙상블 학습에서의 평균과 같은 효과를 낼 수 있음
>   - 드롭아웃은 앙상블 학습과 같은 효과를 하나의 네트워크로 구현한 것이라 생각할 수 있음

## 6.5 적절한 하이퍼 파라미터 값 찾기

* 신경망의 하이퍼 파라미터
    - 각 층의 뉴런 수, 배치 크기, 매개 변수 갱신 시의 학습률, 가중치 감소 등
    - 하이퍼 파라미터의 값을 적절히 설정하지 않으면 모델의 성능이 크게 떨어짐
    - 하이퍼 파라미터의 값은 매우 중요하지만 값을 결정하기까지 많은 시행착오를 겪게됨

* 이번 절은 하이퍼 파라미터를 최대한 효율적으로 탐색하는 방법을 설명함

### 6.5.1 검증 데이터

* 데이터셋
    - 지금까지는 데이터셋을 훈련 데이터와 시험 데이터라는 두 가지로 분리했음
    - 훈련 데이터로는 학습을 하고, 시험 데이터로는 범용 성능을 평가했음
    - 이로써 (훈련 데이터에만) 오버피팅 된 건 아닌지, 범용 성능은 어느 정도인지를 평가할 수 있었음

* 하이퍼 파라미터
    - 앞으로 하이퍼 파라미터를 다양한 값으로 설정하고 검증함
    - 하이퍼 파라미터의 성능을 평가할 때는 시험 데이터를 사용해서는 안 됨 : 매우 중요!
    - 왜냐면 시험 데이터를 사용하여 하이퍼 파라미터를 조정하면 하이퍼 파라미터 값이 시험 데이터에 오버피팅되기 때문임
    - 하이퍼 파라미터의 값이 시험 데이터에만 적합하도록 조정되니, 범용 성능이 떨어지는 모델이 될 수 있음

* 검증 데이터 (validation data)
    - 하이퍼 파라미터를 조정할 때는 하이퍼 파라미터 전용 확인 데이터가 필요함
    - 검증 데이터 : 하이퍼 파라미터 조정용 데이터

> 훈련 데이터 : 매개 변수의 학습에 사용
> 검증 데이터 : 하이퍼 파라미터의 성능을 평가하는데 사용
> 시험 데이터 : 신경망의 범용 성능을 평가하는데 사용 - (이상적으로는) 마지막에 (한 번만) 사용

* MNIST 데이터셋
    - 훈련 데이터와 시험 데이터로만 분리되어 있음
    - 훈련 데이터 중 20% 정도를 검증 데이터로 미리 분리할 필요 있음

```python
(x_train, t_train), (x_test, t_test) = load_mnist(normalize=True)

# 훈련 데이터를 섞음
x_train, t_train = shuffle_dataset(x_train, t_train)

# 20% 를 검증 데이터로 분할
validation_rate = 0.20
validation_num = int(x_train.shape[0] * validation_rate)

x_val = x_train[:validation_num]
t_val = t_train[:validation_num]
x_train = x_train[validation_num:]
t_train = t_train[validation_num:]
```

* 데이터셋 안의 데이터가 치우쳐져 있을 수 있으므로, 훈련 데이터를 분리하기 전에 입력 데이터와 정답 레이블을 섞음

### 6.5.2 하이퍼 파라미터 최적화

* 하이퍼 파라미터 최적화
    - 핵심은 하이퍼 파라미터의 '최적값' 이 존재하는 범위를 조금씩 줄여나간다는 것
    - 범위를 조금씩 줄이려면 대략적인 범위를 설정하고 그 범위에서 무작위로 하이퍼 파라미터 값을 고른 후 (샘플링), 그 값으로 정확도를 평가함
    - 정확도를 살펴가면서, 작업을 반복하여 하이퍼 파라미터의 '최적값' 범위를 좁혀감 

> 신경망의 하이퍼 파라미터 최적화에는 '그리드 탐색 (Grid Search)' 같은 규칙적인 탐색보다는 '무작위 샘플링' 이 더 좋다고 알려짐
>
> 최종 정확도에 미치는 영향력이 하이퍼 파라미터마다 다르기 때문임

* 하이퍼 파라미터 최적화 시의 고려 사항
    - 딥러닝 학습에 오랜 시간 (며칠 또는 몇 주 이상) 이 걸림 : 나쁠 것 같은 값은 일찍 포기하는 것이 좋음
    - 학습을 위한 에폭을 작게 하여, 1회 평가에 걸리는 시간을 단축하는 것이 효과적

* 하이퍼 파라미터 최적화 총정리
    1. 0 단계 : 하이퍼 파라미터 값의 범위를 설정함
    2. 1 단계 : 설정된 범위에서 하이퍼 파라미터 값을 무작위로 추출함
    3. 2 단계 : 1 단계에서 샘플링한 하이퍼 파라미터 값으로 학습하고, 검증 데이터로 정확도를 평가함 (에폭은 작게 설정함)
    4. 3 단계 : 1 단계와 2 단계를 특정 횟수만큼 (100호 등) 반복하여, 정확도의 결과를 보고 하이퍼 파라미터의 범위를 좁힘

- 이렇게 하이퍼 파라미터의 범위를 좁혀가다가, 압축한 범위에서 값을 하나 고름

> 여기 있는 방법은 '직관' 적인 방법에 해당함. 보다 과학적인 방법으로는 '베이즈 최적화 (Bayesian Optimization)' 가 있음. 
>
> 이는 '베이즈 정리' 라는 수학 이론을 사용하여 더 엄밀하고 효율적으로 최적화를 수행함

### 6.5.3 하이퍼 파라미터 최적화 구현하기

* MNIST 데이터셋으로 하이퍼 파라미터 최적화 수행
    - 학습률과 가중치 감소 계수를 탐색함 : 스탠퍼드 대학교 CS231N 수업 참고
    - 여기서는 가중치 감소 계수를 10^-8 ~ 10^-4, 학습률을 10^-6 ~ 10^-2 범위에서 시작함
    - 해당 코드는 아래와 같음

```python
weight_decay = 10 ** np.random.uniform(-8, -4)
lr = 10 ** np.random.uniform(-6, -2)
```

* 무작위로 추출한 값으로 학습을 수행함
* 그 후 다양한 하이퍼 파라미터 값으로 학습을 반복하며 신경망에 좋을 것 같은 값을 관철함

- 구현 및 결과 : Jupytor 참고

* 결과는 검증 데이터의 학습 추이를 정확도가 높은 순서로 나열함
* Best-5 까지는 학습이 순조로움을 알 수 있음 : 이들의 하이퍼 파라미터 값 (학습률과 가중치 감소 계수) 를 살펴봄
* 학습률은 0.005 ~ 0.009, 가중치 감소 계수는 10^-8 ~ 10^-5 정도임을 알 수 있음
* 잘 될 것 같은 값의 범위를 관찰하고 좁혀나간 후, 축소된 범위로 똑같은 작업을 반복함
* 이렇게 범위를 좁혀가다가 특정 단계에서 최종 하이퍼 파라미터 값을 하나 선택함

## 6.6 정리

1. 매개 변수 갱신 방법에는 확률적 경사 하강범 (SGD) 외에도, 모멘텀, AdaGrad, Adam 등이 있음
2. 가중치 초기값을 정하는 방법은 올바른 학습을 하는데 있어 매우 중요함
3. 가중치 초기값으로는 'Xavier 초기값' 과 'He 초기값' 이 효과적임
4. 배치 정규화를 사용하면 학습을 빠르게 진행할 수 있으며, 초기값에 영향을 덜 받게 됨
5. 오버피팅을 억제하는 정류화 기술로는 '가중치 감소' 와 '드롭아웃' 이 있음
6. 하이퍼 파라미터 값 탐색은 최적값이 존재할 법한 범위를 점차 좁히면서 하는 것이 효과적임

