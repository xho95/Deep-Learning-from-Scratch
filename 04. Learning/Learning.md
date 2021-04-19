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

* 평균 제곱 오차 (MSE; mean squared error) : 가장 많이 쓰이는 손실 함수

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

```python
def mean_squared_error(y, t):
    return 0.5 * np.sum((y-t)**2)
```

* 함수의 실제 사용 예



