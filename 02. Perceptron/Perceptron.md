# 2. Perceptron

Perceptron : 1957년 Frank Rosenblatt 가 고안한 알고리즘
Perceptron 은 신경망의 기원이 되는 알고리즘
 
## 2.1 Perceptron 이란?
 
Perceptron 은 다수의 신호를 입력받아 하나의 신호를 출력함
Perceptron 신호는 1/0 의 두 가지 값을 가질 수 있음
 
![Perceptron with 2-inputs](https://www.mathworks.com/matlabcentral/mlc-downloads/downloads/cf83ad39-f286-409c-81e5-6ab72e547cda/7d310d1c-dda4-4828-b9c9-fb0260a15c0b/images/screenshot.jpg) (그림 2-1)
 
* x1, x2: 입력 신호
* y: 출력 신호
* w1, w2: 가중치 (Weight)
* 원: 뉴런 or 노드
 
입력 신호가 뉴런에 보내질 때 고유한 가중치가 곱해짐
뉴런 신호의 총합이 정해진 한계를 넘을 때만 1을 출력함 (뉴런의 활성화)
이 한계를 임계값 (threshold) 이라 하며 theta (Θ) 로 나타냄
 
위 그림을 수식으로 나타내면 다음과 같음

<img src="https://latex.codecogs.com/svg.latex?y=\begin{cases}0\left(&space;w_{1}x_{1}&plus;w_{2}x_{2}\leq&space;\theta&space;\right)&space;\\&space;1\left(&space;w_{1}x_{1}&plus;w_{2}x_{2}&space;>\theta&space;\right)&space;\end{cases}" title="y=\begin{cases}0\left( w_{1}x_{1}+w_{2}x_{2}\leq \theta \right) \\ 1\left( w_{1}x_{1}+w_{2}x_{2} >\theta \right) \end{cases}" />
 
(식 2-1)
 
Perceptron 은 복수의 입력 신호에 가중치를 부여함
가중치는 각각의 신호가 결과에 주는 영향을 조절하는 요소임
가중치가 클수록 해당 신호가 그만큼 더 중요함을 의미함
 
## 2.2 단순한 논리 회로
 
Perceptron 을 활용한 문제를 살펴봄
 
### 2.2.1 AND 게이트
 
AND 게이트는 두 입력이 모두 1일 때만 1을 출력하고, 그 외에는 0을 출력함
 
AND 게이트 진리표 (그림 2-2)
 
이 AND 게이트를 Perceptron 으로 표현하려면, 진리표대로 작동하는 w1, w2, Θ 를 정하면 됨
이 조건을 만족하는 매개 변수 조합은 무한하나, 하나를 정하면 됨
 
### 2.2.2 NAND 게이트와 OR 게이트
 
NAND 게이트는 Not AND 를 의미하며, AND 게이트의 출력을 뒤집은 것처럼 동작함
x1, x2 가 모두 1일 때만 0을 출력하고, 그 외에는 1을 출력함
 
NAND 게이트 진리표 (그림 2-3)
 
NAND 게이트의 매개 변수는 AND 게이트 매개 변수의 부호를 반전하기만 하면 됨
 
OR 게이트는 입력 신호 중 하나 이상이 1이면 출력이 1이 되는 논리 회로
 
OR 게이트 진리표 (그림 2-4)
 
> Perceptron 의 매개 변수를 정하는 것은 컴퓨터가 아니라 인간임
>
> 기계 학습은 이 매개 변수를 정하는 작업을 컴퓨터가 자동으로 하도록 함
>
> (기계 학습일 때) 학습 : 적절한 매개 변수를 정하는 작업
>
> (기계 학습일 때) 사람 : Perceptron 의 구조 (모델) 을 고민하고, 학습 데이터를 줌
 
Perceptron 으로 AND, NAND, OR 논리 회로를 표현할 수 있음
중요한 것은 AND, NAND, OR 게이트 모두에서 Perceptron 의 구조가 똑같다는 것임
세 가지 게이트에서 다른 것은 매개 변수 (가중치와 임계값) 뿐임
 
## 2.3 Perceptron 구현하기
 
### 2.3.1 간단한 구현부터

```swift
func AND_1(_ x1: Double, _ x2: Double) -> Double {
    let (w1, w2, theta) = (0.5, 0.5, 0.7)
    let temp = x1 * w1 + x2 * w2
    
    if temp <= theta {
        return 0.0
    } else {
        return 1.0
    }
}

AND_1(0, 0)
AND_1(1, 0)
AND_1(0, 1)
AND_1(1, 1)
```

### 2.3.2 Weight 와 Bias 도입
 
앞에서 구현한 AND 게이트를 수정하고자 함
Θ를 -b 로 치환하면 Perceptron 의 동작이 다음 식처럼 됨

<img src="https://latex.codecogs.com/svg.latex?y=\begin{cases}0\left(&space;b&plus;w_{1}x_{1}&plus;w_{2}x_{2}\leq&space;0\right)&space;\\&space;1\left(&space;b&plus;w_{1}x_{1}&plus;w_{2}x_{2}&space;>0\right)&space;\end{cases}\" title="y=\begin{cases}0\left( b+w_{1}x_{1}+w_{2}x_{2}\leq 0\right) \\ 1\left( b+w_{1}x_{1}+w_{2}x_{2} >0\right) \end{cases}\" />
 
(식 2-2)
 
* b: 편향 (Bias)
* w1, w2 : 가중치 (Weight )
 
두 배열의 내적을 구한다음 Bias 를 더하면 다음과 같음

```swift
import Accelerate

let x = [0.0, 1.0]
let w = [0.5, 0.5]
let b = -0.7

vDSP.dot(w, x)
vDSP.dot(w, x) + b
```

### 2.3.3 Weight 와 Bias 구현하기
 
가중치 (Weight) 와 편향 (Bias) 을 도입한 AND 게이트를 구현하면 다음과 같음

```swift
func AND(_ x1: Double, _ x2: Double) -> Double {
    let x = [x1, x2]
    let w = [0.5, 0.5]
    let b = -0.7
    let temp = vDSP.dot(w, x) + b
    
    if temp <= 0 {
        return 0.0
    } else {
        return 1.0
    }
}

AND(0, 0)
AND(1, 0)
AND(0, 1)
AND(1, 1)

가중치인 w1, w2 는 각 입력 신호가 결과에 주는 영향 (중요도) 를 조절하는 매개 변수
편향인 b 는 뉴런을 얼마나 쉽게 활성화하는냐를 조절하는 매개 변수
w1, w2 를 가중치로, b 를 편향으로 구별하지만, 셋 모두를 가중치라고 할 때도 있음

NAND 와 OR 구현은 다음과 같음

func NAND(_ x1: Double, _ x2: Double) -> Double {
    let x = [x1, x2]
    let w = [-0.5, -0.5]
    let b = 0.7
    let temp = vDSP.dot(w, x) + b
    
    if temp <= 0 {
        return 0.0
    } else {
        return 1.0
    }
}

NAND(0, 0)
NAND(1, 0)
NAND(0, 1)
NAND(1, 1)

func OR(_ x1: Double, _ x2: Double) -> Double {
    let x = [x1, x2]
    let w = [0.5, 0.5]
    let b = -0.2
    let temp = vDSP.dot(w, x) + b
    
    if temp <= 0 {
        return 0.0
    } else {
        return 1.0
    }
}

OR(0, 0)
OR(1, 0)
OR(0, 1)
OR(1, 1)
```

AND, NAND, OR 는 모두 같은 구조의 Perceptron 임
차이점은 (편향을 포함한) 가중치 매개 변수의 값 뿐임
실제 구현한 코드에서도 다른 부분은 가중치와 편향의 값을 설정하는 부분뿐임
 
## 2.4 Perceptron 의 한계
 
Perceptron 으로 AND, NAND, OR 라는 3 가지 논리 회로를 구현할 수 있었음
XOR 게이트를 생각하도록 함
 
### 2.4.1 도전! XOR 게이트
 
XOR 게이트는 배타적 논리합이라는 논리 회로임
x1, x2 중 한 쪽이 1일 때만 1을 출력함
 
XOR 게이트의 진리표 (그림-2-5)
 
지금까지의 Perceptron 으로는 XOR 게이트를 구현할 수 없음
 
OR 게이트의 동작을 시각적으로 생각하도록 함
가중치 매개 변수가 (b, w1, w2) = (-0.5, 1.0, 1.0) 일 때 (그림 2-4) 진리표를 만족함
이 때의 Perceptron 은 다음 식과 같음
 
<img src="https://latex.codecogs.com/svg.latex?y=\begin{cases}0\left(&space;-0.5&plus;x_{1}&plus;x_{2}\leq&space;0\right)&space;\\&space;1\left(&space;-0.5&plus;x_{1}&plus;x_{2}&space;>0\right)&space;\end{cases}" title="y=\begin{cases}0\left( -0.5+x_{1}+x_{2}\leq 0\right) \\ 1\left( -0.5+x_{1}+x_{2} >0\right) \end{cases}" />
 
(식 2-3)
 
위의 Perceptron 은 직선으로 나뉜 두 영역을 만듬
직선의 한쪽 영역은 1을 출력하고, 다른 한 쪽은 0을 출력함
 
Perceptron 의 시각화 (그림 2-6)
 
OR 게이트를 만들려면 (그림 2-6) 의 ◯ 과 △을 직선으로 나눠야 함
그림의 직선은 네 점을 제대로 나누고 있음
 
XOR 게이트는 직선 하나로 ◯ 과 △을 나누는 영역을 만들 수 없음
 
직선의 제약 (그림 2-7)
 
### 2.4.2 션형과 비선형
 
직선 하나로 (그림 2-7) 의 ◯ 과 △ 을 나누기란 불가능함
하지만 직선이라는 제약을 없애면 가능함
 
곡선은 가능 (그림 2-7)
 
Perceptron 은 직선 하나로 나눈 영역만 표현할 수 있음
(그림 2-8) 같은 곡선 영역을 비선형 (Nonlinear) 영역, 직선 영역을 선형 (Linear) 영역이라 함
 
## 2.5 다층 Perceptron 이 충돌한다면
 
Perceptron 으로는 XOR 게이트를 표현할 수 없음
하지만, Perceptron 은 층을 쌓아 다층 (Multi-layer) Perceptron 을 만들 수 있음
층을 하나 더 쌓아서 XOR 를 표현하도록 함
 
### 2.5.1 기존 게이트 조합하기
 
기존 AND, NAND, OR 게이트를 조합하여 XOR 게이트를 만들 수 있음
 
AND, NAND, OR 게이트 (그림 2-9)
 
AND, NAND, OR 게이트를 조합하여 XOR 게이트 만드는 문제 (그림 2-10)
 
다음 같은 조합이면 XOR 게이트를 구현할 수 있음
 
AND, NAND, OR 게이트를 조합하여 XOR 게이트 만드는 정답 (그림 2-11)
 
* x1, x2 가 입력 신호, y 가 출력 신호
* x1, x2 는 NAND 와 OR 게이트의 입력
* NAND 와 OR 의 출력이 AND 게이트의 입력으로 이어짐
 
진리표로 정말 XOR 인지 확임
 
XOR 게이트 진리표 (그림 2-12)
 
* s1: NAND 출력
* s2: OR 출력
 
x1, x2, y 에 주목하면 XOR 의 출력과 같음
 
지금까지 구현한 AND, NAND, OR 함수를 사용하면 다음같이 쉽게 구현할 수 있음

```swift
func XOR(_ x1: Double, _ x2: Double) -> Double {
    let s1 = NAND(x1, x2)
    let s2 = OR(x1, x2)
    let y = AND(s1, s2)
    return y
}

XOR(0, 0)
XOR(1, 0)
XOR(0, 1)
XOR(1, 1)
```

지금 구현한 XOR 를 뉴런을 이용한 퍼셉트론으로 표현다면 다음과 같음
 
XOR 의 Perceptron (그림 2-13)
 
XOR 는 (그림 2-13) 같은 다층 구조의 네트워크 임
층이 여러 개인 Perceptron 을 다층 Perceptron 이라 함
 
단층 Perceptron 으로 표현하지 못한 것을 층을 하나 늘려 구현할 수 있음
 
### 2.5.2 XOR 게이트 구현하기
 
## 2.6 NAND 에서 컴퓨터까지
 
## 2.7 정리
 
* Perceptron 은 입출력을 가진 알고리즘이다. 입력을 주면 정해진 규칙에 따른 값을 출력한다
* Perceptron 에서는 '가중치 (Weight)' 와 '편향 (Bias)' 를 매개 변수로 설정한다
* Perceptron 으로 AND, OR 게이트 등의 논리 회로를 표현할 수 있다
* XOR 게이트는 단층 Perceptron 으로 표현할 수 없다
* 2층 Perceptron 을 사용하면 XOR 게이트를 표현할 수 있다
* 단층 Perceptron 은 선형 영역만 표현할 수 있고, 다층 Perceptron 은 비선형 영역도 표현할 수 있다
* 다층 Perceptron 은 (이론상) 컴퓨터도 표현할 수 있다

### 참고 자료

[밑바닥부터 시작하는 딥러닝](https://fliphtml5.com/hkuy/riaq/basic)
[Online LaTex Equation Editor](https://www.codecogs.com/latex/eqneditor.php)