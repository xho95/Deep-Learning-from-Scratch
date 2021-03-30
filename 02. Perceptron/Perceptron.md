# 2. Perceptron

* Perceptron : 1957년 Frank Rosenblatt 가 고안한 알고리즘[^book]
* Perceptron 은 신경망의 기원이 되는 알고리즘
 
## 2.1 Perceptron 이란?
 
* Perceptron 은 다수의 신호를 입력받아 하나의 신호를 출력함
* Perceptron 신호는 1/0 의 두 가지 값을 가질 수 있음
 
> ![Perceptron with 2-inputs](https://www.mathworks.com/matlabcentral/mlc-downloads/downloads/cf83ad39-f286-409c-81e5-6ab72e547cda/7d310d1c-dda4-4828-b9c9-fb0260a15c0b/images/screenshot.jpg) 
> 
> 그림 2-1: 입력이 2 개인 퍼셉트론
>
> `x1`, `x2` : 입력 신호 \
> `y` : 출력 신호 \
> `w1`, `w2` : 가중치 (Weight) \
> 원 : 뉴런 or 노드
 
* 입력 신호가 뉴런에 보내질 때 고유한 가중치가 곱해짐
* 뉴런 신호의 총합이 정해진 한계를 넘을 때만 1을 출력함 : 뉴런의 활성화
* 이 한계를 임계값 (threshold) 이라 하며 `Θ` (theta) 로 나타냄
 
- 위 그림을 수식으로 나타내면 다음과 같음[^equation]

> <img src="https://latex.codecogs.com/svg.latex?y=\begin{cases}0\left(&space;w_{1}x_{1}&plus;w_{2}x_{2}\leq&space;\theta&space;\right)&space;\\&space;1\left(&space;w_{1}x_{1}&plus;w_{2}x_{2}&space;>\theta&space;\right)&space;\end{cases}" title="y=\begin{cases}0\left( w_{1}x_{1}+w_{2}x_{2}\leq \theta \right) \\ 1\left( w_{1}x_{1}+w_{2}x_{2} >\theta \right) \end{cases}" />
> 
> 식 2-1
 
* Perceptron 은 복수의 입력 신호에 가중치를 부여함
* 가중치는 각각의 신호가 결과에 주는 영향을 조절하는 요소임
* 가중치가 클수록 해당 신호가 그만큼 더 중요함을 의미함
 
## 2.2 단순한 논리 회로
 
* Perceptron 을 활용한 문제
 
### 2.2.1 AND 게이트
 
* AND 게이트: 두 입력이 모두 1 일 때만 1 을 출력, 그 외는 0 을 출력

> ![AND 게이트 진리표](https://cdn.instrumentationtools.com/wp-content/uploads/2018/10/Logic-Gates-and-Truth-tables.png)
> 
> 그림 2-2: AND 게이트의 진리표
 
* AND 게이트를 Perceptron 으로 표현하려면, 진리표대로 작동하는 `w1`, `w2`, `Θ` 를 정하면 됨
* 조건을 만족하는 매개 변수 조합은 무한하지만, 하나를 정하면 됨
 
### 2.2.2 NAND 게이트와 OR 게이트
 
* NAND 게이트: Not AND 라는 의미, AND 게이트의 출력을 뒤집은 것처럼 동작함
* `x1`, `x2` 가 모두 1 일 때만 0 을 출력, 그 외는 1 을 출력

> ![NAND 게이트 진리표](https://cdn.instrumentationtools.com/wp-content/uploads/2018/10/Universal-Logic-Gates-and-Truth-tables.png) 
> 
> 그림 2-3: NAND 게이트의 진리표
 
* NAND 게이트의 매개 변수는 AND 게이트 매개 변수의 부호를 반전하면 됨
 
- OR 게이트: 입력 신호 중 하나 이상이 1 이면 출력이 1 이 되는 논리 회로

> ![AND 게이트 진리표](https://cdn.instrumentationtools.com/wp-content/uploads/2018/10/Logic-Gates-and-Truth-tables.png)
> 
> 그림 2-4: OR 게이트의 진리표
  
> Perceptron 의 매개 변수를 정하는 것은 컴퓨터가 아니라 인간임 \
> 기계 학습은 이 매개 변수를 정하는 작업을 컴퓨터가 자동으로 하도록 함 \
> (기계 학습일 때) 학습 : 적절한 매개 변수를 정하는 작업 \
> (기계 학습일 때) 사람 : Perceptron 의 구조 (모델) 을 고민하고, 학습 데이터를 줌
 
* Perceptron 으로 AND, NAND, OR 논리 회로를 표현할 수 있음
* 중요한 것은 AND, NAND, OR 게이트 모두 Perceptron 구조가 똑같다는 것임
* 세 가지 게이트에서 다른 것은 매개 변수 (가중치와 임계값) 뿐임
 
## 2.3 Perceptron 구현하기
 
### 2.3.1 간단한 구현부터

* 논리 회로를 Swift 로 구현: Python 구현은 책 코드 참조

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

* 출력은 '그림 2-2' 와 같음: AND 게이트 구현 확인 

### 2.3.2 Weight 와 Bias 도입
 
* 앞에서 구현한 AND 게이트 수정: `Θ` 를 `-b` 로 치환하여 다음 식 만듦

> <img src="https://latex.codecogs.com/svg.latex?y=\begin{cases}0\left(&space;b&plus;w_{1}x_{1}&plus;w_{2}x_{2}\leq&space;0\right)&space;\\&space;1\left(&space;b&plus;w_{1}x_{1}&plus;w_{2}x_{2}&space;>0\right)&space;\end{cases}\" title="y=\begin{cases}0\left( b+w_{1}x_{1}+w_{2}x_{2}\leq 0\right) \\ 1\left( b+w_{1}x_{1}+w_{2}x_{2} >0\right) \end{cases}\" />
>  
> 식 2-2
> 
> `b` : 편향 (Bias) \
> `w1`, `w2` : 가중치 (Weight )
 
* '식 2-1' 과 '식 2-2' 는 표기만 다르고, 의미는 같음

- '식 2-2' 방식으로 구현: 두 배열의 내적을 구한 다음 Bias 를 더함

```swift
import Accelerate

let x = [0.0, 1.0]
let w = [0.5, 0.5]
let b = -0.7

vDSP.dot(w, x)
vDSP.dot(w, x) + b
```

### 2.3.3 Weight 와 Bias 구현하기
 
* '가중치 (Weight)' 와 '편향 (Bias)' 을 도입한 AND 게이트 구현

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
```

* 가중치인 `w1`, `w2` 는 각각의 입력 신호가 결과에 주는 '영향 (중요도)' 를 조절하는 매개 변수
* 편향인 `b` 는 뉴런이 얼마나 쉽게 활성화 되는지를 조절하는 매개 변수

> `w1`, `w2` 는 가중치, `b` 는 편향이라고 하지만, 책에서 셋 모두를 '가중치' 라고 할 때도 있음

* NAND 게이트와 OR 게이트 구현

```swift
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

* AND, NAND, OR 는 모두 같은 구조의 Perceptron 이고 차이점은 매개 변수 뿐임
* 실제 구현 코드에서도 가중치와 편향의 값을 설정하는 부분만 다름
 
## 2.4 Perceptron 의 한계
 
* Perceptron 으로 AND, NAND, OR, 3 가지 논리 회로를 구현할 수 있었음
* XOR 게이트를 생각해 보도록 함
 
### 2.4.1 도전! XOR 게이트
 
* XOR 게이트: '배타적 논리합' 이라는 논리 회로, `x1`, `x2` 중 한 쪽이 1 일 때만 1 을 출력
 
> ![XOR 게이트의 진리표](https://cdn.instrumentationtools.com/wp-content/uploads/2018/10/Logic-Gates-and-Truth-tables.png)
> 
> 그림-2-5: XOR 게이트의 진리표
 
* 지금까지의 Perceptron 으로는 XOR 게이트를 구현할 수 없음: 시각적으로 이해
 
- OR 게이트 동작을 시각적으로 생각
- 가중치 매개 변수가 (b, w1, w2) = (-0.5, 1.0, 1.0) 일 때 '그림 2-4' 진리표를 만족함
- 이 때, Perceptron 은 다음 식과 같음
 
<img src="https://latex.codecogs.com/svg.latex?y=\begin{cases}0\left(&space;-0.5&plus;x_{1}&plus;x_{2}\leq&space;0\right)&space;\\&space;1\left(&space;-0.5&plus;x_{1}&plus;x_{2}&space;>0\right)&space;\end{cases}" title="y=\begin{cases}0\left( -0.5+x_{1}+x_{2}\leq 0\right) \\ 1\left( -0.5+x_{1}+x_{2} >0\right) \end{cases}" />

> 식 2-3
 
* '식 2-3' 의 Perceptron 은 직선으로 나뉜 두 영역을 만듦
* 직선 한 쪽은 1 출력, 다른 쪽은 0 을 출력

![Perceptron 의 시각화](https://i.stack.imgur.com/epSZC.png)

> 그림 2-6: Perceptron 의 시각화
 
* OR 게이트를 만들려면, '그림 2-6' 의 ◯ 과 △ 을 직선으로 나눠야 함
* 그림의 직선은 네 점을 제대로 나눔
 
- XOR 게이트 동작: 직선 하나로 ◯ 과 △ 을 나눌 수 없음

> ![직선의 제약](https://encrypted-tbn0.gstatic.com/images?q=tbn:ANd9GcTZf0vxDMiiYFfb-j7kn4mSkZTeLwrpAO4kOPF3RxtuFM7EMR4veDn03YGLtOq86K5hVNg&usqp=CAU)
>
> 그림 2-7
 
### 2.4.2 선형과 비선형
 
* 직선으로는 ◯ 과 △ 을 나눌 수 없지만, 직선이라는 제약을 없애면 가능함
 
> ![곡선은 가능](https://www.allaboutcircuits.com/uploads/articles/advanced-machine-learning-with-the-multilayer-perceptron_rk_aac_image5.jpg)
>
> 그림 2-7
 
* Perceptron 의 한계: 직선 하나로 나눈 영역만 표현
* '그림 2-8' 같은 곡선 영역 = 비선형 (Nonlinear) 영역, 직선 영역 = 선형 (Linear) 영역
 
## 2.5 다층 Perceptron 이 충돌한다면
 
* Perceptron 으로는 XOR 게이트를 표현할 수 없음
* 하지만, Perceptron 은 층을 쌓아 **다층 (Multi-layer) Perceptron** 을 만들 수 있음
* 층을 하나 더 쌓아서 XOR 를 표현
 
### 2.5.1 기존 게이트 조합하기
 
* XOR 게이트 문제를 다른 관점에서 생각
* XOR 게이트를 만드는 방법은 다양함 : AND, NAND, OR 조합으로 XOR 게이트를 만들 수 있음

> ![AND, NAND, OR 게이트](https://lh3.googleusercontent.com/proxy/GeHz6crQXOwfx0NsECFbwOLxRxGYLWjQTNMhIwfqlGtzW0Yf7Nq5h9I7mvZVjhlRSkX_oU2eZzarIy3gqDMnM3AAXzipr6-5oUgLxtXC4EAFml5Drc9qczf0NKHRFg)
> 
> 그림 2-9: AND, NAND, OR 게이트

* AND, NAND, OR 게이트 조합 방법을 생각

> ![XOR 게이트](https://lh3.googleusercontent.com/proxy/XecFpHpoyribt9UiO2BEVFxEJ5pa_fUTNs70pkjbod-pXNgd1TRXgLWXDlFMCYj7p3tZhWF3JSbWovkA062GKQt8GoRThT1rTA0Yv31d8HbMTmq2lsXjoX2EGmk)
>
> 그림 2-10
 
다음 같은 조합이면 XOR 게이트를 구현할 수 있음
 
* `x1`, `x2`: 입력 신호, `y`: 출력 신호
* `x1`, `x2`: NAND 와 OR 게이트의 입력
* NAND 와 OR 의 출력: AND 게이트의 입력

> ![AND, NAND, OR 조합](https://encrypted-tbn0.gstatic.com/images?q=tbn:ANd9GcSivBH0BtB1McAzlYHznAvViQFFxkVtANCo8A&usqp=CAU) 
>
> 그림 2-11: AND, NAND, OR 조합으로 만든 XOR 게이트
 
* '그림 2-11' 이 XOR 인지 진리표로 확인: `s1` 은 NAND 출력, `s2` 는 OR 출력
* `x1`, `x2`, `y` 에 주목하면 XOR 출력과 같음

> 그림 2-12: XOR 게이트 진리표
 
### 2.5.2 XOR 게이트 구현하기

* '그림 2-11' 로 조합한 XOR 게이트 구현
* 정의한 AND, NAND, OR 함수를 사용하여 구현

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

* 구현한 XOR 를 뉴런을 이용한 Perceptron 으로 표현하면 다음과 같음

> ![XOR Perceptron](https://i.stack.imgur.com/z9iMR.png) 
> 
> 그림 2-13: XOR 의 Perceptron
 
* XOR 는 '그림 2-13' 같은 다층 구조의 네트워크: 왼쪽부터 0층, 1층, 2층
* '그림 2-13' 의 Perceptron 은 이전 Perceptron 들과 형태가 다름
* AND, OR: 단층 Perceptron, XOR: 다층 Perceptron - 층이 여러 개인 Perceptron

> '그림 2-13' 의 Perceptron 은 가충치를 가진 층이 2개이므로 '2층 (2-layer) Perceptron 임 \
> 책에 따라 이를 '3층 Perceptron' 이라고 하는 곳도 있음

* 2층 Perceptron 에서는 0층에서 1층으로 신호를 전달하고, 이어서 1층에서 2층으로 신호를 전달함
    1. 0층의 두 뉴런이 입력 신호를 받아 1층의 뉴런으로 신호를 보냄
    2. 1층의 뉴런이 2층의 뉴런으로 신호를 보내고, 2층의 뉴런은 이 입력 신호를 바탕으로 y 출력

- 2층 Perceptron 동작을 공장 조립 라인에 비유할 수 있음: 1층 부품을 2층 작업자에 전달
- 이 처럼 XOR 게이트 Perceptron 에서는 작업자들 사이에 '부품을 전달' 하는 일이 이루어짐

* '2층 Perceptron' 으로 XOR 게이트를 구현
* 단층 Perceptron 으로 표현하지 못한 것을 층을 하나 늘려 구현함
* Perceptron 은 층을 쌓아서 다양한 것을 표현할 수 있음
 
## 2.6 NAND 에서 컴퓨터까지

* 다층 Perceptron 은 보다 복잡한 회로를 만들 수 있음
* 덧셈을 처리하는 가산기, 2진수를 10진수로 변환하는 인코더, 그리고 컴퓨터도 표현할 수 있음
* 컴퓨터는 NAND 게이트만으로 재현할 수 있음: 그리고 NAND 게이트는 Perceptron 으로 만들 수 있음

- 이론상 2층 Perceptron 이면 컴퓨터를 만들 수 있음
- 2층 Perceptron, 즉 비선형 Sigmoid 함수를 활성 함수로 사용하면 임의의 함수를 표현할 수 있음
- 하지만 2층 Perceptron 구조에서 가중치 설정만으로 컴퓨터를 만들기는 너무 어려움
- Perceptron 으로 표현하는 컴퓨터도 여러 층을 겹겹이 겹친 구조로 만드는 방향이 자연스러움

## 2.7 정리

* Perceptron 은 다음 장에서 배울 신경망의 기초
* 이번 장에서 배운 것
    1. Perceptron 은 입출력을 가진 알고리즘이다. 입력을 주면 정해진 규칙에 따른 값을 출력한다
    2. Perceptron 에서는 '가중치 (Weight)' 와 '편향 (Bias)' 를 매개 변수로 설정한다
    3. Perceptron 으로 AND, OR 게이트 등의 논리 회로를 표현할 수 있다
    4. XOR 게이트는 단층 Perceptron 으로 표현할 수 없다
    5. 2층 Perceptron 을 사용하면 XOR 게이트를 표현할 수 있다
    6. 단층 Perceptron 은 선형 영역만 표현할 수 있고, 다층 Perceptron 은 비선형 영역도 표현할 수 있다
    7. 다층 Perceptron 은 (이론상) 컴퓨터도 표현할 수 있다

### 참고 자료

* [^book]: 여기 정리한 내용들은 한빛 출판사에서 출판한 [밑바닥부터 시작하는 딥러닝](https://fliphtml5.com/hkuy/riaq/basic) 이라는 책의 내용을 정리한 것입니다.
* [^equation]: 본문에 있는 수식들은 'LaTex' 수식을 [Online LaTex Equation Editor](https://www.codecogs.com/latex/eqneditor.php) 를 사용하여 웹에서 볼 수 있도록 한 것입니다.