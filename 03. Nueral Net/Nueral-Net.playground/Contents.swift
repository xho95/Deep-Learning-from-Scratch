/*:
 # Nueral Net
 
 ## 3.1 Perceptron 에서 Nueral-Net 으로
 
 ### 3.1.1 Nueral-Net 의 예
 
 ### 3.1.2 Perceptron 복습
 
 ### 3.1.3 Activation Fuction 의 등장
 
 * 입력 신호의 총합을 출력 신호로 변환하는 함수
 
 a = b + w1x1 + w2x2
 y = h(a)
 
 ## 3.2 Activation Function
 
 * Activation Fuction 을 Step Fuction 에서 다른 함수로 변경하는 것이 Nueral-Net 으로 가는 열쇠
 
 ### 3.2.1 Sigmoid Function
 
 h(x) = 1 / (1 + exp(-x))
 
 * Nueral-Net 에서는 Activation Function 으로 Sigmoid Function 을 사용하여 신호를 변환하고, 이 변환된 신호를 다음 뉴런에 전달함
 
 ### 3.2.2 Step Function 구현하기
 
 */

import Accelerate

func step_function(x: Double) -> Double {
    if x > 0.0 {
        return 1.0
    } else {
        return 0.0
    }
}

func step_function(x: [Double]) -> [Double] {
    return x.map { $0 > 0 ? 1.0 : 0.0 }
}

var x = [-1.0, 1.0, 2.0]
var y = step_function(x: x)

/*:

 ### 3.2.3 Step Function 의 그래프
 
 */

var z: Double = 0

for i in stride(from: -5.0, to: 5.0, by: 0.1) {
    z = step_function(x: i)
}

/*:

 ### 3.2.4 Sigmoid Function 구현하기
 
 */

func sigmoid(x: Double) -> Double {
    1.0 / (1.0 + exp(-x))
}

func sigmoid(x: [Double]) -> [Double] {
    x.map { 1.0 / (1.0 + exp(-$0)) }
}

x = [-1.0, 1.0, 2.0]
sigmoid(x: x)

for i in stride(from: -5.0, to: 5.0, by: 0.1) {
    z = sigmoid(x: i)
}

/*:
 
 ### 3.2.5 Sigmoid Function 과 Step Function 비교

 ### 3.2.6 비선형 함수
 
 * Nueral-Net 에서는 Activation Function 으로 비선형 함수를 사용해야 함
 * 선형 함수를 사용하면 Nueral-Net 의 층을 깊게 하는 의미가 없어지기 때문
 * 선형 함수의 문제는 층을 아무리 깊게 해도 '은닉 층이 없는 네트웍' 으로 똑같은 기능을 할 수 있다는 것임
 * 층을 쌓는 혜택을 얻고 싶으면 Activation Function 으로 반드시 비선형 함수를 사용해야 함
 
 ### 3.2.7 ReLU Function
 
 * 최근에는 ReLU (Rectified Linear Unit) Function 을 주로 이용함
 * ReLU 는 입력이 0 을 넘으면 입력을 출력하고, 0 이하면 0 을 출력하는 함수임
 
 */

func relu(x: [Double]) -> [Double] {
    x.map { max(0, $0) }
}

/*:
 
 ## 3.3 다차원 배열의 계산

 ### 3.3.1 다차원 배열
 */

var A1 = [1, 2, 3, 4]

A1.count

var B1 = [[1, 2], [3, 4], [5, 6]]

B1.count

/*:

 ### 3.3.2 행렬의 내적 (행렬 곱)
 */

//
//func dot(a: [[Double]], b: [[Double]]) -> [[Double]] {
//    var result = [[Double]]()
//
//    for i in 0..<a.count {
//        for j in 0..<b[0].count {
//            for k in 0..<a[0].count {
//                result[i][j] += a[i][k] * b[k][j]
//            }
//        }
//    }
//
//    return result
//}
//
//var A2 = [[1.0, 2.0], [3.0, 4.0]]
//var B2 = [[5.0, 6.0], [7.0, 8.0]]
//
//dot(a: A2, b: B2)

/*:
 ### 3.3.3 신경망의 내적

 */
