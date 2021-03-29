/*:
 # Table of Contents

 Perceptron : 1957년 Frank Rosenblatt 가 고안한 알고리즘
 Perceptron 은 신경망의 기원이 되는 알고리즘
 
 ## 2.1 Perceptron 이란?
 
 ## 2.2 단순한 논리 회로
 
 ### 2.2.1 AND 게이트
 
 ### 2.2.2 NAND 게이트와 OR 게이트
 
 ## 2.3 Perceptron 구현하기
 
 ### 2.3.1 간단한 구현부터
 */

func AND(_ x1: Double, _ x2: Double) -> Double {
    let (w1, w2, theta) = (0.5, 0.5, 0.7)
    let temp = x1 * w1 + x2 * w2
    
    if temp <= theta {
        return 0.0
    } else {
        return 1.0
    }
}

AND(0, 0)
AND(1, 0)
AND(0, 1)
AND(1, 1)

/*:
 ### 2.3.2 Weight 와 Bias 도입
 
 ### 2.3.3 Weight 와 Bias 구현하기
 
 ## 2.4 Perceptron 의 한계
 
 ### 2.4.1 도전! XOR 게이트
 
 ### 2.4.2 션형과 비선형
 
 ## 2.5 다층 Perceptron 이 충돌한다면
 
 ### 2.5.1 기존 게이트 조합하기
 
 ### 2.5.2 XOR 게이트 구현하기
 
 ## 2.6 NAND 에서 컴퓨터까지
 
 ## 2.7 정리
 
 */
