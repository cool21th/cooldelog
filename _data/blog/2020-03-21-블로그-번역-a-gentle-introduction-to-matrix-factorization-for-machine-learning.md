---
template: BlogPost
path: /machinelearning/matrixfactorizationintro
date: 2018-12-29T16:22:34.564Z
title: '[블로그 번역]A Gentle Introduction to Matrix Factorization for Machine Learning'
thumbnail: /assets/matrixfactorization.png
---
### Overview

Matrix Factorization(행렬 분해)는 한정된 컴퓨팅 파워를 가지고 최대 효율을 누릴 수 있는 방법입니다. 복잡한 행렬 연산을 쉽게 계산할 수 있도록 구성부분으로 줄임으로써 가능합니다.
대수학(linear algebra), 역함수 등이 주된 계산을 합니다.

### Matrix Decomposition

Matrix decomposition은 행렬 그대로 복잡한 계산을 수행하는 것이 아니라, 행렬을 분해해서 계산을 단순하게 만드는 방법입니다. 
가장 적합한 예로 10을 2 x 5로 변환하는 숫자의 인수분해를 들수 있습니다. 그래서 Matrix Factorization 이라고 부르기도 합니다.
가장많이 쓰이는 방법은 LU, QR Matrix Decomposition이다.

### LU Matrix Decomposition

LU decomposition은 정사각행렬을 위한 것으로, L 과 U 성분으로 분해합니

> ```
> A = L.U
> ```

> A: 분해 대상 정사각 행렬(square matrix) \
> L: 아래 삼각형 행렬(lower triangle matrix)\
> U: 위 삼각형 행렬(higer triangle matrix)

주의할 점은 LU 분해는 반복적인 수치 프로세스를 사용하며, 분해되지 않거나 쉽게 분해되는 경우에는 사용할수 없습니다

실제적으로 더 안정된방법은 LUP분해(부분피벗을 사용한 LU분해)입니다.

> ```
> A = P.L.U
> ```


P 는 결과를 치환하거나 원래 순서로 반환하는 방법을 지정합니다 \
LU분해는 주로 선형회귀 분석에서 계수를 찾거나, 대수에서 해를 단순화 하는데 사용됩니다 \

**python code**

```python
# LU decomposition
from numpy import array
from scipy.linalg import lu
# define a square matrix
A = array([[1, 2, 3], [4, 5, 6], [7, 8, 9]])
print(A)
# LU decomposition
P, L, U = lu(A)
print(P)
print(L)
print(U)
# reconstruct
B = P.dot(L).dot(U)
print(B)
```


### QR Matrix Decomposition

QR Decomposition 은 m X n 행렬에 대한 것으로 행렬을 Q, R 구성 요소로 분해합니다. 즉, 원래 행렬이 정사각 행렬이 아니어도 됩니다 


> ```
> A = Q.R
> ```

> A: 분해 대상 행렬 Q: 크기가 m X m 인 행렬 \
> R: 크기가 m X n 인 상단 삼각 행렬

**python code**

```python
# QR decomposition
from numpy import array
from numpy.linalg import qr
# define a 3x2 matrix
A = array([[1, 2], [3, 4], [5, 6]])
print(A)
# QR decomposition
Q, R = qr(A, 'complete')
print(Q)
print(R)
# reconstruct
B = Q.dot(R)
print(B)
```

### Cholesky Decomposition

Cholesky Decomposition은 모든 고유 값(eigen value)이 0보다 큰 정사각행렬에 대한 것입이다.\
실수에 대한 분해에 중점으로 수행합니다(복소수 사례 무시)

> ```
> A = L.L^T
> ```

> L: lower triangle matrix \
> L^T: transpose of L

> ```
> A = U^T.U
> ```

> L: lower triangle matrix \
> U: Upper triangle matrix


Cholesky decomposition은 선형회귀에 대한 최소제곱근 및 시뮬레이션/ 최적화 방법에 사용하며, LU 보다 2배 가까이 좋은 효율을 가지고 있습니다 


**python code**

```python
# Cholesky decomposition
from numpy import array
from numpy.linalg import cholesky
# define a 3x3 matrix
A = array([[2, 1, 1], [1, 2, 1], [1, 1, 2]])
print(A)
# Cholesky decomposition
L = cholesky(A)
print(L)
# reconstruct
B = L.dot(L.T)
print(B)
```

