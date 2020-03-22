---
template: BlogPost
path: /machinelearning/eigenvalueintro
date: 2018-12-31T04:43:58.848Z
title: >-
  [블로그 번역] Gentle Introduction to Eigenvalues and Eigenvectors for Machine
  Learning
thumbnail: /assets/eigenvector.png
---
### Overview

앞서 [Matrix Factorization 기본적인 내용](https://coolog.netlify.com/machinelearning/matrixfactorizationintro)을 기술한 내용을 보면 Matrix decomposition은 복잡한 연산 범위를 단순화 시켜 효율적인 계산을 도와주는 방법인 것을 알 수 있었습니다.  가장많이 사용하는 Matrix decomposition 유형은 eigenvector(고유벡터)와 eigenvalues(고유 값) 분해하는 것입니다. 이러한 형태의 분해는 PCA(Principal Component Analysis)와 같은 머신러닝에서 주요 역할을 합니다. 

### Eigendecomposition of a Matrix

Eigendecomposition(고유 분해)는 **정사각 행렬**을 eigenvector 와 eigenvalues로 분해합니다

> Eigenvalue equation 
>
> ```
> A . v  = lambda . v
> ```
>
> A: 대상 정사각 행렬(부모행렬)\
> v: 행렬의 eigenvector \
> lambda: eigenvalue로 scalar값을 의미

Eigenvalue 공식에 따르면, 정사각행렬 A는 1 개의 eigenvector와 각 차원에 대응하는 eigenvalue 를 갖게 된다. 여기서 주의할 것은 LU Matrix Factorization 처럼 모든 정사각 행렬이 분해 가능한 것은 아닙니다.(일부는 복소수 형태로 분해) A와 같은 부모행렬은 eigenvector와 eigenvalue의 곱으로 표시가 가능해야 합니다. 

이쯤에서 Eigenvector가 가지는 수학적의미를 찾아보면, 위 식의 Eigenvalue Equation은 결국 A.v 는 lambda.v 와 평행하다는 것이고, 크기나 길이만 다른 벡터로 표현된다는 것입니다. v에 여러 수들이 곱해지면, cv, dv 등으로 변해지지만, lambda는 변하지 않은 속성인 eigenvalue 입니다. 

A . v = lambda . v 를 이해하기위해 익숙한 A x = x' 행렬 형태에서 먼저 설명을 시작해 보겠습니다.  

![linearsystem](/assets/Linearsystem.png "linearsystem")

x = \[1, 1] 이면, x' = \[3, 0] 으로 선형변환이 되고, x = \[1, -1] 이면, x' = \[1, 2] 으로 선형변환 됩니다. 즉, x가 가지고 있는 값에 따라 값이 달라지는 것을 알 수 있는 것입니다. 

x 에 어떤 행렬 x'' 대입해서 나온 A x 의 값이 ⍺\[1, 1] , µ\[1, -1]로 처음 대입한 값과 동일한 방향의 벡터가 나온다면, x''은 A의 eigenvector 벡터가 되는 것입니다. 물론 ⍺\[1, 1] 형태가 나올수 없습니다. 왜냐하면, ⍺\[3,0] 형태로 나올 것이니깐요. 즉, 전혀 다른 형태의 x''값이 들어 갈 것입니다. 2개의 eigenvector x1, x2의 실제값은 다음과 같습니다.

![eigenvector1](/assets/eigenvector_1.png "eigenvector1")


보통 n x n 행렬은 n 개의 eigenvector를 갖게 됩니다. 사실 A . cx = lambda (cx) 형태로 본다면, 무수히 많은 eigenvector를 가질 수 있음을 알 수 있습니다. 다만 eigenvalue는 유일합니다. \
여기서 eigenvector 들은 공간 형태(eigenspace)로 표현될 수 있음을 알 수 있습니다. 이에 대한 자세한 내용은 생략하겠습니다 (고유값(eigenvalues)과 고유벡터(eigenvectors) 참고)


> Eigenvalue equation 
>
> ```
> A  = Q . diag(v) . Q^-1
> ```
>
> Q: eigenvector로 구성된 행렬 diag(v): eigenvalue가 대각선으로 구성된 행렬(대문자 lambda로도 표시) Q^-1: Q(eigenvector로 구성된 행렬)의 역행렬

행렬 분해는 압축을 수행하는 것과 달리, 특정 행렬연산을 쉽게 수행할 수 있도록 part로 나누는 작업을 합니다. 

데이터 차원을 줄이는 PCA(Principal Component Analysis)와 같은 주성분 분석에도 Eigendecomposition 을 사용합니다. 

### Eigenvectors and Eigenvalues

Eigenvector는 길이 또는 크기(magnitude)가 1.0인 단위 벡터이고, 보통 right vector(Column vector)라고 말합니. 음의 값은 방향이 반대를 의미합니다

> **right vector 추가 설명**

![rightvector](/assets/rightvector.png "rightvector")

> A: x에 대한 계수 행렬 \
> x: 미지수 벡터 \
> b: right vector 

\
\
![column](/assets/column.png "column")

> 그림을 보면 이와 같이 **b(right vector)**는 칼럼 벡터를 의미하는 것을 알수 있다.
>
> 선형대수 관점에서 보면 row 방향으로 계산하는 방법과 column 방향으로 계산하는 방법 2가지 있는데, row 방향계산은 내적을 의미하고, column 방향계산은 선형결합을 의미한다는 것을 알 수 있다


### Calculation of Eigendecomposition

Numpy의 eig 함수를 써서 계산하는 코드는 다음과 같습니다

> ```python
> # eigendecomposition
> from numpy import array
> from numpy.linalg import eig
> # define matrix
> A = array([[1, 2, 3], [4, 5, 6], [7, 8, 9]])
> print(A)
> # calculate eigendecomposition
> values, vectors = eig(A)
> print(values)
> print(vectors)
> ```


### Confirm an Eigenvector and Eigenvalue

증명하는 코드는 다음과 같습니다. 

> ```python
> # confirm eigenvector
> from numpy import array
> from numpy.linalg import eig
> # define matrix
> A = array([[1, 2, 3], [4, 5, 6], [7, 8, 9]])
> # calculate eigendecomposition
> values, vectors = eig(A)
> # confirm first eigenvector
> B = A.dot(vectors[:, 0])
> print(B)
> C = vectors[:, 0] * values[0]
> print(C)
> ```


### Reconstruct Original Matrix

주어진 eigenvector, eigenvalue 만으로 원래 행렬 A 를 B로 재구성하는 코드입니다.

> ```python
> # reconstruct matrix
> from numpy import diag
> from numpy import dot
> from numpy.linalg import inv
> from numpy import array
> from numpy.linalg import eig
> # define matrix
> A = array([[1, 2, 3], [4, 5, 6], [7, 8, 9]])
> print(A)
> # calculate eigenvectors and eigenvalues
> values, vectors = eig(A)
> # create matrix from eigenvectors
> Q = vectors
> # create inverse of eigenvectors matrix
> R = inv(Q)
> # create diagonal matrix from eigenvalues
> L = diag(values)
> # reconstruct the original matrix
> B = Q.dot(L).dot(R)
> print(B)
> ```

참고자료: \
[Gentle Introduction to Eigenvalues and Eigenvectors for Machine Learning](https://machinelearningmastery.com/introduction-to-eigendecomposition-eigenvalues-and-eigenvectors/)\
[The Geometry of Linear Equations](https://twlab.tistory.com/6?category=668741)\
[고유값(eigenvalues)과 고유벡터(eigenvectors)](https://twlab.tistory.com/46)
