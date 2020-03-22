---
template: BlogPost
path: /machinelearning/eigenvalueintro
date: 2018-12-31T04:43:58.848Z
title: >-
  [블로그 번역] Gentle Introduction to Eigenvalues and Eigenvectors for Machine
  Learning
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
> A: 대상 정사각 행렬(부모행렬) v: 행렬의 eigenvector
> lambda: eigenvalue로 scalar값을 의미

Eigenvalue 공식에 따르면, 정사각행렬 A는 1 개의 eigenvector와 각 차원에 대응하는 eigenvalue 를 갖게 된다. 여기서 주의할 것은 LU Matrix Factorization 처럼 모든 정사각 행렬이 분해 가능한 것은 아닙니다.(일부는 복소수 형태로 분해) A와 같은 부모행렬은 eigenvector와 eigenvalue의 곱으로 표시가 가능해야 합니다. 

> Eigenvalue equation 
>
> ```
> A  = Q . diag(v) . Q^-1
> ```
>
> Q: eigenvector로 구성된 행렬 diag(v): eigenvalue가 대각선으로 구성된 행렬(대문자 lambda로도 표시)
> Q^-1: Q(eigenvector로 구성된 행렬)의 역행렬

행렬 분해는 압축을 수행하는 것과 달리, 특정 행렬연산을 쉽게 수행할 수 있도록 part로 나누는 작업을 합니다. 

데이터 차원을 줄이는 PCA(Principal Component Analysis)와 같은 주성분 분석에도 Eigendecomposition 을 사용합니다. 

### Eigenvectors and Eigenvalues

Eigenvector는 길이 또는 크기(magnitude)가 1.0인 단위 벡터이고, 보통 right vector(Column vector)라고 말합니. 음의 값은 방향이 반대를 의미합니다

> right vector 추가 설명

![rightvector](/assets/rightvector.png "rightvector")

> A: x에 대한 계수 행렬
> x: 미지수 벡터
> b: right vector


![column](/assets/column.png "column")

> 그림을 보면 이와 같이 b(right vector)는 칼럼 벡터만을 의미하는 것을 알수 있다.
> 선형대수 관점에서 보면 row 방향으로 계산하는 방법과 column 방향으로 계산하는 방법 2가지 있는데, row 방향계산은 내적을 의미하고, column 방향계산은 선형결합을 의미한다는 것을 알 수 있다

참고자료: \
[The Geometry of Linear Equations](https://twlab.tistory.com/6?category=668741)
