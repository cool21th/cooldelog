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
> ```
> A . v  = lambda . v
> ```

> A: 대상 정사각 행렬(부모행렬)
> v: 행렬의 eigenvector
> lambda: eigenvalue로 scalar값을 의미


Eigenvalue 공식에 따르면, 정사각행렬 A는 1 개의 eigenvector와 각 차원에 대응하는 eigenvalue 를 갖게 된다. 여기서 주의할 것은 LU Matrix Factorization 처럼 모든 정사각 행렬이 분해 가능한 것은 아닙니다.(일부는 복소수 형태로 분해)
A와 같은 부모행렬은 eigenvector와 eigenvalue의 곱으로 표시가 가능해야 합니다. 

> Eigenvalue equation 
> ```
> A  = Q . diag(v) . Q^-1
> ```
