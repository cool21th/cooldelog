---
template: BlogPost
path: /machinelearnin/svdintro
date: 2019-01-01T04:33:41.192Z
title: >-
  [블로그 번역]A Gentle Introduction to Singular-Value Decomposition for Machine
  Learning
---
### Overview

앞서, [Matrix Factorization 기본적인 내용](https://coolog.netlify.com/machinelearning/matrixfactorizationintro)을 기술했었습니다. 이번 블로그에서는 Matrix decomposition(Factorization) 방법 중 SVD(Singular-value Decomposition)에 대해 이야기 하고자 합니다. SVD는 그 어떤 행렬 분해(Eigendecomposition 등 포함)보다 안정적이기에 가장 널리 사용되고 있습니다.

압축, 노이즈제거 및 데이터 축소 등 다양한 응용 프로그램에서 자주 사용됩니다.

1. Singular-value Decomposition

A = U.Sigma.V^T

A: 분해 대상 m x n 행렬
U: m X m 행렬  (A 왼쪽 특이벡터)
Sigma: m X n 대각선 행렬
V^T: n X n 전치행렬 (A 오른쪽 특이 벡터)


2. Calculate Singular-Value Decomposition

Scipy의 svd function을 이용하여 쉽게 구할 수 있다


3. Reconstruct Matrix from SVD

오리지날 행렬은 U, Sigma, V^T로부터 재구성 될 수 있습니다.
그러나 svd() function에서 반환되는 U,s, V^T는 직접 곱할 수 없습니다.

diag() function 을 이용하여 s벡터를 대각선 행렬로 변환해야 합니다. 
이 함수는 오리지날 행렬을 기준으로 m X m인 정사각행렬을 만듭니다.
이로 인해 열과 뒤의 행렬의 행과 맞지 않는 결과를 낼 수 있습니다.

Sigma 대각선 정사각 행렬을 만든후 분할을 해야 합니다.

실제적인 return : U(m X m). Sigma(m X m). V^T(n X n)

원하는 return   : U(m x m). Sigma(m X n). V^T(n X n)

m> n으로 가정하여, 0으로 채워진 m X n 행렬에 diag() 함수를 통해 생성된 n X n 값을 넣어 m X n 형태로 변환한다

4. SVD for Pseudoinverse

Pseudoinverse는 정사각행렬이 아닌 행렬의 역행렬의 일반화 입니다

A^+ = V.D^+. U^T

A^+ : A의 유사 역행렬
D^+ : Sigma의 대각행렬의 유사 역행렬
U^T : U의 전치 행렬



5. SVD for Dimensionality Reduction


B = U.Sigmak.V^Tk



참고자료:\
[A Gentle Introduction to Singular-Value Decomposition for Machine Learning](https://machinelearningmastery.com/singular-value-decomposition-for-machine-learning/)
