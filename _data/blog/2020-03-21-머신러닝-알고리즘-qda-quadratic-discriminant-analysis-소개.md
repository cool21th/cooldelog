---
template: BlogPost
path: /machinelearning/qda
date: 2020-03-21T15:30:42.005Z
title: '[머신러닝 알고리즘] QDA(Quadratic Discriminant Analysis) 소개'
thumbnail: /assets/qda.jpg
---
합성데이터를 만드는 일반적인 것중 하나는 두개의 다른 Multivariate Gaussian 분포를 가지고 만드는 것입니다. 

> Multivariate Gaussian 설명
>
> 1. variate : 허용이 가능한 어떤 집합이나 데이터의 그룹내에서 어떤 확률에 따라 데이터가 자유롭게 사용되는 변수
> 2. multivariate: 어느 집합에서만 추출되는 멀티 변수.
>
> A(비정상), B(정상) 데이터들을 각각의 분포를 합쳐서 2차원 평면으로 보게 되면, 정규분포를 형성할 가능성이 높아, Anomaly Detection이 힘들게 된다.  이러한 현상을 해결/ 보완하기 위해 Multivariate Gaussian 이 도입되었다. 
>
> Multivarate Gaussian Distribution은 간단하게 정리하면, Gaussian Distribution이 두개이상 있는 분포인 것이다.  통계학적인 설명은 생략 (Multivariate Gaussian 정리 참고)

하나의 분포에서 선택한 관측된 값의 target을 1, 다른 분포에서 관측된 값을 target을 0으로 Labeling 하고, 이러한 합성데이터에서 가장 적합한 분류기인 Quadratic Discriminant Analysis(QDA)를 사용합니다. 



QDA 설명

![qda_known](/assets/qda_known.png "qda_known")

> QDA는 한글로 해석하면 이차 판별 분석법으로, LDA(Linear discriminant analysis)와 함께 대표되는 확률론적 생성모형이다. 베이즈정리를 사용해서, y클래스 값에 따른 x의 분포 정보를 기준으로 x에 대한 y의 확률 분포를 찾아낸다. 
>
> 다만, x는 실수이고, 확률 분포는 다변수 정규분포로 가정하고 있기 때문에 x의 분포의 위치와 형태는 y클래스에 따라 달라진다. 통계학적 설명은 생략(선형판별분석법과 이차판별분석법 참고)

![qda_unknown](/assets/qda_unknown.png "qda_unknown")

QDA 는 Multivariate Gaussian Distribution의 여러 타원체에서 예측 대상 데이터가 들어오면, 그 데이터가 어느 타원체에 속하는지를 판별해줍니다. 

> 변수가 2개인(P0,P1 서로 독립변수로 가정) 경우 설명
>
> Pr(target=1 given X) = P1/(P0+P1)
>
> P1 = Pr(X is in ellipsoid 1 given X) P0 = Pr(X is in ellipsoid 0 given X)
> Pr mean Probability.

QDA는 가우스 분포와 편차를 최소화 하는 방법으로 각 클래스에 대해 공분산 행렬(Covariance Matrix)을 추정합니다.  40개의 변수와 2개의 클래스가 있으면, 1640개의 Parameter를 가지게 될 것이고, 공분산 행렬 추정하는 방법을 통해 유의미한 변수들을 추립니다.

> 공분산 행렬(Covariance Matrix) 설명
>
> 변수가 여러개(Multivariate)인 데이터에서는 변수간의 상관성(Correlation)이 매우 중요한 요소가 된다.  예를들어 두개 변수의 방향이 같이 움직이면 (A가 증가할때, B도 증가/ A가 감소 할때 B 도 감소) 상관계수가 양일 가능성이 크다
>
> 공분산(Covariance)은 상관계수와 밀접한 관계를 가지고 있는데,  공분산을 X, Y의 표준편차로 나누어 표준화 한 값이, X 와 Y간의 상관관계라고 할 수 있습니다. 
>
> 통계학적 설명은 생략(기초 행렬연산 참고)

참고:\  
[QDA Explained](https://www.kaggle.com/c/instant-gratification/discussion/93843)\
[대소니 이상탐지: 또다른 알고리즘](https://daeson.tistory.com/218)\
[Multivariate Gaussian 정리](https://www.sallys.space/blog/2018/03/20/multivariate-gaussian/) [선형판별분석법과 이차판별분석법](https://datascienceschool.net/view-notebook/2c6a8e003219446995f3d866ac8a6fd1/)\
[Classification, LDA](http://web.stanford.edu/class/stats202/content/lec9.pdf)\
[기초 행렬연산)(https://ratsgo.github.io/linear%20algebra/2017/03/14/operations/)
