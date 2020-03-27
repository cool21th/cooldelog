---
template: BlogPost
path: /machinelearnin/sklearn_intro
date: 2019-10-27T17:23:00.000Z
title: '[블로그 번역] Scikit-learn for Machine learning'
thumbnail: /assets/sklearn.png
---

### Overview

[A Gentle Introduction to Scikit-Learn: A Python Machine Learning Library](https://machinelearningmastery.com/a-gentle-introduction-to-scikit-learn-a-python-machine-learning-library/)

머신러닝 사용여부를 떠나서, python 코딩을 하는 사람들 중 Scikit-learn이라는 Library를 못본 사람은 없었을 것입니다.

그만큼 Scikit-learn은 머신러닝 개발에서 부터 운영까지 충족시켜주는 아주 강력한 라이브러리입니다.
Scikit-learn의 소개와 기본적인 기능에 대해 먼저 소개하고 이후 다양한 사용방법에 대한 내용을 다루고자 합니다.


### Scikit-learn 소개

Scikit-learn 은 다양한 supervised, unsupervised Learning 알고리즘을 제공하는 Python Library 입니다

BSD 라이센스 정책을 따르며, 학업적/ 상업적으로 사용하는데 있어서 제약이 없습니다. 

Scikit-learn Library는 SciPy(Scientific Python)을 기반으로 구성되어 있으며, 
Numpy, SciPy, Matplotlib, IPython, Sympy, Pandas 등의 라이브러리들도 함께 구성하고 있습니다. 

- Numpy: Base n-dimensional array package
- SciPy: Fundamental library for scientific computin
- Matplotlib: Comprehensive 2D/3D plotting
- Sympy : Symbolic mathematics
- Pandas : Data Structure and analysis

SciKits은 SciPy 라이브러리의 확장형 모듈을 의미합니다. 
그래서 scikit-learn 은 SciPy 모듈과 learning 알고리즘을 제공하는 모듈이라고 쉽게 이해할 수 있습니다. 

주 사용 언어는 Python이지만, c 라이브러리들을 활용해서 LAPACK, LibSVM, cython을 활용하여 
배열 및 행렬 연산에 있어서 Numpy와 같은 성능을 발휘합니다. 


### Scikit-learn Feature

Scikit-learn 은 Numpy, Pandas와 다르게 데이터를 로딩, 조작, 요약하는데 중점을 두고 있지 않습니다. 

주요 제공하는 모델은 다음과 같습니다. 

- Clustering: KMeans 와 같이 Unlabeled 된 데이터 그룹핑 기능 제공(흔히 비지도 학습의 대표적)
- Cross Validation: unseen data(테스트 or 실제 들어올 데이터 등)에 대한 supervised model의 성능 측정 기능 제공
- Datasets : 모델의 추이를 확인 목적으로 특정 속성을 가진 테스트/ 생성용 데이터 세트 관련 기능제공
- Dimensionality Reduction: PCA(Principal component analyis)와 같이 요약, 시각화 및 feature 선택을 목적으로 데이터 속성(차원 등) 감소
- Ensemble methods: supervised 모델들의 예측 조합 기능 지원
- Feature extraction: 이미지, 텍스트 데이터 속성 정의
- Feature selection: supervised 모델들로부터 의미있는 속성 정의
- Parameter Tuning: supervised 모델의 최선의 예측결과 도출
- Manifold Learning: 복잡한 다차원의 데이터 요약및 묘사
- Supervised Learning: Linear, 나이브베이즈, Decision Tree, SVM, nn 등 다양한 모델 지원

### Example: Classification and Regression trees

```python
# Sample Decision Tree Classifier
from sklearn import datasets
from sklearn import metrics
from sklearn.tree import DecisionTreeClassifier
# load the iris datasets
dataset = datasets.load_iris()
# fit a CART model to the data
model = DecisionTreeClassifier()
model.fit(dataset.data, dataset.target)
print(model)
# make predictions
expected = dataset.target
predicted = model.predict(dataset.data)
# summarize the fit of the model
print(metrics.classification_report(expected, predicted))
print(metrics.confusion_matrix(expected, predicted))

```



### Example:[Data Rescaling](https://machinelearningmastery.com/rescaling-data-for-machine-learning-in-python-with-scikit-learn/)

데이터 속성에서 달러, 킬로그램 및 판매량과 같은 다양한 수량에 대한 비율이 혼합된 속성이 포함될 수 있습니다. 
기준이 서로 다르기 때문에 이러한 경우 데이터를 2가지 방법을 주로 사용합니다. Normalization과 Standardization 입니다. 


1. Data Normalization

> Normalization은 numeric 속성인 데이터를 0과 1범위안으로 rescaling 하는 방법입니다. 
> kNN 과 같이 distance에 의존하는 모델에 적합한 방법입니다. 

> ```python
> 
> # Normalize the data attributes for the Iris dataset.
> from sklearn.datasets import load_iris
> from sklearn import preprocessing
> # load the iris dataset
> iris = load_iris()
> print(iris.data.shape)
> # separate the data from the target attributes
> X = iris.data
> y = iris.target
> # normalize the data attributes
> normalized_X = preprocessing.normalize(X)
> 
> ```



2. Data Standardization

> Standardization은 평균이 0이고 표준 편차가 1이 되도록 각 속성의 의 분포를 이동시키는 방법입니다. 
> Gaussian 을 활용한 모델링에 적합한 방법입니다. 
> 
> ```python
> 
> # Standardize the data attributes for the Iris dataset.
> from sklearn.datasets import load_iris
> from sklearn import preprocessing
> # load the Iris dataset
> iris = load_iris()
> print(iris.data.shape)
> # separate the data and target attributes
> X = iris.data
> y = iris.target
> # standardize the data attributes
> standardized_X = preprocessing.scale(X)
> 
> ```



### [Feature Selection](https://machinelearningmastery.com/feature-selection-machine-learning-python/)

Feature selection은 예측 변수 또는 도출하고 싶은 결과에 기여하는 feature들을 선택하는 방법입니다. 

데이터 분석 or 모델링을 하는데 있어서 관련없는 feature가 많으면 성능은 당연히 떨어집니다. 
Feature selection을 통해 얻는 이점은 다음과 같이 3가지 입니다. 

- Reduces Overfitting: 불필요한 데이터를 줄여 노이즈를 줄입니다. 
- Improve Accuracy: 잘못된 데이터를 줄여 모델의 정확도를 높입니다. 
- Reduce Training: 필요한 데이터만을 가지고 모델을 빠르게 훈련시킵니다. 

이를 위해 Scikit-learn에서 가장 기본적인 방법은 Recursive Feature Elimination 과 Feature importance ranking 이 있고,
선형대수를 적용한 Principal Component Analysis, 변수 하나에 대한 통계학적인 접근(Univariate Selection) 등이 있습니다



1. Recursive Feature Elimination

> Recursive Feature Elimination 은 모델의 속성을 재귀적으로 제거하고, 모델링함으로써 정확도가 가장 높은 속성 조합을 찾아냅니다. 
> 
> ```python
> 
> # Recursive Feature Elimination
> from sklearn import datasets
> from sklearn.feature_selection import RFE
> from sklearn.linear_model import LogisticRegression
> # load the iris datasets
> dataset = datasets.load_iris()
> # create a base classifier used to evaluate a subset of attributes
> model = LogisticRegression()
> # create the RFE model and select 3 attributes
> rfe = RFE(model, 3)
> rfe = rfe.fit(dataset.data, dataset.target)
> # summarize the selection of the attributes
> print(rfe.support_)
> print(rfe.ranking_)
> ```

2. Feature Importance

> Feature Importance는 decision tree 기반의 앙상블모델(Random Forest or extra trees)들의 상대적인 중요 속성을 찾는데 사용합니다. 
> 
> ```python
> 
> # Feature Importance
> from sklearn import datasets
> from sklearn import metrics
> from sklearn.ensemble import ExtraTreesClassifier
> # load the iris datasets
> dataset = datasets.load_iris()
> # fit an Extra Trees model to the data
> model = ExtraTreesClassifier()
> model.fit(dataset.data, dataset.target)
> # display the relative importance of each attribute
> print(model.feature_importances_)
> ```



3. Principal Component Analysis

> PCA는 선형대수를 사용하여 데이터들을 압축한 형태로 변환합니다.
> 
> 일반적으로 Data reduction 기술이라고 말하고, PCA 결과로 차원의 수 또는 구성 요소들을 선택할 수 있습니다. 
> 
> ```python
> # Feature Extraction with PCA
> import numpy
> from pandas import read_csv
> from sklearn.decomposition import PCA
> # load data
> url = "https://raw.githubusercontent.com/jbrownlee/Datasets/master/pima-indians-diabetes.csv"
> names = ['preg', 'plas', 'pres', 'skin', 'test', 'mass', 'pedi', 'age', 'class']
> dataframe = read_csv(url, names=names)
> array = dataframe.values
> X = array[:,0:8]
> Y = array[:,8]
> # feature extraction
> pca = PCA(n_components=3)
> fit = pca.fit(X)
> # summarize components
> print("Explained Variance: %s" % fit.explained_variance_ratio_)
> print(fit.components_)
> ```



4. Univariate Selection

> Output 변수와 가장 밀접한 feature를 찾는 방법론입니다. 
> Scikit-learn에서는 SelectKBest 를 제공해 다양한 통계적 기법과 병행해서 사용할 수 있도록 도와줍니다. 
> 이번 예시는 ANOVA F-value 를 통해 통계적인 테스트 스캔을 사용합니다. 
> 
> 
> ```Python
> # Feature Selection with Univariate Statistical Tests
> from pandas import read_csv
> from numpy import set_printoptions
> from sklearn.feature_selection import SelectKBest
> from sklearn.feature_selection import f_classif
> # load data
> filename = 'pima-indians-diabetes.data.csv'
> names = ['preg', 'plas', 'pres', 'skin', 'test', 'mass', 'pedi', 'age', 'class']
> dataframe = read_csv(filename, names=names)
> array = dataframe.values
> X = array[:,0:8]
> Y = array[:,8]
> # feature extraction
> test = SelectKBest(score_func=f_classif, k=4)
> fit = test.fit(X, Y)
> # summarize scores
> set_printoptions(precision=3)
> print(fit.scores_)
> features = fit.transform(X)
> # summarize selected features
> print(features[0:5,:])
> 
> ```

회귀모델에는 보통 Recursive Feature Elimination 방법을, tree기반 앙상블 모델에는 feature importace 방법을 사용합니다. 




### [Algorithm Parameter tuning](https://machinelearningmastery.com/how-to-tune-algorithm-parameters-with-scikit-learn/)


머신러닝 알고리즘에서 parameter tuning은 모델링의 마지막 단계에서 수행됩니다. 
이러한 과정을 Hyperparameter optimization이라고 합니다.(딥러닝의 하이퍼파라미터와는 다릅니다)

Scikit-learn에서는 hyper parameter 튜닝전략을 grid search와 random search 두가지 제시합니다. 



1. Grid Search Parameter Tuning

> Grid Search는 알고리즘 파라미터 조합을 grid 형식으로 모델에 적용해 평가하는 방법입니다. 
> 
> ```python
> 
> # Grid Search for Algorithm Tuning
> import numpy as np
> from sklearn import datasets
> from sklearn.linear_model import Ridge
> from sklearn.model_selection import GridSearchCV
> # load the diabetes datasets
> dataset = datasets.load_diabetes()
> # prepare a range of alpha values to test
> alphas = np.array([1,0.1,0.01,0.001,0.0001,0])
> # create and fit a ridge regression model, testing each alpha
> model = Ridge()
> grid = GridSearchCV(estimator=model, param_grid=dict(alpha=alphas))
> grid.fit(dataset.data, dataset.target)
> print(grid)
> # summarize the results of the grid search
> print(grid.best_score_)
> print(grid.best_estimator_.alpha)
> 
> ```




2. Random Search Parameter Tuning

> Random search는 Random distribution으로 부터 알고리즘의 parameter들을 샘플링해서 튜닝하는 방법입니다. 
> 
> ```python
> 
> # Randomized Search for Algorithm Tuning
> import numpy as np
> from scipy.stats import uniform as sp_rand
> from sklearn import datasets
> from sklearn.linear_model import Ridge
> from sklearn.model_selection import RandomizedSearchCV
> # load the diabetes datasets
> dataset = datasets.load_diabetes()
> # prepare a uniform distribution to sample for the alpha parameter
> param_grid = {'alpha': sp_rand()}
> # create and fit a ridge regression model, testing random alpha values
> model = Ridge()
> rsearch = RandomizedSearchCV(estimator=model, param_distributions=param_grid, n_iter=100)
> rsearch.fit(dataset.data, dataset.target)
> print(rsearch)
> # summarize the results of the random parameter search
> print(rsearch.best_score_)
> print(rsearch.best_estimator_.alpha)
> 
> ```


참고자료: \
[scikit-learn homepage](https://scikit-learn.org/)\
[scikit-learn github page](https://github.com/scikit-learn)

