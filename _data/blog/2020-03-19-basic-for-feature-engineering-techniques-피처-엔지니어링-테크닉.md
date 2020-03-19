---
template: BlogPost
path: /DataPreparation
date: 2020-03-19T05:27:51.882Z
title: Basic for Feature Engineering Techniques(피처 엔지니어링 테크닉)
---
최근에 가장 많이 사용하고 있는 Boosting 계열(LightGBM, XGBoost) 및 Baggin계열을 활용한 분석 프로젝트에 있어서 Key가 되는 새로운 Feature(파생 변수 포함) 찾는 것이 분석의 핵심이라고 말할 수 있다.  업무 도메인에 따라, 데이터 종류에 따라 접근하는 방법은 매번 다를 것이지만, 가장 기본이 되는 테크닉을 몇가지 소개하고자 한다. 



#### Train and Test 동시 처

특정 칼럼들에 대해 Label Encoding을 진행할 때는, Train Data, Test Data를 함께 진행해야 

동일한 Label로 처리가 된다. 


```python
# 1. Train, Test를 Concatenate 해서 처리

df = pd.concat([train[col], test[col], axis=0)
# Label Encoding 작업 수행
train[col] = df[:len(train)]
test[col] = df[len(train):]


# 2. 분리해서 처리하는 방법

df = train
# Label Encoding 작업 수행
df = test 
# Label Encoding 작업 수행
```

#### NAN Processing

LGBM(LigthGBM)을 사용할때 np.nan을 사용하게 되면, NAN을 기준으로 값을 먼저 tree 노드로 분할하하는 작업을 한다. 즉, NAN 왼쪽 오른쪽 트리로 트리구조를 만들게 된다. 그결과, 모든 노드들이 오버피팅되는 결과를 받게 된다. 따라서, NAN의 경우에는 -999와 같이 아주 작은 값으로 변환시켜주는 게 효과적으로 알고리즘의 결과를 도출할 수 있다. 

```python 

df[col].fillna(-999, inplace=True)

```  


#### Label Encode / Factorize/ Memory reduction

Label Encoding(Factorizing)은 String, Category, Object 타입의 칼럼들을 integer로 변환시켜준다. 대량의 데이터를 작업할때 Label Encoding or Factorizing 후 각 칼럼들이 int64 타입으로 변경되는데, 이는 많은 메모리를 점유하게 되는 결과를 초래한다. 필드 Integer 크기에 따라 타입을 변환시켜 준다면, 보다 효율적으로 데이터 처리가 가능할 것이다. 

```python

df[col], _ = df[col].factorize()

if df[col].max() <128: df[col] = df[col].astype('int8')
elif df[col].max() <32768: df[col] = df[col].astype('int16')
else: df[col].astype('int32')

```

물론 memory_reduce라는 함수가 있지만, 일반적으로 더 안전한 방법은 float64 -> float32, int64 -> int32 와 같이 한단계 낮추는 것이다.. (float16은 피하고, int8과 int16은 잘 활용하면 좋다)


#### Categorical Features

Categorical 변수를 사용하면, LGBM에게 클래스 형태로 처리하거나, 숫자형으로 처리(Label Encode를 우선)가 가능하다

```python

df[col] = df[col].astype('category')

```

#### Splitting

하나의 칼럼이 String과 Numeric으로 구성되어 있는 경우 이를 2개 이상으로 분리하여 처리하는 방법이 있다. 예를들어 "Mac OS X 10_9_5" 라는 칼럼을 Operating systemd인 "Max OS X" , Version인 "10_9_5"로 나눌 수 있다. 금액의 경우 "1230.45"인 칼럼을 Dollars "1230", Cents "45" 로 나눈다면 훨씬 좋은 성능을 낼 수 있다. 


Combining/ Transforming/ Interaction

위에서 칼럼을 나누는 방법을 이야기 했다면, 이번에는 칼럼을 합치는 방법이 있다. 

```python

df['uid'] = df['card1'].astype(str) + '_' + df['card2'].astype(str)

```

상관관계가 없는 두 칼럼 들간 LGBM에서 굳이 분리를 안해도 되기 때문에 데이터를 Combine 하는것도 방법이다. Combine 하는 방법에는 Adding, Substracting, Multiplying 등등이 있다.

```python

df['x1_x2' = df['x1'] * df['x2']

```

#### Frequency Encoding

Frequency Encoding은 칼럼값이 Rare인지 Common인지 LGBM이 확인할 수 있는 강력한 방법이다. 

```python

temp = df['card1'].value_counts().to_dict()
df['card1_counts'] = df['card1'].map(temp)

```


#### Aggregations / Group Statistics

LGBM에 Group 통계정보를 제공하면 특정 Group에 대한 Value가 Common인지 Rare인지 알 수 있다. 


```python

temp = df.groupby('card1')['TransactionAmt'].agg(['mean']).rename({'mean':'TransactionAmt_Card1_mean'}, axis=1)

df = pd.merge(df, temp, on='card1', how='left')


```

Normalize/ Standardize




Outlier Removal / Relax / Smooth/ PCA



Chris Deotte의  [Feature Engineering Techinques](https://www.kaggle.com/c/ieee-fraud-detection/discussion/108575) 글을 번역 및 의역한 내용입니다.
