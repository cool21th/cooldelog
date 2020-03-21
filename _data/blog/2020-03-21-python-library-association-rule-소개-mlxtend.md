---
template: BlogPost
path: /machinlearning/associationrule
date: 2020-03-21T14:04:57.223Z
title: '[Python Library] Association Rule 소개(mlxtend)'
thumbnail: /assets/mlxtend.png
---
## overview

추천 알고리즘으로 서비스를 제공하는 목적으로 데이터 간의 상관관계 분석 및 연관분석을 기본으로 수행합니다. 여러 알고리즘중에서 쉽게 적용할 수 있는 라이브러리를 소개하고자 합니다.  

Frequent 패턴 마이닝에서 규칙을 생성하는 것은 일반적인 작업입니다. 연관성 규칙(Rule)은 특정 X, Y 집합들간의 함축적인 관계를 표현하는 것입니다.  
예를들어,마트에서 기저기를 사는 고객이 맥주도 함께 산다는 것은 {Diapers} -> {Beer}로 표현할 수 있게 됩니다. 

이렇게 표현된 내용을 증명하기 위해 [mlxtend (machine learning extensions)](http://rasbt.github.io/mlxtend/)에서 Confidence, Lyft, Leverage, Conviction 값들을 사용합니다. 

이 값에 대한 정의 및 내용 들은 다음과 같습니다.

## Metrics

**1. Support**

> ```
> support(A → C) = support(A U C), range:[0,1]
> ```

> support는 집합 내의 원소들간의 정의를 나타내는 항목으로 연관관계 자체를 의미하지는 않습니다.

> ```
> A = ['Milk', 'Onion', 'Nutmeg', 'Kidney Beans', 'Eggs', 'Yogurt'],
> B = ['Dill', 'Onion', 'Nutmeg', 'Kidney Beans', 'Eggs', 'Yogurt'],
> C = ['Milk', 'Apple', 'Kidney Beans', 'Eggs'],
> D = ['Milk', 'Unicorn', 'Corn', 'Kidney Beans', 'Yogurt'],
> E = ['Corn', 'Onion', 'Onion', 'Kidney Beans', 'Ice cream', 'Eggs']
>
> Egg는 A, B, C, E 집합에 있으므로 support 값은 0.8
> MilK는 A, C, D 집합에 있으므로 support 값은 0.6
> Kidney Beans, Eggs 는 A, B, C, D, E / A, B, C, E  있으므로 0.8
> Onion, Eggs 는 A, B, E / A, B, C, E 이나 Onion이 선행이므로 0.6
> ```


> mlxtend에서의 support 항목은 총 3가지 입니다. (Antecedent support, Consequent support, Support)


> * Antecedent support : 선행한 Transaction의 비율을 의미하며, 위 식에 A에 해당. 
> * Consequent support : 후행한 Transaction의 비율을 의미하며, 위 식에 C에 해당.
> * Support: 위 식의 support(A U C) 에 해당하며, 보통 빈번함 또는 중요성을 의미. 

> frequent itemsets으로 명명하는 집합은 최소 support 임계값보다 큰 support 값을 가지게 됩니다. 일반적으로 downward closure property에 따라 frequent itemsets의 하위 집합 역시 빈번함으로 측정합니다. 


**2. Confidence**

> ```
> confidence(A → C) = support(A → C)/support(A),range: [0,1]
> ```

> confidence는 후행의 Transaction(C)이 선행 Transaction(A)에 연쇄적으로 일어날 확률을 의미하며(1이 최대값), A → C, C → A 의 confidence값은 서로 비대칭인 결과값이 나옵니다. 

**3. Lift**

> ``` 
> lift(A → C) = confidence(A → C) /support(C),range: [0,∞]
> ```

> Lift는 A, C가 독립사건일 때 보다 얼마나 자주 발생했는지를 측정합니다.    A, C가 서로 독립사건이면 Lift 값은 1 로 결정됩니다. 


**4. Leverage**

> ```
> levarage(A → C) = support(A → C) − support(A) × support(C),range: [−1,1]
> ```

> Leverage 는 A, C가 독립사건일 때와 얼마나 다른지를 비교합니다.    A, C가 서로 독립사건이면 값은 0이 됩니다. 


**5. Conviction**

> ```
> conviction(A → C) = 1 − support(C) / 1 − confidence(A → C),range: [0,∞]
> ```

> 높은 Conviction 값은 선행(A)의 의존도가 높은 것을 의미합니다.   예를 들어, confidence 값이 최대(1)이면, 분모가 0이 되어 Conviction 값은 무한대가 되고,    A, C가 서로 독립이면 Lift값과 유사하게 1의 값을 가지게 됩니다. 

내용을 이해하면, 알고리즘을 적용하는데 큰 어려움은 없습니다. 시간이 갈수록 많은 사람들이 보다 쉽게 머신러닝 모델을 만들게 될 것인데, 가장 중요한 것은 기초라고 생각합니다. 

참고: [mlxtend (machine learning extensions)](http://rasbt.github.io/mlxtend/)
