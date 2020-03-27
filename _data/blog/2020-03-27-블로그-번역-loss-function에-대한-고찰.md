---
template: BlogPost
path: /machinelearning/loss_func
date: 2020-03-27T17:30:55.199Z
title: '[블로그 번역] Loss function에 대한 고찰'
thumbnail: /assets/crossentropy.png
---

# Cross Entropy
[A Gentle Introduction to Cross-Entropy for Machine Learning](https://machinelearningmastery.com/cross-entropy-for-machine-learning/)


### Overview

Cross-entropy는 머신러닝에서 가장 일반적인  loss function 입니다. 

Cross-entropy는 정보이론(Information field) 분야에서 **두 확률 분포 사이의 차이를 계산하는 측정값** 을 의미합니다.
두 확률 분포간에 상대적인 entropy 값을 계산하는 것이 KL diivergence 와 유사하지만, Cross-entropy에서는 Totla entropy를 계산한다는 부분이 다른점입니다

Cross-entropy는 Logistic loss(log loss)와 유사하면서도 다릅니다.
classification에서의 loss function으로 사용될 때, 서로다른 대상 데이터에서 다른 측정값이 도출되더라도, 
두 측정값은 같은 수량으로 계산하여, 서로 바꿔 사용할 수 있습니다.


### Entropy 
[A Gentle Introduction to Information Entropy](https://machinelearningmastery.com/what-is-information-entropy/)

정보이론(Information Theory)은 노이즈가 많은 채널을 통해 데이터를 전송할때 사용하는 수학의 한 분야 입니다. 

정해진 메세지가 보유하고 있는 정보를 수량으로 측정하기 위해 시작한 아이디어로, 
Event 와 엔트로피(called random variable)를 정량화 하고, 확률로 계산이 가능합니다. 

Information 과 Entropy를 계산하는 것은 머신러닝에 아주 유용합니다. 
가장 기본적인 feature selection에서 부터, decision tree 모델링, classification 모델링 등에 사용합니다. 
따라서, 머신러닝 전문가는 Information과 entropy에 대해 정학히 알아야 합니다.

##### Information Theory

정보이론(Information Theory)은 데이터 압축 및 신호 처리와 같이 통신 분야와 밀접한 관계를 가진 수학의 한 분야 입니다. 

Information은 event, variables와 distribution(분포) 등에 대한 정보의 양을 정량화하는 것이 기본개념입니다. 

정보의 양을 정량화는 확률을 가지고 사용해야 하기 때문에 서로간의 관계 정의가 필요하며,
정보의 측정은 통신을 넘어서 AI, 머신러닝 분야까지 넓게 사용되고 있습니다. 


##### Calculate the Information for an Event

정보를 정량화 한다는 것은 낮은 확률의 이벤트는 높은 정보를, 높은 확률의 이벤트는 낮은 정보를 갖는다는 데서 시작합니다. 

discrete event x는 다음과 같은 공식을 따릅니다. 

information(x) = -log(p(x)) 

log 는 밑이 2인 것을 의미하며, 그 선택기준은 정보 측정단위가 비트(이진수) 이기 때문입니다. 
이것은 정보 처리 의미에서 이벤트를 나타내는 데 필요한 비트 수로 직접 해석이 가능합니다. 

log 함수에서 -가 붙여진 것은 x의 범위가 0~ 1사이로 항상 그 값은 양수를 의미하는 것입니다. 


example) 동전에 대한 정보 값

```python

# calculate the information for a coin flip
from math import log2
# probability of the event
p = 0.5
# calculate information for event
h = -log2(p)
# print the result
print('p(x)=%.3f, information: %.3f bits' % (p, h))

```

example) 주사위에 대한 정보 값

```python

# calculate the information for a dice roll
from math import log2
# probability of the event
p = 1.0 / 6.0
# calculate information for event
h = -log2(p)
# print the result
print('p(x)=%.3f, information: %.3f bits' % (p, h))

```

example) 확률과 정보 간의 관계


```python


# compare probability vs information entropy
from math import log2
from matplotlib import pyplot
# list of probabilities
probs = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]
# calculate information
info = [-log2(p) for p in probs]
# plot probability vs information
pyplot.plot(probs, info, marker='.')
pyplot.title('Probability vs Information')
pyplot.xlabel('Probability')
pyplot.ylabel('Information')
pyplot.show()

```

##### Calculate the Entropy for a Random Variable


Entropy는 Random variable에 대한 확률 분포로 도출 된 이벤트를 나타내거나 전송하는데 필요한 비트수를 의미합니다. 


H(x) = -sum(each k in K p(k)*log(p(k))

각 이벤트의 발생 확률과 비트수를 곱한 값의 합의 - 값으로 표현됩니다. 

example) original python

```python

# calculate the entropy for a dice roll
from math import log2
# the number of events
n = 6
# probability of one event
p = 1.0 /n
# calculate entropy
entropy = -sum([p * log2(p) for _ in range(n)])
# print the result
print('entropy: %.3f bits' % entropy)

```

example) SciPy

```python

# calculate the entropy for a dice roll
from scipy.stats import entropy
# discrete probabilities
p = [1/6, 1/6, 1/6, 1/6, 1/6, 1/6]
# calculate entropy
e = entropy(p, base=2)
# print the result
print('entropy: %.3f bits' % e)

```

example) entropy 와 probability 간 관계

```python

# compare probability distributions vs entropy
from math import log2
from matplotlib import pyplot
 
# calculate entropy
def entropy(events, ets=1e-15):
	return -sum([p * log2(p + ets) for p in events])
 
# define probabilities
probs = [0.0, 0.1, 0.2, 0.3, 0.4, 0.5]
# create probability distribution
dists = [[p, 1.0 - p] for p in probs]
# calculate entropy for each distribution
ents = [entropy(d) for d in dists]
# plot probability distribution vs entropy
pyplot.plot(probs, ents, marker='.')
pyplot.title('Probability Distribution vs Entropy')
pyplot.xticks(probs, [str(d) for d in dists])
pyplot.xlabel('Probability Distribution')
pyplot.ylabel('Entropy (bits)')
pyplot.show()

```





### Cross-Entropy

지금까지 Entropy에 대해 이야기 했고, 이제는 Cross-entropy에 대해 좀더 자세하게 이야기 해보겠습니다. 

Information(정보) 이론에서는 event를 인코딩하고 전송하기위해 비트수로 그 양을 보여줍니다. 
확률이 적은 이벤트는 보다 많은 정보(비트수)를 가지고 있고, 확률이 높은 이벤트는 적은 정보(비트수)를 가지고 있습니다.

Entropy는 확률 분포에서 전송할 비트를 무작위로 뽑은 수를 말합니다. 
비대칭 분포는 엔트로피가 낮지만, 정규분포를 따를 경우에는 엔트로피가 큽니다. 
즉, 비대칭 분포에서 많은 정보수를 가진다는 것을 알 수 있습니다. 

Entropy의 식은 다음과 같이 이벤트가 발생할 확률에 대한 비트수를 곱한 값으로 표현됩니다. 

H(X) = -sum x in X P(X) * log(P(x))

Cross-entropy는 두개의 분포(P, Q)에서 위치한 이벤트의 평균을 계산한 Entropy에서 아이디어를 착안하였습니다. 

H(P, Q) = -sum x in X P(x) * log(Q(x))

결과는 비트 단위로 측정된 양수이며, 두 확률 분포가 동일한 경우  분포값은 Entropy와 동일합니다. 



### Cross-Entropy versus KL Divergence

Cross-entropy 와 KL Divergence는 동일하지 않습니다. 
KL(Kullback-Leibler) divergnece는 전체 비트수가 아닌, P대신에 Q로 메세지를 나타내는데 필요한 평균 추가 비트수 측정합니다.


비교하기에 앞서, KL Divergence에 대해 좀더 이야기 해보고자 합니다. 

##### [How to caculate the KL Divergence for Machine Learning](https://machinelearningmastery.com/divergence-between-probability-distributions/)

주어진 Random variable의 확률분포간의 차이를 정량화하는 것은 매우 중요한데,
머신러닝을 다루다 보면 실제 확률분포와 관측된 확률분포의 차이를 계산하는 경우가 종종 생기기 때문입니다. 

이런 부분은 KL Divergence나 relative entropy, Jensen-Shannon Divergence(KL Divergence의 정규화된 버전)를 통해 해결합니다.
KL Divergence 에 대해 이야기 하기 전에 통계학적 거리(Statistical Distance)에 대해 먼저 알아보겠습니다. 

1. Statistical Distance

> 앞서 이야기한 것처럼, 두개의 확률분포를 비교하는 상황은 종종 발생합니다. 
> 단일 변수에 대한 확률분포, 두개 변수에 대한 확률분포를 가지는 케이스를 말합니다. 
> 두 변수간의 차이를 정량화 하는 것이 중요한데 일반적으로 두 통계 객체간의 Statistical Distance 로 계산합니다.
> 방법은 **두 분포사이에 거리를 구하는 방법**과 **두 확률 분포 사이에 divergence를 구하는 방법**이 있는데, 전자의 경우 결과 해석에 어려움이 있어 후자를 보통 많이 선택합니다.
> Divergence는 두 분포(P, Q)간의 서로의 score(P-> Q score, Q-> P score)로 서로 얼마나 다른 지에 대해 scoring 합니다. 
> Divergence score는 Information 이론 과 머신러닝에서 다양한 계산을 위한 중요한 방법입니다.
> 예를들면 Mutual Information과 Classification 모델링의 loss function으로 사용하는 cross-entropy 와 같은 점수를 계산하는데 중요한 이론입니다. 
> GAN(Generative Advesarial Network)모델을 최적화 할때에도 목표(target) 확률 분포를 근사화 하는등 복잡한 모델링을 이해하는 데 직접적으로 사용합니다.
> Information 이론에서 주로 사용하는 Divergence score는 KL Divergence와 Jensen-Shannon Divergence 입니다. 


2. Kullback-Leibler Divergence

> KL Divergence는 두 확률분포간에 서로 얼마나 다른지를 정량화 해서 점수를 매기는 방법입니다. 
> P와 Q 간의 Divergence는 KL(P || Q)로 표기를 합니다.
> 
> KL(P || Q) = -sum x in X P(x) * log(Q(x)/P(x)) = sum x in X P(x) * log(P(x)/Q(x))
>
> KL divergence는 각 이벤트가 P에서 발생할 확률과 Q에 발생한 것을 P로 나눈후 log를 취한 값을 곱한 결과들의 합을 -로 취한 결과입니다. 
> log 안에 P(X)와 Q(x)를 분모분자를 바꾸어놓은 결과도 로그 특성상으로 동일합니다 
>
> x값이 P에서 크면서 Q에서 작거나, P가 작으면서 Q가 크면 발산을 가진다는 것을 의미합니다.
> 전자의 경우는 매우 큰 발산을 의미하고, 후자는 전자보다 작지만 이또한 발산을 의미합니다.

> 이를 통해, KL(P || Q) != KL(Q || P) 인 것을 알 수 있습니다. 

> 구하고자 하는것은 P의 확률분포이고, Q는 P의 근사치 인 것을 알 수 있습니다. 

> KL divergence를 코드로 확인하도록 하겠습니다. 

> - 이벤트 별 확률 확인

```python
# plot of distributions
from matplotlib import pyplot
# define distributions
events = ['red', 'green', 'blue']
p = [0.10, 0.40, 0.50]
q = [0.80, 0.15, 0.05]
print('P=%.3f Q=%.3f' % (sum(p), sum(q)))
# plot first distribution
pyplot.subplot(2,1,1)
pyplot.bar(events, p)
# plot second distribution
pyplot.subplot(2,1,2)
pyplot.bar(events, q)
# show the plot
pyplot.show()

```

> - KL divergence 를 통한 결과값 확인(original Python)

```python

# example of calculating the kl divergence between two mass functions
from math import log2
 
# calculate the kl divergence
def kl_divergence(p, q):
	return sum(p[i] * log2(p[i]/q[i]) for i in range(len(p)))
 
# define distributions
p = [0.10, 0.40, 0.50]
q = [0.80, 0.15, 0.05]
# calculate (P || Q)
kl_pq = kl_divergence(p, q)
print('KL(P || Q): %.3f bits' % kl_pq)
# calculate (Q || P)
kl_qp = kl_divergence(q, p)
print('KL(Q || P): %.3f bits' % kl_qp)

```

> - SciPy를 통한 KL divergence 확인
> 
> SciPy에서 KL divergence를 relative entropy 함수로 계산할수 있습니다. 
> 다른 점은 log 밑이 2가 아닌 자연로그로 계산합니다.



```python

# example of calculating the kl divergence (relative entropy) with scipy
from scipy.special import rel_entr
# define distributions
p = [0.10, 0.40, 0.50]
q = [0.80, 0.15, 0.05]
# calculate (P || Q)
kl_pq = rel_entr(p, q)
print('KL(P || Q): %.3f nats' % sum(kl_pq))
# calculate (Q || P)
kl_qp = rel_entr(q, p)
print('KL(Q || P): %.3f nats' % sum(kl_qp))

```


> 3. Jensen-Shannon Divergence

> JS Divergence는 두 확률분포간에 유사도 또는 차이를 정량화하는 방법들 중 하나 입니다. 
> 확률 분포 P와 Q사이에 동일한 값을 갖도록 하는 방법으로 KL Divergence를 사용합니다.  
>
> JS(P || Q) == JS(Q || P)
>
> JS(P || Q) = 1/2 * KL(P || M) + 1/2 * KL(Q || M)
> 
> M = 1/2 * (P + Q)

> - Original Python

```python


# example of calculating the js divergence between two mass functions
from math import log2
from math import sqrt
from numpy import asarray
 
# calculate the kl divergence
def kl_divergence(p, q):
	return sum(p[i] * log2(p[i]/q[i]) for i in range(len(p)))
 
# calculate the js divergence
def js_divergence(p, q):
	m = 0.5 * (p + q)
	return 0.5 * kl_divergence(p, m) + 0.5 * kl_divergence(q, m)
 
# define distributions
p = asarray([0.10, 0.40, 0.50])
q = asarray([0.80, 0.15, 0.05])
# calculate JS(P || Q)
js_pq = js_divergence(p, q)
print('JS(P || Q) divergence: %.3f bits' % js_pq)
print('JS(P || Q) distance: %.3f' % sqrt(js_pq))
# calculate JS(Q || P)
js_qp = js_divergence(q, p)
print('JS(Q || P) divergence: %.3f bits' % js_qp)
print('JS(Q || P) distance: %.3f' % sqrt(js_qp))

```

> - SciPy 활용

```python

# calculate the jensen-shannon distance metric
from scipy.spatial.distance import jensenshannon
from numpy import asarray
# define distributions
p = asarray([0.10, 0.40, 0.50])
q = asarray([0.80, 0.15, 0.05])
# calculate JS(P || Q)
js_pq = jensenshannon(p, q, base=2)
print('JS(P || Q) Distance: %.3f' % js_pq)
# calculate JS(Q || P)
js_qp = jensenshannon(q, p, base=2)
print('JS(Q || P) Distance: %.3f' % js_qp)

```


위의 글에서 볼 수 있듯이 KL Divergence 는 relative entropy인 것을 확인 할 수 있습니다. 

- Cross-Entropy : Average number of total bits to represent an event from Q instead of P
- Relative Entropy : Average number of extra bits to represents an event from Q instead of P

따라서, P, Q의 교차 엔트로피는 P에 대한 엔트로피와 P와 Q의 KL Divergence로  표현이 가능합니다. 

H(P, Q) = H(P) + KL(P || Q)


### Calculate Cross-Entropy

1. Two Discrete Probability Distribution

```python

# plot of distributions
from matplotlib import pyplot
# define distributions
events = ['red', 'green', 'blue']
p = [0.10, 0.40, 0.50]
q = [0.80, 0.15, 0.05]
print('P=%.3f Q=%.3f' % (sum(p), sum(q)))
# plot first distribution
pyplot.subplot(2,1,1)
pyplot.bar(events, p)
# plot second distribution
pyplot.subplot(2,1,2)
pyplot.bar(events, q)
# show the plot
pyplot.show()

```


2. Calculate Cross-Entropy Between Distribution

```python

# example of calculating cross entropy
from math import log2
 
# calculate cross entropy
def cross_entropy(p, q):
	return -sum([p[i]*log2(q[i]) for i in range(len(p))])
 
# define data
p = [0.10, 0.40, 0.50]
q = [0.80, 0.15, 0.05]
# calculate cross entropy H(P, Q)
ce_pq = cross_entropy(p, q)
print('H(P, Q): %.3f bits' % ce_pq)
# calculate cross entropy H(Q, P)
ce_qp = cross_entropy(q, p)
print('H(Q, P): %.3f bits' % ce_qp)

```


3. Calculate Cross-Entropy Between a Distribution and Itself

```python

# example of calculating cross entropy for identical distributions
from math import log2
 
# calculate cross entropy
def cross_entropy(p, q):
	return -sum([p[i]*log2(q[i]) for i in range(len(p))])
 
# define data
p = [0.10, 0.40, 0.50]
q = [0.80, 0.15, 0.05]
# calculate cross entropy H(P, P)
ce_pp = cross_entropy(p, p)
print('H(P, P): %.3f bits' % ce_pp)
# calculate cross entropy H(Q, Q)
ce_qq = cross_entropy(q, q)
print('H(Q, Q): %.3f bits' % ce_qq)

```

4. Calculate Cross-Entropy Using KL Divergence

```python

# example of calculating cross entropy with kl divergence
from math import log2
 
# calculate the kl divergence KL(P || Q)
def kl_divergence(p, q):
	return sum(p[i] * log2(p[i]/q[i]) for i in range(len(p)))
 
# calculate entropy H(P)
def entropy(p):
	return -sum([p[i] * log2(p[i]) for i in range(len(p))])
 
# calculate cross entropy H(P, Q)
def cross_entropy(p, q):
	return entropy(p) + kl_divergence(p, q)
 
# define data
p = [0.10, 0.40, 0.50]
q = [0.80, 0.15, 0.05]
# calculate H(P)
en_p = entropy(p)
print('H(P): %.3f bits' % en_p)
# calculate kl divergence KL(P || Q)
kl_pq = kl_divergence(p, q)
print('KL(P || Q): %.3f bits' % kl_pq)
# calculate cross entropy H(P, Q)
ce_pq = cross_entropy(p, q)
print('H(P, Q): %.3f bits' % ce_pq)

```



### Cross-Entropy as a Loss Function

Cross-Entropy는 classification 모델을 최적화 할때 주로 사용되는 loss function입니다. 
Classification 모델은 입력된 변수가 레이블링 된 클래스의 위치를 예측합니다. 
클래스간의 확률 분포 사이에 input 변수의 클래스를 예측하는 것입니다. 

실제 값(Expected Probability)의 확률 값은 P, 모델 예측값(Predicted Probability)의 확률 값을 Q라고 한다면
P와 Q의 Cross-entropy 를 활용해서 H(P, Q) = -sum x in X P(x) * log(Q(x)) 인 것을 구할 수 있습니다. 

Binary Classification(label 값이 0과 1로 구성) 인 경우로 보면 H(P, Q) = -(P(class 0) * log(Q(class 0)) + p(class 1) * log(Q(class 1))) 로 표현되는 것을 확인 할 수 있습니다. Bernoulli 분포와 동일하다는 것을 확인 할 수 있습니다. 


### Calculate Entropy for Class Labels

```python

# entropy of examples from a classification task with 3 classes
from math import log2
from numpy import asarray
 
# calculate entropy
def entropy(p):
	return -sum([p[i] * log2(p[i]) for i in range(len(p))])
 
# class 1
p = asarray([1,0,0]) + 1e-15
print(entropy(p))
# class 2
p = asarray([0,1,0]) + 1e-15
print(entropy(p))
# class 3
p = asarray([0,0,1]) + 1e-15
print(entropy(p))


```

앞서 이야기 한것처럼, P와 Q 간의 크로스 Entropy 는 P 의 엔트로피와 P Q 간의 KL Divergence 값과 동일합니다. 
결국 KL Divergence 값 만으로도 loss function으로 사용이 가능하다는 것을 알 수 있습니다. 



### Calculate Cross-Entropy Between Class Label and Probabilities

classification 의 cross-entropy는 분류 형태에 따라 Binary Cross-entropy, categorical Cross-entropy 와같이 특정이름을 줍니다. 

- Binary Cross-Entropy: Cross-entropy as a loss function for a binary classification task
- Categorical Cross-Entropy: Cross-entropy as a loss function for a multi-class classification task.

example)

```python

# calculate cross entropy for classification problem
from math import log
from numpy import mean
 
# calculate cross entropy
def cross_entropy(p, q):
	return -sum([p[i]*log(q[i]) for i in range(len(p))])
 
# define classification data
p = [1, 1, 1, 1, 1, 0, 0, 0, 0, 0]
q = [0.8, 0.9, 0.9, 0.6, 0.8, 0.1, 0.4, 0.2, 0.1, 0.3]
# calculate cross entropy for each example
results = list()
for i in range(len(p)):
	# create the distribution for each event {0, 1}
	expected = [1.0 - p[i], p[i]]
	predicted = [1.0 - q[i], q[i]]
	# calculate cross entropy for the two events
	ce = cross_entropy(expected, predicted)
	print('>[y=%.1f, yhat=%.1f] ce: %.3f nats' % (p[i], q[i], ce))
	results.append(ce)
 
# calculate the average cross entropy
mean_ce = mean(results)
print('Average Cross Entropy: %.3f nats' % mean_ce)

```

- calculate Cross-entropy using Keras

```python

# calculate cross entropy with keras
from numpy import asarray
from keras import backend
from keras.losses import binary_crossentropy
# prepare classification data
p = asarray([1, 1, 1, 1, 1, 0, 0, 0, 0, 0])
q = asarray([0.8, 0.9, 0.9, 0.6, 0.8, 0.1, 0.4, 0.2, 0.1, 0.3])
# convert to keras variables
y_true = backend.variable(p)
y_pred = backend.variable(q)
# calculate the average cross-entropy
mean_ce = backend.eval(binary_crossentropy(y_true, y_pred))
print('Average Cross Entropy: %.3f nats' % mean_ce)

```

### Intuition for Cross-Entropy on Predicted Probabilites

예측 확률 분포가 목표 분포에서 멀어질수록 교차 엔트로피가 증가할 것입니다. 
KL Divergence의 예를 통해 알수 있듯이 P가 크고 Q가 작거나 Q가 크고 P가 작으면 발산하기 때문입니다. 

아래 코드를 통해 확인할 수 있습니다.
```python

# cross-entropy for predicted probability distribution vs label
from math import log
from matplotlib import pyplot
 
# calculate cross-entropy
def cross_entropy(p, q, ets=1e-15):
	return -sum([p[i]*log(q[i]+ets) for i in range(len(p))])
 
# define the target distribution for two events
target = [0.0, 1.0]
# define probabilities for the first event
probs = [1.0, 0.9, 0.8, 0.7, 0.6, 0.5, 0.4, 0.3, 0.2, 0.1]
# create probability distributions for the two events
dists = [[1.0 - p, p] for p in probs]
# calculate cross-entropy for each distribution
ents = [cross_entropy(target, d) for d in dists]
# plot probability distribution vs cross-entropy
pyplot.plot([1-p for p in probs], ents, marker='.')
pyplot.title('Probability Distribution vs Cross-Entropy')
pyplot.xticks([1-p for p in probs], ['[%.1f,%.1f]'%(d[0],d[1]) for d in dists], rotation=70)
pyplot.subplots_adjust(bottom=0.2)
pyplot.xlabel('Probability Distribution')
pyplot.ylabel('Cross-Entropy (nats)')
pyplot.show()

```

일반적으로 Cross-Entropy 구간별 값의 의미는 다음과 같습니다.

- Cross-Entropy = 0.00: Perfect Probabilities
- Cross-Entropy < 0.02: Great Probabilites
- Cross-Entropy < 0.05: On the right track
- Cross-Entropy < 0.20: Fine
- Cross-Entropy > 0.30 : Not great
- Cross-Entropy > 1.00 : Terrible
- Cross-Entropy > 2.00 : Something broken


### Cross-Entropy Versus Log Loss

Log(logistic) loss 역시 분류 문제에 loss function으로 사용됩니다. 특히, 딥러닝에서 사용하는데 그 이유는 다양한 확률분포를 가정할 수 있다는 이점을 가지고 있기 때문입니다.

많은 모델들이 MLE(Maximum likelihood estimation)을 기준으로 최적화 합니다. MLE는 관측 데이터를 잘 설명할 수 있는 파라미터 셋을 찾는 것입니다. 
주어진 모델 파라미터 변수들을 정의하는 likelihood function을 선택하는 것이 포함됩니다.
실제로 함수의 값을 최대화 하기 보다는 최소화 하는 것이 일반적이기 때문에 함수에 -값을 붙여줍니다.
그래서 Negative Log Likelihood function이라고 말하기도 합니다. 

Log loss는 다음과 같이 정의 됩니다. E = -log(Q(x))
P와 Q간의 loss function은 log(P(x)) - log(Q(x)) = log(P(x)/ Q(x)) 이며 KL(P || Q) 를 따르는 형태로 나옵니다.
이러한 이점이 있기 때문에 딥러닝 모델에서 log loss를 손실함수(loss function)로 사용합니다.

여기까지 이해하셨으면,  Bernoulli 확률분포에 대한 likelihood function과 cross entropy는 동일한 계산을 가져온다는 것을 캐치하실 수 있으실 것입니다.
Binary Classification과 Bernoulli는 동일한 분포를 가지고 있기 때문입니다. 

- negative log-likelihood(P, Q) = -(P(class 0) * log(Q(class 0 )) + P(class 1) * log(Q(class 1)))
- Binary Classification : H(P, Q) = -(P(class 0) * log(Q(class 0)) + p(class 1) * log(Q(class 1)))


```python

# calculate log loss for classification problem with scikit-learn
from sklearn.metrics import log_loss
from numpy import asarray
# define classification data
p = [1, 1, 1, 1, 1, 0, 0, 0, 0, 0]
q = [0.8, 0.9, 0.9, 0.6, 0.8, 0.1, 0.4, 0.2, 0.1, 0.3]
# define data as expected, e.g. probability for each event {0, 1}
y_true = asarray([[1-v, v] for v in p])
y_pred = asarray([[1-v, v] for v in q])
# calculate the average log loss
ll = log_loss(y_true, y_pred)
print('Average Log Loss: %.3f' % ll)

```

MLE로 선형 회귀를 최적화 한다는 것은 Gaussian 연속 확률 분포안에서 목표 변수가 mean squarred error 함수를 최소화 한다는 것을 알 수 가정합니다. 이것은 Gaussian 확률 분포에서 cross-entropy를 가 된다는 것입니다. 

아래 순서대로 보면 이해하실 수 있으실 것입니다. 

- 선형 회귀는 distance 차이를 제곱한 값들 합의 최소값을 구합니다. 이것이 MSE 입니다.
- MSE(Mean Squarred Error)가 (P)예측 분포와 (Q)사전 정의된 분포 두개 사이의 Gaussian 모델간의 Cross-Entropy가 됩니다.

이를 통해 (-)log loss 로 구성된 손실은 training set에 정의된 경험적 분포와 model에 정의된 확률 분포 사이의 교차 엔트로피라는 것을 알수 있습니다.

differential entropy 개념(확률 밀도 함수의 미분)으로 넘어가면, 두개의 Gaussian Random Variable 사이에서 MSE 계산은 두 variable 사이의 cross-entropy 계산이 됩니다. 따라서, Neural Network에서 MSE를 사용하는 것은 Cross-entropy를 사용하는 것과 동일하다는 것을 알 수 있습니다.


