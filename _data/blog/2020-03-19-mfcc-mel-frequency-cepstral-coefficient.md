---
template: BlogPost
path: /speech/mfcc
date: 2020-03-19T05:18:36.758Z
title: '[Speech] MFCC(Mel-Frequency Cepstral Coefficient)'
thumbnail: /assets/mfcc.png
---
### MFCC(Mel-Frequency Cepstral Coefficient) Overview

MFCC는 오디오 신호에서 추출할 수 있는 Feature로, 소리의 고유한 특징을 나타내는 수치로, 주로 음성인식, 화자인식, 음성합성, 음악 장르 분류 등 오디오 도메인의 문제를 해결하는데 사용되는 알고리즘 입니다. 물론 복잡한 기법을 요구하지 않습니다. 1980년대 소개된 알고리즘이지만, 현재 음성기반 딥러닝에 가장 중요한 요소로 자리잡고 있습니다.

좀더 음성 데이터와 관련해서 이야기를 하면, 보통 음성 및 소리 데이터들은 target 이 되는 소리 뿐 아니라, 각종 Noise, 배경 소리 들과 함께 Input 이 되어 Training Dataset으로 구성됩니다. 여기서 추출할 대상은 화자 또는 음성 데이터에서 가지고 있는 Contents가 무엇인지를 추출해야 하는 것입니다. MFCC알고리즘은 Input으로 들어온 데이터를 일정구간에 대해 스펙트럼 분석으로 Training 할 수 있는 Feature 들을 추출할 수 있습니다. 

### MFCC 활용

1. 화자 검증(Speaker Verification)

> 화자 검증이란 화자 인식(Speaker Recognitin)의 세부 분류로서 말하는 사람이 그 사람이 맞는 지를 확인하는 기술로서, 시스템에 등록된 음성에만 반응하는 아이폰의 Siri를 예로 들수 있습니다.
>
> 확인하는 방법은 훈련된 MFCC패턴과 입력된 음성 패턴의 유사도를 기준으로 판별합니다.  초기 SKT Nugu의 경우 아이들의 목소리를 잘 알아듣지 못하는 현상들이 있었는데, 그 이유는 아이들의 음성 패턴이 화자 검증의 훈련데이터로 부족하지 않았나 추측됩니다.

2. 음악 장르 분류(Music Genre Classification)

MFCC는 소리 패턴을 분석하기 때문에 음악 신호에서도 패턴 추출이 가능합니다.음악의 특징들을 추출하여, 장르 구분에 사용가능합니다. 

### MFCC의 기술적인 이해

MFCC 는 Mel Spectrum에서 Cepstral 분석을 통해 추출된 값을 의미합니다. 그 구성요소는 Spectrum, Cepstrum, Mel Spectrum있습니다. 

![mfcc](/assets/mfcc.png "mfcc")



1. MFCC의 추출과정

* 입력시간 도메인의 소리 신호를 작은 프레임으로 자름

  > 오디오 신호를 프레임별(보통 20ms ~ 40ms)로 나누어 FFT 를 적용해 Spectrum 계산
* Spectrum 에 Mel Filter Bank 를 적용해 각 필터 에너지를 합쳐서 Mel Spectrum을 구한다
* Mel Spectrum에 Cepstral 분석을 적용해 MFCC를 구한다

오디오 신호는 시간(가로축)에  따른 음압(세로축)의 표현, 즉 시간 영역(time domain)의 표현이고, 여기서 FFT를 수행하면 주파수(가로축)에 따른 음압(세로축)의 표현, 즉 주파수 영역(frequency domain)의 표현이 가능해지고, 그것이 Spectrum 인 것이다.

> FFT(Fast Fourier Transform: 고속 푸리에 변환) 신호를 주파수 성분으로 변환하는 알고리즘으로, 기존의 이산 푸리에 변환을 더욱 빠르게 수행할 수 있도록 최적화한 알고리즘

Spectrum을 사용하면 각 주파수의 대역별 세기를 알 수 있으니, 신호에서 어떤 주파수가 강하고 약한지를 알 수 있다 

이렇게 주파수에 대한 정보를 가진 Spectrum에서 소리의 고유한 특징을 추출할 수 있습니다.  그리고 그 정보를 추출할 때 사용하는 방법이 Cepstral 분석이다.

다만 MFCC는 일반적인 Spectrum이 아니라 특수한 필터링을 거친 Mel Spectrum 에 Cepstral 분석을 적용해 추출가능하다 

2. Spectrum

어떻게 Spectrum에서 소리의 고유한 특징을 추출할 수 있을까? 그 답은 먼저 Specturm에 어떤 정보가 숨겨져 있는지 알면 찾을 수 있다.

악기 소리나 사람의 음성은 일반적으로 배음(harmonics)구조를 가지고 있다.

배음(harmonics)구조 소리는 한가지의 주파수만으로 구성되지 않는다. 기본 주파수(fundamental frequency)와 함께 기본 주파수의 정수배인 배음(harmonics)들로 구성된다.

예를 들어 우리가 피아노 건반에서 4옥타브 '라' (440Hz)음을 연주했다면 그 소리는 기본 주파수인 440Hz뿐만 아니라  그 정수배인 880Hz, 그리고 그 다음 배음들까지 포함하고 있다.

배음 구조는 악기나 성대의 구조에 따라 달라지며 배움 구조의 차이가 음색의 차이를 만들어서, Spectrum에서 배음 구조를 유추해낼 수 있다면 소리의 고유한 특징을 찾아낼 수 있다. Cepstral 분석이 이것을 가능하게 한다.

3. Cepstral Analysis

이제 Cepstral 분석이 어떤 과정으로 수행되는 지 알아보기 위해 주파수에 대해 먼저 확인할 수 있다.

피크들은 신호에서 지배적인 주파수 영역을 가리킨다

이 피크들을 포먼트(Formants)라고 한다.

포먼트(Formants) 소리가 공명되는 특정 주파수 대역

사람의 음성은 성대에서 형성되어 성도를 거치며 변형되는데,  소리는 성도를 지나면서 포먼트를 만나 증폭되거나 감쇠 된. 

즉, 포먼트는 배음(harmonics)과 만나 소리를 풍성하게 혹은 선명하게 만드는 필터 역할을 한다.

포먼트는 소리의 특징을 유추할 수 있는 중요한 단서가 된다.. 

우리가 해야 할 잃은 포먼트들을 연결한 곡선과 Spectrum을 분리해내는 일이 가능하다 

그 곡선을 Spectral Envelope라고 하고, MFCC는 둘을 분리하는 과정에서 도출됩니다.  이때 사용하는 수학, 알고리즘이 log와 IFFT(Inverse FFT: 역 고속 푸리에변환)이. 

4. Mel Spectrum

위에서 MFCC는 Spectrum이 아닌 Mel Spectrum에서 Cepstral분석으로 추출한다고 했는데, Mel Spectrum이 어떤 과정을 거쳐 만들어지는지 알아보려고 한다. 

사람의 청각기관은 고주파수 보다 저주파수 대역에서 더 민감하다

사람의 이런 특성을 반영해 물리적인 주파수와 실제 사람이 인식기하는 주파수의 관계를 표현하는 것이 Mel Scale(멜 스케일) 이다. 

이 Mel Scale 에 기반한 Filter Bank를 Spectrum에 적용하여 도출해낸 것이 Mel Spectrum입니다. Filter Bank를 나눌 때 어떤 간격으로 나눠야 하는지 알려주는 역할을 한. 

수학적인 설명은 [여기에서](http://practicalcryptography.com/miscellaneous/machine-learning/guide-mel-frequency-cepstral-coefficients-mfccs/) 참고하시기 바랍니다

##### mfcc 간단 코드

```python
import librosa
SAMPLE_RATE = 44100
fname = '../input/freesound-audio-tagging/audio_train/' + '00044347.wav'
wav, _ = librosa.core.load(fname, sr=SAMPLE_RATE)
wav = wav[:2*44100]

mfcc = librosa.feature.mfcc(wav, sr = SAMPLE_RATE, n_mfcc=40)
mfcc.shape

plt.imshow(mfcc, cmap='hot', interpolation='nearest');
```

참고:  
[Mel Frequency Cepstral Coefficient(MFCC) tutorial](http://practicalcryptography.com/miscellaneous/machine-learning/guide-mel-frequency-cepstral-coefficients-mfccs/)

[mfcc 이해하기](https://brightwon.tistory.com/11)

[Beginner's Guide to Audio Data](https://www.kaggle.com/fizzbuzz/beginner-s-guide-to-audio-data)
