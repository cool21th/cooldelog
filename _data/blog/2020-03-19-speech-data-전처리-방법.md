---
template: BlogPost
path: /Speech
date: 2020-03-19T04:56:05.015Z
title: Speech data 전처리 방법
thumbnail: /assets/stt.png
---
### 오디오 파일 Read

```python
import IPython.display as ipd
fname = '../input/freesound-audio-tagging/audio_train/' + '00044347.wav'
ipd.Audio(fname)
```



```python
import IPython.display as ipd
fname = '../input/freesound-audio-tagging/audio_train/' + '00044347.wav'
signal = np.random.random(750)
ipd.Audio(signal, rate=250)
```

### 샘플링 파일, 토탈 Sampling 확인

```python
import wave
wav = wave.open(fname)
print("Sampling (frame) rete = ", wav.getframerate())
print("Total samples (frames) = ", wav.getnframes())
print("Duration = ", wav.getnframes()/wav.getframerate())
```

* (결과) 

  Sampling (frame) rate =  44100

  Total samples (frames) =  617400

  Duration =  14.0

```python
from scipy.io import wavfile
rate, data = wavfile.read(fname)
print("Sampling (frame) rate = ", rate)
print("Total samples (frames) = ", data.shape)
print(data)
```

* (결과) 

  Sampling (frame) rate =  44100

  Total samples (frames) =  (617400,)

  \[ 0 26 -5 ...  1  0  0]

type(wav)  은 wave.Wave_read 형태

type(data) 은 numpy.ndarray 형태로 구성된다





둘다 total sampling을 볼 수 있지만, plot을 그리기 위해서는 float 을 인수로 받는 Scipy.io 패키지를 이용하는 것이 좋아 보인다. 

```python
plt.plot(data, '-', );
       
plt.figure(figsize=(16,4))
plt.plot(data[:500], '.'); plt.plot(data[:500], '-')
```

### 가지고 있는 오디오 파일 frame 형태(분산) 확인

```python
train['nframes'] = train['fname'].apply(lambda f: wave.open('../input/freesound-audio-tagging/audio_train/'+f).getnframes())
test['nframes'] = test['fname'].apply(lambda f: wave.open('../input/freesound-audio-tagging/audio_test/'+f).getnframes())

_, ax = plt.subplots(figsize=(16, 4))

sns.violinplot(ax= ax, x='label', y='nframes', data=train)

plt.xticks(rotation=90)
plt.title('Distribution of audio frames, per label', fontsize=16)
plt.show()
```
