---
template: BlogPost
path: /Vision/pretrainedsegmentation
date: 2020-03-19T13:26:15.000Z
title: '[Python Library] Pretrained Segmentation Library 소개'
thumbnail: /assets/Segmentation Models.png
---
이미지에서 활용할 수 있는 Pretrained 모델들이 있는데, 누구나 쉽게 사용할 수 있는 Python Library를 소개하고자 합니다. 

[Segmentation Models](https://github.com/qubvel/segmentation_models) 이고, keras와 Tensorflow 를 기반으로 만들어진 Neural Network  입니다. 

사용법은 아주 간단합니다. 

1.  설치 

    pip install -U segmentation-models==0.2.1 


2. 사용방법

```python
    import segmentation_models as sm
```

기본적으로 Keras framework 을 사용하나, tensorflow.keras 를 import 해서 사용할 경우 다음과 같이 환경변수를 변경해줘야 합니다.

- Provide environment variable ``SM_FRAMEWORK=keras`` / ``SM_FRAMEWORK=tf.keras`` before import ``segmentation_models``
- Change framework ``sm.set_framework('keras')`` /  ``sm.set_framework('tf.keras')``


```python

    import keras
    # or from tensorflow import keras

    keras.backend.set_image_data_format('channels_last')
    # or keras.backend.set_image_data_format('channels_first')
    
    # 기본구조
    model = sm.Unet()
    
    # Pretrained 모델 사용시
    model = sm.Unet('resnet34', encoder_weights='imagenet')
    
    # Input 변경시
    model = Unet('resnet34', input_shape=(None, None, 6), encoder_weights=None)

```


3. 샘플 예제

```python

    import segmentation_models as sm

    BACKBONE = 'resnet34'
    preprocess_input = sm.get_preprocessing(BACKBONE)

    # load your data
    x_train, y_train, x_val, y_val = load_data(...)

    # preprocess input
    x_train = preprocess_input(x_train)
    x_val = preprocess_input(x_val)

    # define model
    model = sm.Unet(BACKBONE, encoder_weights='imagenet')
    model.compile(
        'Adam',
        loss=sm.losses.bce_jaccard_loss,
        metrics=[sm.metrics.iou_score],
    )

    # fit model
    # if you use data generator use model.fit_generator(...) instead of model.fit(...)
    # more about `fit_generator` here: https://keras.io/models/sequential/#fit_generator
    model.fit(
       x=x_train,
       y=y_train,
       batch_size=16,
       epochs=100,
       validation_data=(x_val, y_val),
    )

```

**Architecture**

============= ==============
Unet          Linknet
============= ==============
|unet_image|  |linknet_image|
============= ==============

============= ==============
PSPNet        FPN
============= ==============
|psp_image|   |fpn_image|
============= ==============


.. |unet_image| ![unet](https://github.com/cool21th/cooldelog/tree/master/public/assets/unet.png)
.. |linknet_image| ![linknet](https://github.com/cool21th/cooldelog/tree/master/public/assets/linknet.png)
.. |psp_image| ![psp](https://github.com/cool21th/cooldelog/tree/master/public/assets/pspnet.png)
.. |fpn_image| ![fpn](https://github.com/cool21th/cooldelog/tree/master/public/assets/fpn.png)



**Backbones**

.. table:: 

    =============  ===== 
    Type           Names
    =============  =====
    VGG            ``'vgg16' 'vgg19'``
    ResNet         ``'resnet18' 'resnet34' 'resnet50' 'resnet101' 'resnet152'``
    SE-ResNet      ``'seresnet18' 'seresnet34' 'seresnet50' 'seresnet101' 'seresnet152'``
    ResNeXt        ``'resnext50' 'resnext101'``
    SE-ResNeXt     ``'seresnext50' 'seresnext101'``
    SENet154       ``'senet154'``
    DenseNet       ``'densenet121' 'densenet169' 'densenet201'`` 
    Inception      ``'inceptionv3' 'inceptionresnetv2'``
    MobileNet      ``'mobilenet' 'mobilenetv2'``
    EfficientNet   ``'efficientnetb0' 'efficientnetb1' 'efficientnetb2' 'efficientnetb3' 'efficientnetb4' 'efficientnetb5' efficientnetb6' efficientnetb7'``
    =============  =====

.. epigraph::
    All backbones have weights trained on 2012 ILSVRC ImageNet dataset (``encoder_weights='imagenet'``). 

