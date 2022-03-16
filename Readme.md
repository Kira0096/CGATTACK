This is the code for the CVPR 2022 submission 7175, **Boosting Black-Box Attack with Partially Transferred Conditional Adversarial Distribution**. 
This project is developed based on Python 3.6. 

## Install prerequisites
```
pip install -r requirements.txt
```

## Download pre-trained model
Download the pretrained models and dataset [[download link]](https://drive.google.com/file/d/1WwclqsVxezicWHTZeif7HzwWUjCXCkyt/view?usp=sharing) and unzip it with 
```
unzip pretrained.zip
```
Then you can conduct the untargeted attack for CIFAR-10 evaluation without training.


## Robustness evaluation
* Evaluate our CG-ES against TARGET_MODEL [resnet.sh|densenet.sh|vgg.sh|pyramidnet.sh] by running

```
sh scripts/cifar_unt/TARGET_MODEL
```

