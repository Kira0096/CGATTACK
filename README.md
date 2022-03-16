Official repository for CVPR 2022 paper **Boosting Black-Box Attack with Partially Transferred Conditional Adversarial Distribution**.

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

## Citation
Please cite our paper in your publications if it helps your research:

```
@inproceedings{Feng_CGATTACK_2022,
  title={Boosting Black-Box Attack with Partially Transferred Conditional Adversarial Distribution},
  author={Feng, Yan and Wu, Baoyuan and Fan, Yanbo and Liu, Li and Li, Zhifeng and Xia, Shutao},
  booktitle={IEEE Conference on Computer Vision and Pattern Recognition (CVPR)},
  year={2022}
}
```
