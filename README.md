# Aurora Segmentation for Aurora Image with PU Learning.

**This is for segmentation of Aurora images with a deep learning model trained by PU Learning.**

This is also a reproducing code written in ***Keras*** for non-negative PU learning [1] and unbiased PU learning [2] in the paper "Positive-Unlabeled Learning with Non-Negative Risk Estimator". 

To get know about PU Learning, see `pulearn.ipynb` for a short PU Learning explanation.

- **`pu_loss.py`** has a Keras implementation of the risk estimator for non-negative PU (nnPU) learning and unbiased PU (uPU) learning.

- **`pu_loss.py`**

train.py is an example code of nnPU learning and uPU learning. Dataset are MNIST [3] preprocessed in such a way that even digits form the P class and odd digits form the N class and CIFAR10 [4] preprocessed in such a way that artifacts form the P class and living things form the N class. The default setting is 100 P data and 59900 U data of MNIST, and the class prior is the ratio of P class data in U data.

## Quick Start Example

```sh
python main.py --P_dataset dataset/Positive --U_dataset dataset/Unlabelled
```

## Result Example


## Reference

[1] Ryuichi Kiryo, Gang Niu, Marthinus Christoffel du Plessis, and Masashi Sugiyama. "Positive-Unlabeled Learning with Non-Negative Risk Estimator." Advances in neural information processing systems. 2017.

[2] Marthinus Christoffel du Plessis, Gang Niu, and Masashi Sugiyama. "Convex formulation for learning from positive and unlabeled data." Proceedings of The 32nd International Conference on Machine Learning. 2015.

[3] [Chainer implementation of non-negative PU learning and unbiased PU learning](https://github.com/kiryor/nnPUlearning)
