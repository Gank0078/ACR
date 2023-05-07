# [CVPR 2023] Towards Realistic Long-Tailed Semi-Supervised Learning: Consistency Is All You Need

This is PyTorch implementation of Towards Realistic Long-Tailed Semi-Supervised Learning: Consistency Is All You Need.

## Abstract
While long-tailed semi-supervised learning (LTSSL) has received tremendous attention in many real-world classification problems, existing LTSSL algorithms typically assume that the class distributions of labeled and unlabeled data are almost identical. Those LTSSL algorithms built upon the assumption can severely suffer when the class distributions of labeled and unlabeled data are mismatched since they utilize biased pseudo-labels from the model. To alleviate this issue, we propose a new simple method that can effectively utilize unlabeled data of unknown class distributions by introducing the adaptive consistency regularizer (ACR). ACR realizes the dynamic refinery of pseudolabels for various distributions in a unified formula by estimating the true class distribution of unlabeled data. Despite its simplicity, we show that ACR achieves state-of-the-art performance on a variety of standard LTSSL benchmarks, e.g., an averaged 10% absolute increase of test accuracy against existing algorithms when the class distributions of labeled and unlabeled data are mismatched. Even when the class distributions are identical, ACR consistently outperforms many sophisticated LTSSL algorithms. We carry out extensive ablation studies to tease apart the factors that are most important to ACR's success.

## Method

<!-- <p align = "center"> -->
<img src="assets/ACR-framework.png" align="center" width="80%" />
<!-- </p> -->

## Requirements

- Python 3.7.13
- PyTorch 1.12.0+cu116
- torchvision
- numpy



## Dataset

The directory structure for datasets looks like:
```
datasets
├── cifar-10
├── cifar-100
├── stl-10
├── imagenet32
└── imagenet64
```


## Usage

Train our proposed ACR on CIFAR10-LT of different settings.

For consistent:

```
python train.py --dataset cifar10 --num-max 500 --num-max-u 4000 --arch wideresnet --batch-size 64 --lr 0.03 --seed 0 --imb-ratio-label 100 --imb-ratio-unlabel 100 --ema-u 0.99 --out out/cifar-10/N500_M4000/consistent
```

For uniform:

```
python train.py --dataset cifar10 --num-max 500 --num-max-u 4000 --arch wideresnet --batch-size 64 --lr 0.03 --seed 0 --imb-ratio-label 100 --imb-ratio-unlabel 1 --ema-u 0.99 --out out/cifar-10/N500_M4000/uniform
```

For reversed:

```
python train.py --dataset cifar10 --num-max 500 --num-max-u 4000 --arch wideresnet --batch-size 64 --lr 0.03 --seed 0 --imb-ratio-label 100 --imb-ratio-unlabel 100 --ema-u 0.99 --flag-reverse-LT 1 --out out/cifar-10/N500_M4000/reversed
```

## Acknowledgement
Our code of ACR is based on the implementation of FixMatch. We thank the authors of the [FixMatch](https://github.com/kekmodel/FixMatch-pytorch) for making their code available to the public.



## Citation
```
@inproceedings{weitowards,
  title={Towards Realistic Long-Tailed Semi-Supervised Learning: Consistency Is All You Need},
  author={Wei, Tong and Gan, Kai},
  booktitle={Proceedings of the IEEE Conference on Computer Vision and Pattern Recognition},
  year={2023}
}
```


