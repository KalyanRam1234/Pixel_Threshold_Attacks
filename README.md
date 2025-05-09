# Pixel and Threshold Attack on Xview Dataset

This GitHub repository contains the official code for the papers,

> [Adversarial robustness assessment: Why in evaluation both L0 and L∞ attacks are necessary](https://journals.plos.org/plosone/article?id=10.1371/journal.pone.0265723)\
> Shashank Kotyan and Danilo Vasconcellos Vargas, \
> PLOS One (2022).

> [One pixel attack for fooling deep neural networks](https://ieeexplore.ieee.org/abstract/document/8601309)\
> Jiawei Su, Danilo Vasconcellos Vargas, Kouichi Sakurai\
> IEEE Transactions on Evolutionary Computation (2019).
 
## Citation

If this work helps your research and/or project in anyway, please cite:

```bibtex
@article{kotyan2022adversarial,
  title={Adversarial robustness assessment: Why in evaluation both L 0 and L∞ attacks are necessary},
  author={Kotyan, Shashank and Vargas, Danilo Vasconcellos},
  journal={PloS one},
  volume={17},
  number={4},
  pages={e0265723},
  year={2022},
  publisher={Public Library of Science San Francisco, CA USA}
}

@article{su2019one,
  title     = {One pixel attack for fooling deep neural networks},
  author    = {Su, Jiawei and Vargas, Danilo Vasconcellos and Sakurai, Kouichi},
  journal   = {IEEE Transactions on Evolutionary Computation},
  volume    = {23},
  number    = {5},
  pages     = {828--841},
  year      = {2019},
  publisher = {IEEE}
}
```

Additionally it contains the work on the Xview dataset and YoloV3 network as well.

## Environment Setup

Ensure that you have a system with a GPU to do the training, if you already have the model weights, then its not needed.

First clone this repository in your local system.
```bash
git clone <This repo link>
```

To install the required packages, use the XviewAttackenv.yml file
```bash
conda create -f XviewAttackenv.yml
conda activate dqa
```

Link of YoloV3 tensorflow model repo used: [YoloV3 Architecture](https://github.com/zzh8829/yolov3-tf2/blob/master/train.py)

## Dataset Installation

We ran the attacks on the Xview dataset, it can be installed from the Xview dataset challenge page : [Xview Dataset](http://xviewdataset.org/#dataset)

Or it can download the formatted dataset we have used from this link: [Xview Formatted Dataset](https://drive.google.com/drive/folders/1P0fL3wWNJkjwfq5NBYRjw-KzPx4FYEBl?usp=sharing)

### File Structure

The folders are expected to look like: 

```bash
datasets
|__images
    |__train
        |__1.tiff
        |__2.tiff
    |__val
        |__1.tiff
        |__2.tiff
    |__autosplit_train.txt
    |__autosplit_val.txt
|__labels
    |__train
        |__1.txt
```

## Downloading Weights and Results

The results of our experiments, which involve the affects of the attack and the model weights can be found in this link: [Results and Weights](https://drive.google.com/drive/folders/1Ctlv4lhhjTADT0bnHnX4iv7jNUeyqkx7?usp=sharing)

## Work Done

We first started off by setting up the old code of dual quality assessmenet and understanding the working of the code.

It required setting up the required tensorflow data