# Title: Coherence-guided Preference Disentanglement for Cross-domain Recommendations
(This repository is implementation of CoPD submitted to ACM Transactions on Information Systems 2024)

[Zongyi Xiang](), [Yan Zhang](), [Lixin Duan](), [Hongzhi Yin]() and [Ivor W. Tsang](), "Coherence-guided Preference Disentanglement for Cross-domain Recommendations"

## Contents
1. [Introduction](#introduction)
2. [Environment](#Environment)
3. [Running](#Runnning)
4. [Citation](#citation)
5. [Acknowledgements](#acknowledgements)


## Introduction
 Discovering user preferences across different domains is pivotal in cross-domain recommendation systems, particularly when platforms lack comprehensive user-item interactive data. The limited presence of shared users often hampers the effective modeling of common preferences. While leveraging shared items’ attributes, such as category and popularity, can enhance cross-domain recommendation performance, the scarcity of shared items between domains has limited research in this area. To address this, we propose a Coherence guided Preference Disentanglement (CoPD) method aimed at improving cross-domain recommendation by i) explicitly extracting shared item attributes to guide the learning of shared user preferences and ii) disentangling these preferences to identify specific user interests transferred between domains. CoPD introduces coherence constraints on item embeddings of shared and specific domains,aiding in extracting shared attributes. Moreover, it utilizes these attributes to guide the disentanglement of user preferences into separate embeddings for interest and conformity through a popularity-weighted loss. Experiments conducted on real-world datasets demonstrate the superior performance of our proposed CoPD over existing competitive baselines, highlighting its effectiveness in enhancing cross-domain recommendation performance.

## Environment
Python 3.9.0           

PyTorch=1.12.0

numpy=1.24.3

scipy=1.11.1      

## Running

-->on Amazon Elec & Phone: 

```shell
CUDA_VISIBLE_DEVICES=0 python train_rec.py --dataset electronic_phone --lambda1 1 --lambda2 1 
```

-->on Amazon  Sport & Cloth: 

```shell
CUDA_VISIBLE_DEVICES=1 python train_rec.py --dataset sport_cloth --lambda1 1 --lambda2 1 
```

-->on Amazon  Sport & Phone:

```shell
CUDA_VISIBLE_DEVICES=2 python train_rec.py --dataset sport_phone --lambda1 0.01 --lambda2 1
```

-->on Amazon   Elec & Cloth:  

```shell
CUDA_VISIBLE_DEVICES=3 python train_rec.py --dataset electronic_cloth --lambda1 0.01 --lambda2 1
```

-->on Douban Movie & Book:  

```shell
CUDA_VISIBLE_DEVICES=0 python train_rec.py --dataset movie_book --lambda1 1 --lambda2 1
```

-->on Douban Movie & Music:  

```shell
CUDA_VISIBLE_DEVICES=1 python train_rec.py --dataset movie_music --lambda1 1 --lambda2 1
```


## Citation
If you find the code helpful in your research or work, please cite the following papers.
```
```

# Acknowledgement.
This code refers code from: 

https://github.com/HKUDS/LightGCL

https://github.com/cjx96/DisenCDR

https://github.com/xuChenSJTU/ETL-master

We thank the authors for sharing their codes!
