# CoPD-master
===

This is a pytorch implementation of our paper: "Consistency-guided Preference Disentanglement of Cross Domain Recommendation"  

## Requirements: 
---

python=3.9.0

pytorch=1.12.0

numpy=1.24.3

scipy=1.10.1 

## Runing commands 
---

### Firstly, cd src. Next, run the codes with the following commands on different scenarios.


-->on Elec & Phone: 
```shell
CUDA_VISIBLE_DEVICES=0 python train_rec.py --dataset electronic_phone --lambda1 1 --lambda2 1 
```

-->on Sport & Cloth: 
```shell
CUDA_VISIBLE_DEVICES=0 python train_rec.py --dataset sport_cloth --lambda1 1 --lambda2 1 
```

-->on Sport & Phone:
```shell
CUDA_VISIBLE_DEVICES=0 python train_rec.py --dataset sport_phone --lambda1 0.01 --lambda2 1
```

-->on Elec & Cloth:  
```shell
CUDA_VISIBLE_DEVICES=0 python train_rec.py --dataset electronic_cloth --lambda1 0.01 --lambda2 1
```

### If you find this paper or codes useful, please cite our paper. Thank you!

```
```

## Acknowledgement
---
This code refers code from: 

https://github.com/HKUDS/LightGCL

https://github.com/cjx96/DisenCDR

https://github.com/xuChenSJTU/ETL-master

We thank the authors for sharing their codes!