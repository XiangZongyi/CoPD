# CoPD-master
This is a pytorch implementation of our paper: "Consistency-guided Preference Disentanglement of Cross Domain Recommendation"  

**Requirements:**  
python3.9.0; pytorch=1.12.0; numpy=1.24.3; scipy=1.11.1  

**Runing commands**   
Firstly, cd src. Next, run the codes with the following commands on different scenarios.

-->on Elec & Phone: 
CUDA_VISIBLE_DEVICES=gpu_num python train_rec.py --dataset electronic_phone --lambda1 1 --lambda2 1 

-->on Sport & Cloth:  
CUDA_VISIBLE_DEVICES=gpu_num python train_rec.py --dataset sport_cloth --lambda1 1 --lambda2 1 

-->on Sport & Phone:  
CUDA_VISIBLE_DEVICES=gpu_num python train_rec.py --dataset sport_phone --lambda1 0.01 --lambda2 1

-->on Elec & Cloth:  
CUDA_VISIBLE_DEVICES=gpu_num python train_rec.py --dataset electronic_cloth --lambda1 0.01 --lambda2 1

**Acknowledgement**  
This code refers code from: 
https://github.com/cjx96/DisenCDR
[HKUDS/LightGCL: [ICLR'2023\] "LightGCL: Simple Yet Effective Graph Contrastive Learning for Recommendation" (github.com)](https://github.com/HKUDS/LightGCL)
[cjx96/DisenCDR: [SIGIR 2022\]DisenCDR: Learning Disentangled Representations for Cross-Domain Recommendation (github.com)](https://github.com/cjx96/DisenCDR)
[xuChenSJTU/ETL-master: This is a pytorch implementation of our paper: "Towards Equivalent Transformation of User Preferences in Cross Domain Recommendation" (github.com)](https://github.com/xuChenSJTU/ETL-master)
We thank the authors for sharing their codes!