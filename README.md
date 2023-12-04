# CoPD

CUDA_VISIBLE_DEVICES=0 python train_rec.py --dataset electronic_phone --lambda1 1 --lambda2 1

CUDA_VISIBLE_DEVICES=1 python train_rec.py --dataset sport_cloth --lambda1 1 --lambda2 1

CUDA_VISIBLE_DEVICES=2 python train_rec.py --dataset sport_phone --lambda1 0.01 --lambda2 1

CUDA_VISIBLE_DEVICES=3 python train_rec.py --dataset electronic_cloth --lambda1 0.01 --lambda2 1

