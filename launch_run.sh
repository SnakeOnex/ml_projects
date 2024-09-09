#!/bin/sh
#SBATCH --time 23:59:00
#SBATCH -n 8
#SBATCH --partition=amdgpu
#SBATCH --gres=gpu:1
#BATCH --exclusive
/bin/hostname
nvidia-smi
pwd
ls -l
#source load.sh
#bash load_cuda1170.sh && python train_vqvae.py --dataset bird
ml load Z3/4.13.0-GCCcore-13.2.0
ml load PyTorch/2.4.0-foss-2023b-CUDA-12.4.0
ml load Python/3.11.5-GCCcore-13.2.0
ml load torchvision/0.19.0-foss-2023b-CUDA-12.4.0
ml load matplotlib/3.8.2-gfbf-2023b
ml load PyTorch-Lightning/2.4.0-foss-2023b-CUDA-12.4.0
ml load tqdm/4.66.2-GCCcore-13.2.0
python -m wandb online
#python train_vqgan.py --dataset imagenet
python train_gpt.py --dataset bird
#python train_maskgit.py --dataset bird
