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
ml load Z3/4.10.2-GCCcore-11.3.0-Python-3.10.4
ml load Triton/2.1.0-foss-2022a-CUDA-11.7.0
ml load PyTorch/2.1.0-foss-2022a-CUDA-11.7.0
ml load Python/3.10.4-GCCcore-11.3.0
ml load torchvision/0.16.0-foss-2022a-CUDA-11.7.0
ml load matplotlib/3.5.2-foss-2022a
python -m wandb online
python train_vqvae.py --dataset bird --loss perceptual
