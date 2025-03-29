#!/bin/bash
#SBATCH --account=bdnb-delta-gpu
#SBATCH --job-name=atekkey1
#SBATCH --error=/work/nvme/bdnb/atekkey/sam2/_main/jobError.err
#SBATCH --output=/work/nvme/bdnb/atekkey/sam2/_main/jobOut.log
#SBATCH --tasks=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=140G
#SBATCH --time=07:00:00
#SBATCH --partition=gpuA100x4
#SBATCH --nodes=1
#SBATCH --gpus-per-node=1
#SBATCH --gpu-bind=closest

python BP_mem_enc_1.py
