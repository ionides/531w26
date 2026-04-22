#!/bin/bash
#SBATCH --job-name=stats531_final
#SBATCH --account=stats531w26s001_class
#SBATCH --partition=gpu
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=4
#SBATCH --gpus-per-node=1
#SBATCH --mem=32G
#SBATCH --time=02:00:00
#SBATCH --output=final_%j.out
#SBATCH --mail-type=END,FAIL
#SBATCH --mail-user=prudvi@umich.edu

module purge
source ~/.bashrc
conda activate pypomp_env

export XLA_PYTHON_CLIENT_PREALLOCATE=false
export JAX_PLATFORM_NAME=gpu

echo "Job started: $(date)"
nvidia-smi | head -10

python final_targeted.py

echo "Job finished: $(date)"
