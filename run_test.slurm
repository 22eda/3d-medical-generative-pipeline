#!/bin/bash
#SBATCH --job-name=test_pred
#SBATCH --output=test_pred.out
#SBATCH --error=test_pred.err
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --gres=gpu:1
#SBATCH --time=00:05:00
#SBATCH -p plgrid-gpu-a100

module load GCC/12.3.0 Python/3.11.3 CUDA/12.1.1
source $SCRATCH/env-test/bin/activate

python test_save.py
