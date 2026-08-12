#!/bin/bash
#SBATCH -J 3d_unet_gen
#SBATCH -p plgrid-gpu-a100
#SBATCH -N 1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=4
#SBATCH --gres=gpu:1
#SBATCH --time=01:30:00
#SBATCH -o gen_%j.out
#SBATCH -e gen_%j.err

module load GCC/12.3.0 Python/3.11.3 CUDA/12.1.1
source $SCRATCH/env-test/bin/activate

cd $SCRATCH/ct3d_project
python -u train_generative.py
