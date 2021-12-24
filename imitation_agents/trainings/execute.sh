#!/bin/bash
  
#SBATCH --nodes=1      # do not change
#SBATCH --gres=gpu:1    
#SBATCH --ntasks-per-node=24    # num of cores
#SBATCH --time=05-00:00            # Time limit (D-HH:MM)
#SBATCH -o %j.out                 # Standard output log file
#SBATCH -e %j.err                 # Standard err log file

conda activate train_carla
which python
conda list | grep torch 
conda list | grep cuda
python train.py
