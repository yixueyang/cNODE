#!/bin/bash
#SBATCH --job-name=cNODE
#SBATCH --output=cNODE_%j.out
#SBATCH --error=cNODE_%j.err
#SBATCH --partition=old_40  
#SBATCH --nodelist=cu03
#SBATCH --nodes=1
#SBATCH --ntasks=1

source /home/liufei/anaconda3/bin/activate
conda activate /home/liufei/anaconda3/envs/torch

python real_data_cnode.py