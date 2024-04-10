#!/bin/bash
#SBATCH -e slurm.err 
#SBATCH -c 2
#SBATCH --mem-per-cpu=15G
#SBATCH -p common-old
#SBATCH -o /dev/null

python 09ga_ATTN_Eval_slurm.py

