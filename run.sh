#!/bin/bash -l
#SBATCH --job-name=LOWESTEST
#SBATCH --account=NEXTGENOPT
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1
#SBATCH --time=23:59:59
#SBATCH --output=HAX.log

now=$(date)
echo $now

module load anaconda3
conda init
conda activate autodiff
cd /home/sbose/pglib-opf

# Use of -p replaces the need to use "#SBATCH --cpus-per-task"
python milestone5b.py
scancel -u sbose