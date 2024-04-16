#!/bin/bash

#SBATCH -p 128x24   # Partition name
#SBATCH -J ipopt-opf        # Job name
#SBATCH --mail-user=shbose@ucsc.edu
#SBATCH --mail-type=ALL
#SBATCH -o solve.log    # Name of stdout output file
#SBATCH -N 3        # Total number of nodes requested (128x24/Instructional only)
#SBATCH --ntasks=24
#SBATCH --cpus-per-task=3
#SBATCH -t 48:00:00  # Run Time (hh:mm:ss) - 1.5 hours (optional)

export OMPI_MCA_btl=tcp,sm,self\
module load miniconda3
conda init
conda activate FL
cd /hb/home/shbose/pglib-opf

# Use of -p replaces the need to use "#SBATCH --cpus-per-task"
srun -n 24 --cpus-per-task=3 --mem-per-cpu=4G python milestone4b.py