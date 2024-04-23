#!/usr/bin/bash

#PBS -N HTRPL

#PBS -o t4.log
#PBS -e t4.log

# cd /projects/zhanglab/init_trials_grid2op/L2RPN_delft2023_starting_kit/
echo $PBS_O_WORKDIR
cd /projects/zhanglab/shourya/pglib-opf
conda activate autodiff
/projects/zhanglab/CondaEnv/rlenv/bin/mpiexec -np 30 --map-by core:PE=2 python milestone4b.py

# export PATH="/soe/shbose/miniconda3/bin/:$PATH"
# activate GRIDRL

# /projects/zhanglab/CondaEnv/rlenv/bin/python -c 'from lightsim2grid import gridmodel'

#module load mpi

# /projects/zhanglab/CondaEnv/rlenv/bin/mpiexec -np 18 --map-by core:PE=1 /projects/zhanglab/CondaEnv/rlenv/bin/python /projects/zhanglab/init_trials_grid2op/L2RPN_delft2023_starting_kit/L2RPN2023/Teacher3.py --deg_of_sep 4
# /projects/zhanglab/CondaEnv/rlenv/bin/mpiexec -np 16 --map-by node:pe=3 /projects/zhanglab/CondaEnv/rlenv/bin/python /projects/zhanglab/init_trials_grid2op/L2RPN_delft2023_starting_kit/L2RPN2023/Teacher1.py
# /projects/zhanglab/CondaEnv/rlenv/bin/mpiexec -np 20 --cpus-per-proc 8 /projects/zhanglab/CondaEnv/rlenv/bin/python /projects/zhanglab/init_trials_grid2op/L2RPN_delft2023_starting_kit/L2RPN2023/Teacher1.py
#/soe/shbose/miniconda3/envs/GRIDRL/bin/python /projects/zhanglab/init_trials_grid2op/L2RPN_delft2023_starting_kit/codes/main.py --gamma 0.999 --tau 0.01 --seed 21 --save_suffix test1  --load_model 1 --model_file sac_checkpoint_l2rpn_wcci_2022_test1_0_0.pt
#/soe/shbose/miniconda3/envs/GRIDRL/bin/python /projects/zhanglab/init_trials_grid2op/L2RPN_delft2023_starting_kit/codes/main.py --gamma 0.999 --tau 0.01 --hidden_size 75 --seed 42 --save_suffix test2 --load_model 1 --model_file sac_checkpoint_l2rpn_wcci_2022_test2_0_0.pt

# test0 was tau=0.005 gamma = 0.99
# test1 tau = 0.01 gamma = 0.999