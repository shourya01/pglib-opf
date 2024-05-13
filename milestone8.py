import os, pickle
import numpy as np
from oct2py import Oct2Py
import cyipopt
import warnings
from tqdm import trange
from pypower.ext2int import ext2int
from pypower.api import loadcase
from problemDefJITMRank import opfSocp
from mpi4py import MPI
import argparse

# argument of program to be run
parser = argparse.ArgumentParser(description="Generate data files.")
parser.add_argument('--case', type=str, default='pglib_opf_case118_ieee')
parser.add_argument('--diff', type=float, default=-0.5)
args = parser.parse_args()

# # get octave object
octave = Oct2Py()
# filter warnings
warnings.simplefilter("ignore", np.ComplexWarning)

# user options 
MAX_BUS = 10000 # upper limit of number of buses in cases to be considered
NUM_POINTS = 100 # number of data points to save
DIFF = args.diff # ratio variation of data
TOL = 1E-4 # tolerances of data

# main

if __name__ == "__main__":
    
    # sort out the difference value
    default_diffs = {
        'pglib_opf_case118_ieee':1.5,
        'pglib_opf_case793_goc':1.5,
        'pglib_opf_case1354_pegase':1.5,
        'pglib_opf_case2312_goc':1.5,
        'pglib_opf_case4601_goc':1.5,
        'pglib_opf_case10000_goc':1.5
    }
    
    # get mpi rank
    comm = MPI.COMM_WORLD
    mpi_rank = comm.Get_rank()
    mpi_size = comm.Get_size()
    NUM_POINTS = NUM_POINTS // mpi_size
    
    # get all cases in current directory
    current_directory = os.getcwd()+'/'
    # current_directory = '/home/sbose/pglib-opf/' # for running on BEBOP
    all_files_and_directories = os.listdir(current_directory)
    # three specific cases
    # case_files = [current_directory+i for i in ['pglib_opf_case3970_goc.m','pglib_opf_case2869_pegase.m','pglib_opf_case118_ieee.m','pglib_opf_case9241_pegase.m']]
    case_files = [current_directory+i for i in [args.case+'.m']]

    cases, casenames = [], []
    cases_full, casenames_full = [], []
    for cf in case_files:
        octave.source(current_directory+os.path.basename(cf))
        cname = os.path.basename(cf).split('.')[0]
        num_buses = None
        # determine number of buses in the case from its name
        for ci in cname.split('_'):
            if 'case' in ci:
                num_buses = int(''.join([chr for chr in ci.replace('case','',1) if chr.isdigit()]))
        # fitler out cases with more buses than MAX_BUS
        if num_buses <= MAX_BUS:
            # convert to internal indexing
            case_correct_idx = ext2int(loadcase(octave.feval(cname)))
            # append
            cases.append(case_correct_idx)
            casenames.append(cname)
            
    # solve
    for cn,this_case in zip(casenames,cases):
        optObj = opfSocp(this_case,cn)
        cub, clb = optObj.calc_cons_bounds()
        xub, xlb = optObj.calc_var_bounds()
        
        # default diffs
        if args.diff < 0:
            DIFF = default_diffs[cn]
        
        # Define IPOPT problem
        prob = cyipopt.Problem(
            n = optObj.in_size,
            m = optObj.cons_size,
            problem_obj=optObj,
            lb=xlb,
            ub=xub,
            cl=clb,
            cu=cub
        )
        
        # Setup solver options
        prob.add_option('tol',1e-6)
        prob.add_option('mumps_mem_percent',25000)
        prob.add_option('mu_max',1e+1) 
        prob.add_option('mu_init',1e+1)
        prob.add_option('print_level',0) 
        prob.add_option('max_iter',300)
        
        # Solve ipopt problem
        _, info_base = prob.solve(optObj.calc_x0_flatstart())
        
        # data generation
        print(f"\n--------\nSolving {cn}.\n--------\n",flush=True)
        
        # Setup solver options
        prob.add_option('tol',1e-6)
        prob.add_option('mumps_mem_percent',25000)
        prob.add_option('mu_max',1e+1) 
        prob.add_option('mu_init',1e+1)
        prob.add_option('print_level',0) 
        prob.add_option('max_iter',300)
        
        # generate points
        if cn == 'pglib_opf_case1354_pegase':
            NUM_POINTS = int(10*NUM_POINTS)
        bind, rank, pqdiff = [], [], []
        for pt_idx in (t:=trange(NUM_POINTS)):
            
            # set random seed
            np.random.seed(pt_idx*(mpi_rank+1)*3745)
            
            # get pd, qd and perturb
            pd,qd = optObj.get_loads()
            
            dpd,dqd = (1-DIFF + 2*np.random.rand(*pd.shape)*DIFF)*pd, (1-DIFF + 2*np.random.rand(*qd.shape)*DIFF)*qd
            optObj.change_loads(dpd,dqd)

            # solve problem
            x, info = prob.solve(info_base['x'],lagrange=info_base['mult_g'].tolist(),zl=info_base['mult_x_L'].tolist(),zu=info_base['mult_x_U'].tolist())
            if info['status'] == 0:
                cur_bind = optObj.get_num_binding_constr(x,tol=TOL)
                # cur_rank = optObj.get_rank_partial_jacobian(x,tol=TOL)
                cur_rank = optObj.get_num_binding_constr(x,tol=TOL)
                cur_pqdiff = optObj.get_load_diff_norm(ord = 1)
                bind.append(cur_bind)
                rank.append(cur_rank)
                pqdiff.append(cur_pqdiff)
                t.set_description(f"Case {cn}, problem {pt_idx} solved! Process ({mpi_rank+1}/{mpi_size}) Binding: {cur_bind}, rank = {cur_rank}, pqdiff = {cur_pqdiff}.")
                
            else:
                
                t.set_description(f"Case {cn}, problem {pt_idx} was not solved. :(")
            
        # save data
        os.makedirs(os.getcwd()+f'/data3',exist_ok=True)
        if len(bind) > 0:
            rankdata = np.array([bind,rank,pqdiff])
            np.savez_compressed(os.getcwd()+f'/data3/{cn}_rankdata.npz',data=rankdata)