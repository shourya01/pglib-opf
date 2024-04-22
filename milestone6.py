import os, pickle
import numpy as np
import torch.nn.functional as F
from oct2py import Oct2Py
import cyipopt
import warnings
from tqdm import trange
from pypower.ext2int import ext2int
from pypower.api import loadcase
from problemDefJITM import opfSocp
from problemDefJITMargin import opfSocpMargin
from utils import make_data_parallel
from tqdm import tqdm, trange
import torch
import torch.nn as nn
import gc
from time import time
from ConvexModel import ConvexNet
from ClassifierModel import ClassifierNet
from RidgeModel import RidgeNet
import xgboost as xgb

from sklearn.tree import DecisionTreeClassifier
from sklearn.multioutput import MultiOutputClassifier
from sklearn.linear_model import RidgeClassifier

octave = Oct2Py()
dir_name = 'data/'
MAX_BUS = 10000
NUM_SOLVES = 20

if __name__ == "__main__":
    
    # get all cases in current directory
    current_directory = os.getcwd()+'/'
    # current_directory = '/home/sbose/pglib-opf/' # for running on BEBOP
    all_files_and_directories = os.listdir(current_directory)
    # three specific cases
    case_files = [current_directory+i for i in ['pglib_opf_case118_ieee.m','pglib_opf_case2312_goc.m',"pglib_opf_case4601_goc.m","pglib_opf_case10000_goc.m"]]
    # case_files = [current_directory+i for i in ['pglib_opf_case2312_goc.m',"pglib_opf_case4601_goc.m"]]

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
            
    for cn,this_case in zip(casenames,cases):
    
        inp_data = np.load(os.getcwd()+'/'+dir_name+f'{cn}_inp.npz')['data']
        dual_data = np.load(os.getcwd()+'/'+dir_name+f'{cn}_dual.npz')['data']
        cost_data = np.load(os.getcwd()+'/'+dir_name+f'{cn}_costs.npz')['data'][:,None]
        
        # modify the input
        optObj = opfSocp(this_case,cn) # generate object
        
        
        
        # import data
        inp_data = np.load(os.getcwd()+f'/saved/{cn}_test_inp.npz')['data']
        
        # solve for convex
        convex_out = np.load(os.getcwd()+f'/saved/{cn}_out_convex.npz')['data'][:NUM_SOLVES,:]
        times, solves = [], []
        for sidx in range(NUM_SOLVES):
            
            # time and solve original problem
            optObj = opfSocp(this_case,cn)
            pd,qd = inp_data[sidx,:optObj.n_bus], inp_data[sidx,optObj.n_bus:2*(optObj.n_bus)]
            optObj.change_loads(pd,qd)
            cub, clb = optObj.calc_cons_bounds()
            xub, xlb = optObj.calc_var_bounds()
            
            # time variable for original
            time_orig = 0.
            
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
            prob.add_option('max_iter',2500)
            prob.add_option('mumps_mem_percent',25000)
            prob.add_option('mu_max',1e-0)
            prob.add_option('mu_init',1e-0)
            prob.add_option('nlp_lower_bound_inf',-optObj.LARGE_NUMBER+1)
            prob.add_option('nlp_upper_bound_inf',optObj.LARGE_NUMBER-1)
            prob.add_option('print_level',0) 
            
            start = time()
            x,info = prob.solve(optObj.calc_x0_flatstart())
            end = time()
            
            orig = end-start
            val = info['obj_val']
            print(f'Solved original problem for {cn} in {orig}s, objective value is {val}.')
            
            # solve modified prob - set up indices
            ineqidx = ((1-optObj.is_model)*(1-optObj.is_equality)).astype(bool) # nonmodel inequalities
            nmineq = np.ones(2*optObj.n_bus+4*optObj.n_branch)
            nmineq[:2*optObj.n_bus] = 0
            nmineq = nmineq.astype(bool)
            
            
            this_time, this_solve = 0,0
            marker = convex_out[sidx,:]
            clnew, cunew = clb,cub
            clnew[ineqidx] = np.where(marker[nmineq]==0,-optObj.LARGE_NUMBER,clb[ineqidx])
            cunew[ineqidx] = np.where(marker[nmineq]==0,optObj.LARGE_NUMBER,cub[ineqidx])
            prob = cyipopt.Problem(
                n = optObj.in_size,
                m = optObj.cons_size,
                problem_obj=optObj,
                lb=xlb,
                ub=xub,
                cl=clnew,
                cu=cunew
            )
            # Setup solver options
            prob.add_option('tol',1e-6)
            prob.add_option('max_iter',2500)
            prob.add_option('mumps_mem_percent',25000)
            prob.add_option('mu_max',1e-1)
            prob.add_option('mu_init',1e-1)
            prob.add_option('nlp_lower_bound_inf',-optObj.LARGE_NUMBER+1)
            prob.add_option('nlp_upper_bound_inf',optObj.LARGE_NUMBER-1)
            prob.add_option('print_level',0) 
            start = time()
            x, info = prob.solve(optObj.calc_x0_flatstart())
            end = time()
            this_time += end-start
            this_solve += 1
            # infer constraints
            inferred_cons = (optObj.constraints(x)-cub).clip(min=0)[ineqidx]
            val = info['obj_val']
            print(f'First solve objective is {val}, {np.where(np.abs(inferred_cons)>1e-3,1,0).sum()} violated cons.')
            if np.where(np.abs(inferred_cons)>1e-4,1,0).sum() > 0:
                marker[nmineq] = np.where(np.abs(inferred_cons)>1e-4,0,inferred_cons)
                # resolve
                clnew, cunew = clb,cub
                clnew[ineqidx] = np.where(marker[nmineq]==0,-optObj.LARGE_NUMBER,clb[ineqidx])
                cunew[ineqidx] = np.where(marker[nmineq]==0,optObj.LARGE_NUMBER,cub[ineqidx])
                prob = cyipopt.Problem(
                    n = optObj.in_size,
                    m = optObj.cons_size,
                    problem_obj=optObj,
                    lb=xlb,
                    ub=xub,
                    cl=clnew,
                    cu=cunew
                )
                # Setup solver options
                prob.add_option('tol',1e-6)
                prob.add_option('max_iter',2500)
                prob.add_option('mumps_mem_percent',25000)
                prob.add_option('mu_max',1e-1)
                prob.add_option('mu_init',1e-1)
                prob.add_option('nlp_lower_bound_inf',-optObj.LARGE_NUMBER+1)
                prob.add_option('nlp_upper_bound_inf',optObj.LARGE_NUMBER-1)
                prob.add_option('print_level',0) 
                start = time()
                x, info = prob.solve(optObj.calc_x0_flatstart())
                end = time()
                inferred_cons = (optObj.constraints(x)-cub).clip(min=0)[ineqidx]
                val = info['obj_val']
                print(f'Second solve objective is {val}, {np.where(np.abs(inferred_cons)>1e-4,1,0).sum()} violated cons.')
                this_time += end-start
                this_solve += 1
            print(f'For method convex, solved {cn}-reduced in {this_time} with {this_solve} solve(s).\n')
            
            
            
            
            
        
        