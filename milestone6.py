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
from problemDefJITR import opfSocpR
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
            
    # define kwargs
    problem_settings_kwargs = lambda obj:{
        'tol':1e-6,
        'mu_max':1e-0,
        'mu_init':1e-0,
        'nlp_lower_bound_inf':-obj.LARGE_NUMBER+1,
        'nlp_upper_bound_inf':obj.LARGE_NUMBER-1,
        'print_level':0
    }
    problem_def_kwargs = lambda obj,var_ub,var_lb,cons_ub,cons_lb: {
        'n':var_lb.size,
        'm':cons_lb.size,
        'problem_obj':obj,
        'lb':var_lb,
        'ub':var_ub,
        'cl':cons_lb,
        'cu':cons_ub
    }
            
    for cn,this_case in zip(casenames,cases):
    
        inp_data = np.load(os.getcwd()+'/'+dir_name+f'{cn}_inp.npz')['data']
        dual_data = np.load(os.getcwd()+'/'+dir_name+f'{cn}_dual.npz')['data']
        
        # modify the input
        optObj = opfSocp(this_case,cn) # generate object
        
        # do full unperturbed solve
        optObj = opfSocp(this_case,cn)
        pdk, psk = problem_def_kwargs(optObj,*optObj.calc_var_bounds(),*optObj.calc_cons_bounds()), problem_settings_kwargs(optObj)
        prob = cyipopt.Problem(**pdk)
        for k,v in psk.items():
            prob.add_option(k,v)
        start = time()
        xorig, infoorig = prob.solve(optObj.calc_x0_flatstart())
        end = time()
        obj_orig, status = infoorig['obj_val'], infoorig['status']
        print(f"-----\nUnperturbed full problem solved in {(end-start):.5f}s with objective {obj_orig} and status {status}.\n-----\n\n")
        
        # input data for different methods
        convex_out = np.load(os.getcwd()+f'/saved/{cn}_out_convex.npz')['data'] # convex
        
        # set up relevant indices for reduced solves
        ineqidx = ((1-optObj.is_model)*(1-optObj.is_equality)).astype(bool) # nonmodel inequalities
        nmineq = np.ones(2*optObj.n_bus+4*optObj.n_branch)
        nmineq[:2*optObj.n_bus] = 0
        nmineq = nmineq.astype(bool).tolist()
        
        full_times, times, solves = [], [], []
        optObj = opfSocp(this_case,cn)
        cub, _ = optObj.calc_cons_bounds() # only for inferring violations
        
        for sidx in range(NUM_SOLVES):
            
            # CONVEX
            
            # extract input and dual data and create reduced solve obj
            pd,qd = inp_data[sidx,:optObj.n_bus], inp_data[sidx,optObj.n_bus:2*(optObj.n_bus)]
            convex_marker = convex_out[sidx,:][nmineq]
            
            # do full perturbed solve
            optObj.change_loads(pd,qd)
            pdk, psk = problem_def_kwargs(optObj,*optObj.calc_var_bounds(),*optObj.calc_cons_bounds()), problem_settings_kwargs(optObj)
            prob = cyipopt.Problem(**pdk)
            for k,v in psk.items():
                prob.add_option(k,v)
            start = time()
            xorig, infoorig = prob.solve(optObj.calc_x0_flatstart())
            end = time()
            obj_orig, status = infoorig['obj_val'], infoorig['status']
            print(f"Perturbed full problem solved in {(end-start):.5f}s with objective {obj_orig} and status {status}.")
            full_times.append(end-start)
            
            # calculate first solve for convex
            optObjR = opfSocpR(this_case,convex_marker,cn)
            optObjR.change_loads(pd,qd)
            pdk, psk = problem_def_kwargs(optObjR,*optObjR.calc_var_bounds(),*optObjR.calc_cons_bounds()), problem_settings_kwargs(optObjR)
            prob = cyipopt.Problem(**pdk)
            for k,v in psk.items():
                prob.add_option(k,v)
            start = time()
            xcofirst, infocofirst = prob.solve(optObjR.calc_x0_flatstart())
            obj_cofirst = infocofirst['obj_val']
            end = time()
            timeco = end-start
            solvesco = 1
            
            # infer violated constraints
            inferred_nmineq = (optObj.constraints(xcofirst)-cub).clip(min=0)[ineqidx]
            inferred_viols = np.where(np.abs(inferred_nmineq)>1e-5,1,0)
            
            if inferred_viols.sum() == 0:
                print(f"Reduced problem solved in {(timeco):.5f}s with 1 solve, objective {obj_cofirst} with {inferred_viols.sum()} violations.\n")
                times.append(timeco)
                solves.append(solvesco)
            else:
                convex_marker = np.where(convex_marker+inferred_viols>0,0,1)
                optObjR = opfSocpR(this_case,convex_marker,cn)
                optObjR.change_loads(pd,qd)
                pdk, psk = problem_def_kwargs(optObjR,*optObjR.calc_var_bounds(),*optObjR.calc_cons_bounds()), problem_settings_kwargs(optObjR)
                prob = cyipopt.Problem(**pdk)
                for k,v in psk.items():
                    prob.add_option(k,v)
                start = time()
                xcosecond, infocosecond = prob.solve(optObjR.calc_x0_flatstart())
                obj_cosecond = infocosecond['obj_val']
                end = time()
                timeco += end-start
                solvesco += 1
                inferred_nmineq = (optObj.constraints(xcofirst)-cub).clip(min=0)[ineqidx]
                inferred_viols = np.where(np.abs(inferred_nmineq)>1e-5,1,0)
                print(f"Reduced problem solved in {(timeco):.5f}s with 2 solves, objective {obj_cosecond} with {inferred_viols.sum()} violations.\n")
                times.append(timeco)
                solves.append(solvesco)
                
        # print stats
        print(f"For case {cn}\nfull problem average solve time: {np.array(full_times).mean()}\nreduced problem average solve time: {np.array(times).mean()}\nreduced problem average solves: {np.array(solves).mean()}.\n\n")