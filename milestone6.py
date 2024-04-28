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
NUM_SOLVES = 2
VIOL_THRES = 1e-4

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
            
    # write
    logfile = open('perf2.txt','a')
            
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
        
        print(f"-----\nSolving case {cn}\n-----\n\n")
    
        inp_data = np.load(os.getcwd()+'/'+dir_name+f'{cn}_inp.npz')['data']
        dual_data = np.load(os.getcwd()+'/'+dir_name+f'{cn}_dual.npz')['data']
        
        # modify the input
        optObj = opfSocp(this_case,cn) # generate object
        
        # # do full unperturbed solve
        # optObj = opfSocp(this_case,cn)
        # pdk, psk = problem_def_kwargs(optObj,*optObj.calc_var_bounds(),*optObj.calc_cons_bounds()), problem_settings_kwargs(optObj)
        # prob = cyipopt.Problem(**pdk)
        # for k,v in psk.items():
        #     prob.add_option(k,v)
        # start = time()
        # xorig, infoorig = prob.solve(optObj.calc_x0_flatstart())
        # end = time()
        # obj_orig, status = infoorig['obj_val'], infoorig['status']
        # print(f"-----\nUnperturbed full problem solved in {(end-start):.5f}s with objective {obj_orig} and status {status}.\n-----\n\n")
        
        # input data for different methods
        convex_out = np.load(os.getcwd()+f'/saved/{cn}_out_convex.npz')['data'] # convex
        classifier_out = np.load(os.getcwd()+f'/saved/{cn}_out_classifier.npz')['data'] # classifier
        ridge_out = np.load(os.getcwd()+f'/saved/{cn}_out_ridge.npz')['data'] # ridge
        xgboost_out = np.load(os.getcwd()+f'/saved/{cn}_out_xgboost.npz')['data'] # xgboost
        
        # set up relevant indices for reduced solves
        ineqidx = ((1-optObj.is_model)*(1-optObj.is_equality)).astype(bool) # nonmodel inequalities
        nmineq = np.ones(2*optObj.n_bus+4*optObj.n_branch)
        nmineq[:2*optObj.n_bus] = 0
        nmineq = nmineq.astype(bool).tolist()
        
        # set up lists to record time
        full_times = []
        convex_times, convex_solves = [], []
        classifier_times, classifier_solves = [], []
        ridge_times, ridge_solves = [], []
        xgboost_times, xgboost_solves = [], []
        optObj = opfSocp(this_case,cn)
        cub, _ = optObj.calc_cons_bounds() # only for inferring violations
        
        for sidx in range(NUM_SOLVES):
            
            # extract input and dual data and create reduced solve obj
            pd,qd = inp_data[sidx,:optObj.n_bus], inp_data[sidx,optObj.n_bus:2*(optObj.n_bus)]
            
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
            
            # CONVEX
            
            # print
            print("Convex:")
            
            # marker
            convex_marker = convex_out[sidx,:][nmineq]
            
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
            inferred_viols = np.where(np.abs(inferred_nmineq)>VIOL_THRES,1,0)
            
            if inferred_viols.sum() == 0:
                print(f"Reduced problem solved in {(timeco):.5f}s with 1 solve, objective {obj_cofirst} with {inferred_viols.sum()} violations.")
                convex_times.append(timeco)
                convex_solves.append(solvesco)
            else:
                print(f"Detected {inferred_viols.sum()} violations with objective {obj_cofirst}.")
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
                inferred_nmineq = (optObj.constraints(xcosecond)-cub).clip(min=0)[ineqidx]
                inferred_viols = np.where(np.abs(inferred_nmineq)>VIOL_THRES,1,0)
                print(f"Reduced problem solved in {(timeco):.5f}s with 2 solves, objective {obj_cosecond} with {inferred_viols.sum()} violations.")
                convex_times.append(timeco)
                convex_solves.append(solvesco)
                
            # CLASSIFIER
            
            # print
            print("Classifier:")
            
            # marker
            classifier_marker = classifier_out[sidx,:][nmineq]
            
            # calculate first solve for classifier
            optObjR = opfSocpR(this_case,classifier_marker,cn)
            optObjR.change_loads(pd,qd)
            pdk, psk = problem_def_kwargs(optObjR,*optObjR.calc_var_bounds(),*optObjR.calc_cons_bounds()), problem_settings_kwargs(optObjR)
            prob = cyipopt.Problem(**pdk)
            for k,v in psk.items():
                prob.add_option(k,v)
            start = time()
            xclfirst, infoclfirst = prob.solve(optObjR.calc_x0_flatstart())
            obj_clfirst = infoclfirst['obj_val']
            end = time()
            timecl = end-start
            solvescl = 1
            
            # infer violated constraints
            inferred_nmineq = (optObj.constraints(xclfirst)-cub).clip(min=0)[ineqidx]
            inferred_viols = np.where(np.abs(inferred_nmineq)>VIOL_THRES,1,0)
            
            if inferred_viols.sum() == 0:
                print(f"Reduced problem solved in {(timecl):.5f}s with 1 solve, objective {obj_clfirst} with {inferred_viols.sum()} violations.")
                classifier_times.append(timecl)
                classifier_solves.append(solvescl)
            else:
                print(f"Detected {inferred_viols.sum()} violations with objective {obj_clfirst}.")
                classifier_marker = np.where(classifier_marker+inferred_viols>0,0,1)
                optObjR = opfSocpR(this_case,classifier_marker,cn)
                optObjR.change_loads(pd,qd)
                pdk, psk = problem_def_kwargs(optObjR,*optObjR.calc_var_bounds(),*optObjR.calc_cons_bounds()), problem_settings_kwargs(optObjR)
                prob = cyipopt.Problem(**pdk)
                for k,v in psk.items():
                    prob.add_option(k,v)
                start = time()
                xclsecond, infoclsecond = prob.solve(optObjR.calc_x0_flatstart())
                obj_clsecond = infoclsecond['obj_val']
                end = time()
                timecl += end-start
                solvescl += 1
                inferred_nmineq = (optObj.constraints(xclsecond)-cub).clip(min=0)[ineqidx]
                inferred_viols = np.where(np.abs(inferred_nmineq)>VIOL_THRES,1,0)
                print(f"Reduced problem solved in {(timecl):.5f}s with 2 solves, objective {obj_clsecond} with {inferred_viols.sum()} violations.")
                classifier_times.append(timecl)
                classifier_solves.append(solvescl)
                
            # RIDGE
            
            # print
            print("Ridge:")
            print(ridge_out.shape)
            
            # marker
            ridge_marker = ridge_out[sidx,:][nmineq]
            
            # calculate first solve for classifier
            optObjR = opfSocpR(this_case,ridge_marker,cn)
            optObjR.change_loads(pd,qd)
            pdk, psk = problem_def_kwargs(optObjR,*optObjR.calc_var_bounds(),*optObjR.calc_cons_bounds()), problem_settings_kwargs(optObjR)
            prob = cyipopt.Problem(**pdk)
            for k,v in psk.items():
                prob.add_option(k,v)
            start = time()
            xrgfirst, inforgfirst = prob.solve(optObjR.calc_x0_flatstart())
            obj_rgfirst = inforgfirst['obj_val']
            end = time()
            timerg = end-start
            solvesrg = 1
            
            # infer violated constraints
            inferred_nmineq = (optObj.constraints(xrgfirst)-cub).clip(min=0)[ineqidx]
            inferred_viols = np.where(np.abs(inferred_nmineq)>VIOL_THRES,1,0)
            
            if inferred_viols.sum() == 0:
                print(f"Reduced problem solved in {(timerg):.5f}s with 1 solve, objective {obj_rgfirst} with {inferred_viols.sum()} violations.")
                classifier_times.append(timerg)
                classifier_solves.append(solvesrg)
            else:
                print(f"Detected {inferred_viols.sum()} violations with objective {obj_rgfirst}.")
                ridge_marker = np.where(ridge_marker+inferred_viols>0,0,1)
                optObjR = opfSocpR(this_case,ridge_marker,cn)
                optObjR.change_loads(pd,qd)
                pdk, psk = problem_def_kwargs(optObjR,*optObjR.calc_var_bounds(),*optObjR.calc_cons_bounds()), problem_settings_kwargs(optObjR)
                prob = cyipopt.Problem(**pdk)
                for k,v in psk.items():
                    prob.add_option(k,v)
                start = time()
                xrgsecond, inforgsecond = prob.solve(optObjR.calc_x0_flatstart())
                obj_rgsecond = inforgsecond['obj_val']
                end = time()
                timerg += end-start
                solvesrg  += 1
                inferred_nmineq = (optObj.constraints(xrgsecond)-cub).clip(min=0)[ineqidx]
                inferred_viols = np.where(np.abs(inferred_nmineq)>VIOL_THRES,1,0)
                print(f"Reduced problem solved in {(timerg):.5f}s with 2 solves, objective {obj_rgsecond} with {inferred_viols.sum()} violations.")
                ridge_times.append(timerg)
                ridge_solves.append(solvesrg)
                
            # XGBOOST
            
            # print
            print("XGBoost:")
            
            # marker
            print(xgboost_out.shape)
            xgboost_marker = xgboost_out[sidx,:][nmineq]
            
            # calculate first solve for classifier
            optObjR = opfSocpR(this_case,xgboost_marker,cn)
            optObjR.change_loads(pd,qd)
            pdk, psk = problem_def_kwargs(optObjR,*optObjR.calc_var_bounds(),*optObjR.calc_cons_bounds()), problem_settings_kwargs(optObjR)
            prob = cyipopt.Problem(**pdk)
            for k,v in psk.items():
                prob.add_option(k,v)
            start = time()
            xxgfirst, infoxgfirst = prob.solve(optObjR.calc_x0_flatstart())
            obj_xgfirst = infoxgfirst['obj_val']
            end = time()
            timexg = end-start
            solvesxg = 1
            
            # infer violated constraints
            inferred_nmineq = (optObj.constraints(xxgfirst)-cub).clip(min=0)[ineqidx]
            inferred_viols = np.where(np.abs(inferred_nmineq)>VIOL_THRES,1,0)
            
            if inferred_viols.sum() == 0:
                print(f"Reduced problem solved in {(timexg):.5f}s with 1 solve, objective {obj_xgfirst} with {inferred_viols.sum()} violations.\n")
                classifier_times.append(timexg)
                classifier_solves.append(solvesxg)
            else:
                print(f"Detected {inferred_viols.sum()} violations with objective {obj_rgfirst}.")
                xgboost_marker = np.where(xgboost_marker+inferred_viols>0,0,1)
                optObjR = opfSocpR(this_case,xgboost_marker,cn)
                optObjR.change_loads(pd,qd)
                pdk, psk = problem_def_kwargs(optObjR,*optObjR.calc_var_bounds(),*optObjR.calc_cons_bounds()), problem_settings_kwargs(optObjR)
                prob = cyipopt.Problem(**pdk)
                for k,v in psk.items():
                    prob.add_option(k,v)
                start = time()
                xxgsecond, infoxgsecond = prob.solve(optObjR.calc_x0_flatstart())
                obj_xgsecond = infoxgsecond['obj_val']
                end = time()
                timexg += end-start
                solvesxg  += 1
                inferred_nmineq = (optObj.constraints(xxgsecond)-cub).clip(min=0)[ineqidx]
                inferred_viols = np.where(np.abs(inferred_nmineq)>VIOL_THRES,1,0)
                print(f"Reduced problem solved in {(timexg):.5f}s with 2 solves, objective {obj_xgsecond} with {inferred_viols.sum()} violations.\n")
                xgboost_times.append(timexg)
                xgboost_solves.append(solvesxg)
                
                
        # print stats
        print(
                (f"For case {cn}, solved {NUM_SOLVES} test cases:\n"
                f"full problem average solve time: {np.array(full_times).mean()}"
                f"\nreduced problem average solve time (convex): {np.array(convex_times).mean()}"
                f"\nreduced problem average solves (convex): {np.array(convex_solves).mean()}.\n\n"
                f"\nreduced problem average solve time (classifier): {np.array(classifier_times).mean()}"
                f"\nreduced problem average solves (classifier): {np.array(classifier_solves).mean()}.\n\n"
                f"\nreduced problem average solve time (ridge): {np.array(ridge_times).mean()}"
                f"\nreduced problem average solves (ridge): {np.array(ridge_solves).mean()}.\n\n"
                f"\nreduced problem average solve time (xgboost): {np.array(xgboost_times).mean()}"
                f"\nreduced problem average solves (xgboost): {np.array(xgboost_solves).mean()}.\n\n")
            )
        logfile.write(
                (f"For case {cn}, solved {NUM_SOLVES} test cases:\n"
                f"full problem average solve time: {np.array(full_times).mean()}"
                f"\nreduced problem average solve time (convex): {np.array(convex_times).mean()}"
                f"\nreduced problem average solves (convex): {np.array(convex_solves).mean()}.\n\n"
                f"\nreduced problem average solve time (classifier): {np.array(classifier_times).mean()}"
                f"\nreduced problem average solves (classifier): {np.array(classifier_solves).mean()}.\n\n"
                f"\nreduced problem average solve time (ridge): {np.array(ridge_times).mean()}"
                f"\nreduced problem average solves (ridge): {np.array(ridge_solves).mean()}.\n\n"
                f"\nreduced problem average solve time (xgboost): {np.array(xgboost_times).mean()}"
                f"\nreduced problem average solves (xgboost): {np.array(xgboost_solves).mean()}.\n\n")
        )
        logfile.close()
        
        