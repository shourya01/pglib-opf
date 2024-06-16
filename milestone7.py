import os, pickle
import numpy as np
import torch.nn.functional as F
from oct2py import Oct2Py
import cyipopt
import warnings
from tqdm import trange
from pypower.ext2int import ext2int
from pypower.api import loadcase
from problemDefJITM2 import opfSocp
from problemDefJITR2 import opfSocpR
from utils import make_data_parallel
from tqdm import tqdm, trange
from itertools import zip_longest
import gc
from mpi4py import MPI
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
NUM_SOLVES = 50
VIOL_THRES = 1e-5

if __name__ == "__main__":
    
    # get all cases in current directory
    current_directory = os.getcwd()+'/'
    # current_directory = '/home/sbose/pglib-opf/' # for running on BEBOP
    all_files_and_directories = os.listdir(current_directory)
    # three specific cases
    case_files = [current_directory+i for i in ['pglib_opf_case118_ieee.m','pglib_opf_case793_goc.m'
                                                ,'pglib_opf_case1354_pegase.m'
                                                ,'pglib_opf_case2312_goc.m','pglib_opf_case4601_goc.m','pglib_opf_case10000_goc.m']]
    # case_files = [current_directory+i for i in ['pglib_opf_case2312_goc.m','pglib_opf_case10000_goc.m']]
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
        'tol':1e-8,
        'mu_max':1e-0,
        'mu_init':1e-0,
        'nlp_lower_bound_inf':-obj.LARGE_NUMBER+1,
        'nlp_upper_bound_inf':obj.LARGE_NUMBER-1,
        'print_level':0
    }
    problem_settings_kwargs_reduced = lambda obj:{
        'tol':1e-5,
        'mu_max':1e-0,
        'mu_init':1e-0,
        'nlp_lower_bound_inf':-obj.LARGE_NUMBER+1,
        'nlp_upper_bound_inf':obj.LARGE_NUMBER-1,
        'print_level':0,
        'fixed_variable_treatment':'make_parameter_nodual'
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
    
    # to distribute across MPI process
    comm = MPI.COMM_WORLD
    mpi_rank = comm.Get_rank()
    mpi_size = comm.Get_size()
    pt_split = np.array_split(np.arange(NUM_SOLVES),mpi_size)
    this_split = pt_split[mpi_rank]
    if mpi_rank == 0:
        # write
        logfile = open('perf2.txt','w')
            
    for cn,this_case in zip(casenames,cases):
        
        print(f"-----\nSolving case {cn}\n-----\n\n")
        
        
        # modify the input
        optObj = opfSocp(this_case,cn) # generate object
        
        # output  data for different methods
        inp_data = np.load(os.getcwd()+f'/saved/{cn}_test_inp.npz')['data'] # input
        gt_data = np.load(os.getcwd()+f'/saved/{cn}_test_gt.npz')['data'] # ground truth
        convex_out = np.load(os.getcwd()+f'/saved/{cn}_out_convex.npz')['data'] # convex
        classifier_out = np.load(os.getcwd()+f'/saved/{cn}_out_classifier.npz')['data'].astype(int) # classifier
        ridge_out = np.load(os.getcwd()+f'/saved/{cn}_out_ridge.npz')['data'].astype(int) # ridge
        if cn != 'pglib_opf_case10000_goc':
            xgboost_out = np.load(os.getcwd()+f'/saved/{cn}_out_xgboost.npz')['data'].astype(int) # xgboost
        else:
            xgboost_out = np.ones_like(ridge_out).astype(int) # no xgboost for 10000
            
        # set up relevant indices for reduced solves
        ineqidx = ((1-optObj.is_model)*(1-optObj.is_equality)).astype(bool) # nonmodel inequalities
        nmineq = np.ones(2*optObj.n_bus+4*optObj.n_branch+2*optObj.in_size)
        nmineq[:2*optObj.n_bus] = 0
        nmineq = nmineq.astype(bool).tolist()
        nmi_cons_size = 4*optObj.n_branch
        
        # set up lists to record time
        full_times = []
        convex_times, convex_solves = [], []
        classifier_times, classifier_solves = [], []
        ridge_times, ridge_solves = [], []
        xgboost_times, xgboost_solves = [], []
        optObj = opfSocp(this_case,cn)
        
        for sidx in range(NUM_SOLVES):
            
            # mpi
            if sidx not in this_split:
                continue
            
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
            pdk, psk = problem_def_kwargs(optObjR,*optObjR.calc_var_bounds(),*optObjR.calc_cons_bounds()), problem_settings_kwargs_reduced(optObjR)
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
            inferred_nmineq = optObj.constraints(xcofirst).clip(min=0)[ineqidx]
            inferred_viols = np.where(np.abs(inferred_nmineq)>VIOL_THRES,1,0)
            
            if inferred_viols.sum() == 0:
                print(f"Reduced problem solved in {(timeco):.5f}s with 1 solve, objective {obj_cofirst} with {inferred_viols.sum()} violations.")
                convex_times.append(timeco)
                convex_solves.append(solvesco)
            else:
                print(f"Detected {inferred_viols.sum()} violations with objective {obj_cofirst}.")
                convex_marker = np.concatenate([np.where(convex_marker[:nmi_cons_size]+inferred_viols>0,1,0),convex_marker[nmi_cons_size:]])
                optObjR = opfSocpR(this_case,convex_marker,cn)
                optObjR.change_loads(pd,qd)
                pdk, psk = problem_def_kwargs(optObjR,*optObjR.calc_var_bounds(),*optObjR.calc_cons_bounds()), problem_settings_kwargs_reduced(optObjR)
                prob = cyipopt.Problem(**pdk)
                for k,v in psk.items():
                    prob.add_option(k,v)
                start = time()
                xcosecond, infocosecond = prob.solve(optObjR.calc_x0_flatstart())
                obj_cosecond = infocosecond['obj_val']
                end = time()
                timeco += end-start
                solvesco += 1
                inferred_nmineq = optObj.constraints(xcosecond).clip(min=0)[ineqidx]
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
            pdk, psk = problem_def_kwargs(optObjR,*optObjR.calc_var_bounds(),*optObjR.calc_cons_bounds()), problem_settings_kwargs_reduced(optObjR)
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
            inferred_nmineq = optObj.constraints(xclfirst).clip(min=0)[ineqidx]
            inferred_viols = np.where(np.abs(inferred_nmineq)>VIOL_THRES,1,0)
            
            if inferred_viols.sum() == 0:
                print(f"Reduced problem solved in {(timecl):.5f}s with 1 solve, objective {obj_clfirst} with {inferred_viols.sum()} violations.")
                classifier_times.append(timecl)
                classifier_solves.append(solvescl)
            else:
                print(f"Detected {inferred_viols.sum()} violations with objective {obj_clfirst}.")
                classifier_marker = np.concatenate([np.where(classifier_marker[:nmi_cons_size]+inferred_viols>0,1,0),classifier_marker[nmi_cons_size:]])
                optObjR = opfSocpR(this_case,classifier_marker,cn)
                optObjR.change_loads(pd,qd)
                pdk, psk = problem_def_kwargs(optObjR,*optObjR.calc_var_bounds(),*optObjR.calc_cons_bounds()), problem_settings_kwargs_reduced(optObjR)
                prob = cyipopt.Problem(**pdk)
                for k,v in psk.items():
                    prob.add_option(k,v)
                start = time()
                xclsecond, infoclsecond = prob.solve(optObjR.calc_x0_flatstart())
                obj_clsecond = infoclsecond['obj_val']
                end = time()
                timecl += end-start
                solvescl += 1
                inferred_nmineq = optObj.constraints(xclsecond).clip(min=0)[ineqidx]
                inferred_viols = np.where(np.abs(inferred_nmineq)>VIOL_THRES,1,0)
                print(f"Reduced problem solved in {(timecl):.5f}s with 2 solves, objective {obj_clsecond} with {inferred_viols.sum()} violations.")
                classifier_times.append(timecl)
                classifier_solves.append(solvescl)
                
            # RIDGE
            
            # print
            print("Ridge:")
            
            # marker
            ridge_marker = ridge_out[sidx,:][nmineq]
            
            # calculate first solve for classifier
            optObjR = opfSocpR(this_case,ridge_marker,cn)
            optObjR.change_loads(pd,qd)
            pdk, psk = problem_def_kwargs(optObjR,*optObjR.calc_var_bounds(),*optObjR.calc_cons_bounds()), problem_settings_kwargs_reduced(optObjR)
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
            inferred_nmineq = optObj.constraints(xrgfirst).clip(min=0)[ineqidx]
            inferred_viols = np.where(np.abs(inferred_nmineq)>VIOL_THRES,1,0)
            
            if inferred_viols.sum() == 0:
                print(f"Reduced problem solved in {(timerg):.5f}s with 1 solve, objective {obj_rgfirst} with {inferred_viols.sum()} violations.")
                ridge_times.append(timerg)
                ridge_solves.append(solvesrg)
            else:
                print(f"Detected {inferred_viols.sum()} violations with objective {obj_rgfirst}.")
                ridge_marker = np.concatenate([np.where(ridge_marker[:nmi_cons_size]+inferred_viols>0,1,0),ridge_marker[nmi_cons_size:]])
                optObjR = opfSocpR(this_case,ridge_marker,cn)
                optObjR.change_loads(pd,qd)
                pdk, psk = problem_def_kwargs(optObjR,*optObjR.calc_var_bounds(),*optObjR.calc_cons_bounds()), problem_settings_kwargs_reduced(optObjR)
                prob = cyipopt.Problem(**pdk)
                for k,v in psk.items():
                    prob.add_option(k,v)
                start = time()
                xrgsecond, inforgsecond = prob.solve(optObjR.calc_x0_flatstart())
                obj_rgsecond = inforgsecond['obj_val']
                end = time()
                timerg += end-start
                solvesrg  += 1
                inferred_nmineq = optObj.constraints(xrgsecond).clip(min=0)[ineqidx]
                inferred_viols = np.where(np.abs(inferred_nmineq)>VIOL_THRES,1,0)
                print(f"Reduced problem solved in {(timerg):.5f}s with 2 solves, objective {obj_rgsecond} with {inferred_viols.sum()} violations.")
                ridge_times.append(timerg)
                ridge_solves.append(solvesrg)
                
            # XGBOOST
            
            # print
            print("XGBoost:")
            
            # marker
            xgboost_marker = xgboost_out[sidx,:][nmineq]
            
            # calculate first solve for classifier
            optObjR = opfSocpR(this_case,xgboost_marker,cn)
            optObjR.change_loads(pd,qd)
            pdk, psk = problem_def_kwargs(optObjR,*optObjR.calc_var_bounds(),*optObjR.calc_cons_bounds()), problem_settings_kwargs_reduced(optObjR)
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
            inferred_nmineq = optObj.constraints(xxgfirst).clip(min=0)[ineqidx]
            inferred_viols = np.where(np.abs(inferred_nmineq)>VIOL_THRES,1,0)
            
            if inferred_viols.sum() == 0:
                print(f"Reduced problem solved in {(timexg):.5f}s with 1 solve, objective {obj_xgfirst} with {inferred_viols.sum()} violations.\n")
                xgboost_times.append(timexg)
                xgboost_solves.append(solvesxg)
            else:
                print(f"Detected {inferred_viols.sum()} violations with objective {obj_rgfirst}.")
                xgboost_marker = np.concatenate([np.where(xgboost_marker[:nmi_cons_size]+inferred_viols>0,1,0),xgboost_marker[nmi_cons_size:]])
                optObjR = opfSocpR(this_case,xgboost_marker,cn)
                optObjR.change_loads(pd,qd)
                pdk, psk = problem_def_kwargs(optObjR,*optObjR.calc_var_bounds(),*optObjR.calc_cons_bounds()), problem_settings_kwargs_reduced(optObjR)
                prob = cyipopt.Problem(**pdk)
                for k,v in psk.items():
                    prob.add_option(k,v)
                start = time()
                xxgsecond, infoxgsecond = prob.solve(optObjR.calc_x0_flatstart())
                obj_xgsecond = infoxgsecond['obj_val']
                end = time()
                timexg += end-start
                solvesxg  += 1
                inferred_nmineq = optObj.constraints(xxgsecond).clip(min=0)[ineqidx]
                inferred_viols = np.where(np.abs(inferred_nmineq)>VIOL_THRES,1,0)
                print(f"Reduced problem solved in {(timexg):.5f}s with 2 solves, objective {obj_xgsecond} with {inferred_viols.sum()} violations.\n")
                xgboost_times.append(timexg)
                xgboost_solves.append(solvesxg)
                
        full_times = np.array(full_times)
        convex_times, convex_solves = np.array(convex_times), np.array(convex_solves)
        classifier_times, classifier_solves = np.array(classifier_times), np.array(classifier_solves)
        ridge_times, ridge_solves = np.array(ridge_times), np.array(ridge_solves)
        xgboost_times, xgboost_solves = np.array(xgboost_times), np.array(xgboost_solves)
        
        if mpi_rank == 0:
            sz = []
            # sz.append(np.array(full_times.size,dtype=int))
            for rk in range(mpi_size-1):
                this_sz = np.array(full_times.size,dtype=int)
                comm.Recv([this_sz,MPI.INT],source=rk+1,tag=500)
                sz.append(this_sz)
        else:
            comm.Send([np.array(full_times.size,dtype=int),MPI.INT],dest=0,tag=500)
            
        if mpi_rank == 0:
            ftimes, ctimes, cltimes, rtimes, xtimes = [full_times], [convex_times], [classifier_times], [ridge_times], [xgboost_times]
            csolves, clsolves, rsolves, xsolves = [convex_solves], [classifier_solves], [ridge_solves], [xgboost_solves]
            for rk in range(mpi_size-1):
                this_data1 = np.empty(sz[rk],dtype=np.float64)
                comm.Recv([this_data1,MPI.DOUBLE],source=rk+1,tag=0)
                ftimes.append(this_data1)
                this_data2 = np.empty(sz[rk],dtype=np.float64)
                comm.Recv([this_data2,MPI.DOUBLE],source=rk+1,tag=1)
                ctimes.append(this_data2)
                this_data3 = np.empty(sz[rk],dtype=np.float64)
                comm.Recv([this_data3,MPI.DOUBLE],source=rk+1,tag=2)
                cltimes.append(this_data3)
                this_data4 = np.empty(sz[rk],dtype=np.float64)
                comm.Recv([this_data4,MPI.DOUBLE],source=rk+1,tag=3)
                rtimes.append(this_data4)
                this_data5 = np.empty(sz[rk],dtype=np.float64)
                comm.Recv([this_data5,MPI.DOUBLE],source=rk+1,tag=4)
                xtimes.append(this_data5)
                this_data6 = np.empty(sz[rk],dtype=np.float64)
                comm.Recv([this_data6,MPI.DOUBLE],source=rk+1,tag=5)
                csolves.append(this_data6)
                this_data7 = np.empty(sz[rk],dtype=np.float64)
                comm.Recv([this_data7,MPI.DOUBLE],source=rk+1,tag=6)
                clsolves.append(this_data7)
                this_data8 = np.empty(sz[rk],dtype=np.float64)
                comm.Recv([this_data8,MPI.DOUBLE],source=rk+1,tag=7)
                rsolves.append(this_data8)
                this_data9 = np.empty(sz[rk],dtype=np.float64)
                comm.Recv([this_data9,MPI.DOUBLE],source=rk+1,tag=8)
                xsolves.append(this_data9)
        else:
            comm.Send([full_times,MPI.DOUBLE],dest=0,tag=0)
            comm.Send([convex_times,MPI.DOUBLE],dest=0,tag=1)
            comm.Send([classifier_times,MPI.DOUBLE],dest=0,tag=2)
            comm.Send([ridge_times,MPI.DOUBLE],dest=0,tag=3)
            comm.Send([xgboost_times,MPI.DOUBLE],dest=0,tag=4)
            comm.Send([convex_solves.astype(np.float64),MPI.DOUBLE],dest=0,tag=5)
            comm.Send([classifier_solves.astype(np.float64),MPI.DOUBLE],dest=0,tag=6)
            comm.Send([ridge_solves.astype(np.float64),MPI.DOUBLE],dest=0,tag=7)
            comm.Send([xgboost_solves.astype(np.float64),MPI.DOUBLE],dest=0,tag=8)    
                    
                
        if mpi_rank == 0:        
            full_times = np.concatenate(ftimes)
            convex_times = np.concatenate(ctimes)
            classifier_times = np.concatenate(cltimes)
            ridge_times = np.concatenate(rtimes)
            xgboost_times = np.concatenate(xtimes)
            convex_solves = np.concatenate(csolves)
            classifier_solves = np.concatenate(clsolves)
            ridge_solves = np.concatenate(rsolves)
            xgboost_solves = np.concatenate(xsolves)
            print(f"Convex solves: {convex_solves}, classifier_solves: {classifier_solves}, ridge_solves: {ridge_solves}, xgboost_solves: {xgboost_solves}.")
            # print stats
            print(
                    (f"For case {cn}, solved {NUM_SOLVES} test cases:\n"
                    f"full problem average solve time: {np.array(full_times[full_times!=0]).mean()}"
                    f"\nreduced problem average solve time (convex): {np.array(convex_times[convex_times!=0]).mean()}"
                    f"\nreduced problem average solves (convex): {np.array(convex_solves[convex_solves!=0]).mean()}."
                    f"\nreduced problem average solve time (classifier): {np.array(classifier_times[classifier_times!=0]).mean()}"
                    f"\nreduced problem average solves (classifier): {np.array(classifier_solves[classifier_solves!=0]).mean()}."
                    f"\nreduced problem average solve time (ridge): {np.array(ridge_times[ridge_times!=0]).mean()}"
                    f"\nreduced problem average solves (ridge): {np.array(ridge_solves[ridge_solves!=0]).mean()}."
                    f"\nreduced problem average solve time (xgboost): {np.array(xgboost_times[xgboost_times!=0]).mean()}"
                    f"\nreduced problem average solves (xgboost): {np.array(xgboost_solves[xgboost_solves!=0]).mean()}.\n\n")
                ,flush=True)
            logfile.write(
                    (f"For case {cn}, solved {NUM_SOLVES} test cases:\n"
                    f"full problem average solve time: {np.array(full_times[full_times!=0]).mean()}"
                    f"\nreduced problem average solve time (convex): {np.array(convex_times[convex_times!=0]).mean()}"
                    f"\nreduced problem average solves (convex): {np.array(convex_solves[convex_solves!=0]).mean()}."
                    f"\nreduced problem average solve time (classifier): {np.array(classifier_times[classifier_times!=0]).mean()}"
                    f"\nreduced problem average solves (classifier): {np.array(classifier_solves[classifier_solves!=0]).mean()}."
                    f"\nreduced problem average solve time (ridge): {np.array(ridge_times[ridge_times!=0]).mean()}"
                    f"\nreduced problem average solves (ridge): {np.array(ridge_solves[ridge_solves!=0]).mean()}."
                    f"\nreduced problem average solve time (xgboost): {np.array(xgboost_times[xgboost_times!=0]).mean()}"
                    f"\nreduced problem average solves (xgboost): {np.array(xgboost_solves[xgboost_solves!=0]).mean()}.\n\n")
            )
        comm.Barrier()
    if mpi_rank == 0:
        logfile.close()
        
        