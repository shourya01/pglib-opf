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
NUM_SOLVES = 100
VIOL_THRES = 1e-5

if __name__ == "__main__":
    
    # get all cases in current directory
    current_directory = os.getcwd()+'/'
    # current_directory = '/home/sbose/pglib-opf/' # for running on BEBOP
    all_files_and_directories = os.listdir(current_directory)
    # three specific cases
    case_files = [current_directory+i for i in ['pglib_opf_case118_ieee.m',
    # ]]
                                                'pglib_opf_case793_goc.m'
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
        moe_out = np.load(os.getcwd()+f'/saved/{cn}_out_moe.npz')['data'] # moe
        convex_out = np.load(os.getcwd()+f'/saved/{cn}_out_convex.npz')['data'] # convex
        classifier_out = np.load(os.getcwd()+f'/saved/{cn}_out_classifier.npz')['data'].astype(int) # classifier
        ridge_out = np.load(os.getcwd()+f'/saved/{cn}_out_ridge.npz')['data'].astype(int) # ridge
        if cn != 'pglib_opf_case10000_goc':
            xgboost_out = np.load(os.getcwd()+f'/saved/{cn}_out_xgboost.npz')['data'].astype(int) # xgboost
        else:
            xgboost_out = np.ones_like(ridge_out).astype(int) # no xgboost for 10000
            
        # set up relevant indices for reduced solves
        # inequality indices for optimization object constraints
        ineqidx = np.zeros(optObj.cons_size) # nonmodel inequalities
        ineqidx[np.concatenate([optObj.cidx['flow_f'],optObj.cidx['flow_t']])] = 1
        ineqidx = ineqidx.astype(bool)
        # indices for non model inequalities
        nmineq = np.ones(2*optObj.n_bus+2*optObj.n_branch)
        nmineq[:2*optObj.n_bus] = 0
        nmineq = nmineq.astype(bool).tolist()
        nmi_cons_size = 2*optObj.n_branch
        
        # set up lists to record time
        full_times = []
        moe_times, moe_solves = [], []
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
            
            # MIXTURE OF EXPERTS
            
            # print
            print("Mixture of Experts:")
            
            # marker
            moe_marker = moe_out[sidx,:][nmineq]
            
            # calculate first solve for moe
            optObjR = opfSocpR(this_case,moe_marker,cn)
            optObjR.change_loads(pd,qd)
            pdk, psk = problem_def_kwargs(optObjR,*optObjR.calc_var_bounds(),*optObjR.calc_cons_bounds()), problem_settings_kwargs_reduced(optObjR)
            prob = cyipopt.Problem(**pdk)
            for k,v in psk.items():
                prob.add_option(k,v)
            start = time()
            xmoefirst, infomoefirst = prob.solve(optObjR.calc_x0_flatstart())
            obj_moefirst = infomoefirst['obj_val']
            end = time()
            timemoe = end-start
            solvesmoe = 1
            
            # infer violated constraints
            inferred_nmineq = optObj.constraints(xmoefirst).clip(min=0)[ineqidx]
            inferred_viols = np.where(np.abs(inferred_nmineq)>VIOL_THRES,1,0)
            
            if inferred_viols.sum() == 0:
                print(f"Reduced problem solved in {(timemoe):.5f}s with 1 solve, objective {obj_moefirst} with {inferred_viols.sum()} violations.")
                moe_times.append(timemoe)
                moe_solves.append(solvesmoe)
            else:
                print(f"Detected {inferred_viols.sum()} violations with objective {obj_moefirst}.")
                moe_marker = np.where(moe_marker[:nmi_cons_size]+inferred_viols>0,1,0)
                optObjR = opfSocpR(this_case,moe_marker,cn)
                optObjR.change_loads(pd,qd)
                pdk, psk = problem_def_kwargs(optObjR,*optObjR.calc_var_bounds(),*optObjR.calc_cons_bounds()), problem_settings_kwargs_reduced(optObjR)
                prob = cyipopt.Problem(**pdk)
                for k,v in psk.items():
                    prob.add_option(k,v)
                start = time()
                xmoesecond, infomoesecond = prob.solve(optObjR.calc_x0_flatstart())
                obj_moesecond = infomoesecond['obj_val']
                end = time()
                timemoe += end-start
                solvesmoe += 1
                inferred_nmineq = optObj.constraints(xmoesecond).clip(min=0)[ineqidx]
                inferred_viols = np.where(np.abs(inferred_nmineq)>VIOL_THRES,1,0)
                print(f"Reduced problem solved in {(timemoe):.5f}s with 2 solves, objective {obj_moesecond} with {inferred_viols.sum()} violations.")
                moe_times.append(timemoe)
                moe_solves.append(solvesmoe)
            
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
                convex_marker = np.where(convex_marker+inferred_viols>0,1,0)
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
                classifier_marker = np.where(classifier_marker+inferred_viols>0,1,0)
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
                ridge_marker = np.where(ridge_marker[:nmi_cons_size]+inferred_viols>0,1,0)
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
                xgboost_marker = np.where(xgboost_marker[:nmi_cons_size]+inferred_viols>0,1,0)
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
        moe_times, moe_solves = np.array(moe_times).astype(np.double), np.array(moe_solves).astype(np.double)
        convex_times, convex_solves = np.array(convex_times).astype(np.double), np.array(convex_solves).astype(np.double)
        classifier_times, classifier_solves = np.array(classifier_times).astype(np.double), np.array(classifier_solves).astype(np.double)
        ridge_times, ridge_solves = np.array(ridge_times).astype(np.double), np.array(ridge_solves).astype(np.double)
        xgboost_times, xgboost_solves = np.array(xgboost_times).astype(np.double), np.array(xgboost_solves).astype(np.double)
        
        if mpi_rank == 0:
            sz = []
            # sz.append(np.array(full_times.size,dtype=int))
            for rk in range(mpi_size-1):
                this_sz = np.array(full_times.size,dtype=int)
                comm.Recv([this_sz,MPI.INT],source=rk+1,tag=621)
                sz.append(this_sz)
        else:
            comm.Send([np.array(full_times.size,dtype=int),MPI.INT],dest=0,tag=621)
            
            
        if mpi_rank == 0:
            ftimes, mtimes, ctimes, cltimes, rtimes, xtimes = [full_times], [moe_times], [convex_times], [classifier_times], [ridge_times], [xgboost_times]
            msolves, csolves, clsolves, rsolves, xsolves = [moe_solves], [convex_solves], [classifier_solves], [ridge_solves], [xgboost_solves]
            for rk in range(mpi_size-1):
                this_data1 = np.empty(sz[rk],dtype=np.double)
                this_data1.fill(np.nan)
                comm.Recv([this_data1,MPI.DOUBLE],source=rk+1,tag=0 + 11*(rk+1))
                ftimes.append(this_data1)
                this_data2 = np.empty(sz[rk],dtype=np.double)
                this_data2.fill(np.nan)
                comm.Recv([this_data2,MPI.DOUBLE],source=rk+1,tag=1 + 11*(rk+1))
                mtimes.append(this_data2)
                this_data3 = np.empty(sz[rk],dtype=np.double)
                this_data3.fill(np.nan)
                comm.Recv([this_data3,MPI.DOUBLE],source=rk+1,tag=2 + 11*(rk+1))
                ctimes.append(this_data3)
                this_data4 = np.empty(sz[rk],dtype=np.double)
                this_data4.fill(np.nan)
                comm.Recv([this_data4,MPI.DOUBLE],source=rk+1,tag=3 + 11*(rk+1))
                cltimes.append(this_data4)
                this_data5 = np.empty(sz[rk],dtype=np.double)
                this_data5.fill(np.nan)
                comm.Recv([this_data5,MPI.DOUBLE],source=rk+1,tag=4 + 11*(rk+1))
                rtimes.append(this_data5)
                this_data6 = np.empty(sz[rk],dtype=np.double)
                this_data6.fill(np.nan)
                comm.Recv([this_data6,MPI.DOUBLE],source=rk+1,tag=5 + 11*(rk+1))
                xtimes.append(this_data6)
                this_data7 = np.empty(sz[rk],dtype=np.double)
                this_data7.fill(np.nan)
                comm.Recv([this_data7,MPI.DOUBLE],source=rk+1,tag=6 + 11*(rk+1))
                msolves.append(this_data7)
                this_data8 = np.empty(sz[rk],dtype=np.double)
                this_data8.fill(np.nan)
                comm.Recv([this_data8,MPI.DOUBLE],source=rk+1,tag=7 + 11*(rk+1))
                csolves.append(this_data8)
                this_data9 = np.empty(sz[rk],dtype=np.double)
                this_data9.fill(np.nan)
                comm.Recv([this_data9,MPI.DOUBLE],source=rk+1,tag=8 + 11*(rk+1))
                clsolves.append(this_data9)
                this_data10 = np.empty(sz[rk],dtype=np.double)
                this_data10.fill(np.nan)
                comm.Recv([this_data10,MPI.DOUBLE],source=rk+1,tag=9 + 11*(rk+1))
                rsolves.append(this_data10)
                this_data11 = np.empty(sz[rk],dtype=np.double)
                this_data11.fill(np.nan)
                comm.Recv([this_data11,MPI.DOUBLE],source=rk+1,tag=10 + 11*(rk+1))
                xsolves.append(this_data11)
        else:
            comm.Send([full_times,MPI.DOUBLE],dest=0,tag=0+11*mpi_rank)
            comm.Send([moe_times,MPI.DOUBLE],dest=0,tag=1+11*mpi_rank)
            comm.Send([convex_times,MPI.DOUBLE],dest=0,tag=2+11*mpi_rank)
            comm.Send([classifier_times,MPI.DOUBLE],dest=0,tag=3+11*mpi_rank)
            comm.Send([ridge_times,MPI.DOUBLE],dest=0,tag=4+11*mpi_rank)
            comm.Send([xgboost_times,MPI.DOUBLE],dest=0,tag=5+11*mpi_rank)
            comm.Send([moe_solves,MPI.DOUBLE],dest=0,tag=6+11*mpi_rank)
            comm.Send([convex_solves,MPI.DOUBLE],dest=0,tag=7+11*mpi_rank)
            comm.Send([classifier_solves,MPI.DOUBLE],dest=0,tag=8+11*mpi_rank)
            comm.Send([ridge_solves,MPI.DOUBLE],dest=0,tag=9+11*mpi_rank)
            comm.Send([xgboost_solves,MPI.DOUBLE],dest=0,tag=10+11*mpi_rank)    
                    
                
        if mpi_rank == 0:        
            full_times = np.concatenate(ftimes)
            moe_times = np.concatenate(mtimes)
            convex_times = np.concatenate(ctimes)
            classifier_times = np.concatenate(cltimes)
            ridge_times = np.concatenate(rtimes)
            xgboost_times = np.concatenate(xtimes)
            moe_solves = np.concatenate(msolves)
            convex_solves = np.concatenate(csolves)
            classifier_solves = np.concatenate(clsolves)
            ridge_solves = np.concatenate(rsolves)
            xgboost_solves = np.concatenate(xsolves)
            # print((f"MoE solves: {moe_solves}, shape: {moe_solves.shape},\n"
            #       f"convex solves: {convex_solves}, shape: {convex_solves.shape}\n"
            #       f"classifier_solves: {classifier_solves}, shape: {classifier_solves.shape}\n"
            #       f"ridge_solves: {ridge_solves}, shape: {ridge_solves.shape}\n"
            #       f"xgboost_solves: {xgboost_solves}, shape: {xgboost_solves.shape}."))
            # print stats
            print(
                    (f"For case {cn}, solved {NUM_SOLVES} test cases:\n"
                    f"full problem average solve time: {np.array(full_times[full_times!=0]).mean()}"
                    f"\nreduced problem average solve time (moe): {np.array(moe_times[moe_times!=0]).mean()}"
                    f"\nreduced problem average solves (moe): {np.array(moe_solves[moe_solves!=0]).mean()}."
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
                    f"\nreduced problem average solve time (moe): {np.array(moe_times[moe_times!=0]).mean()}"
                    f"\nreduced problem average solves (moe): {np.array(moe_solves[moe_solves!=0]).mean()}."
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
        
        