import os, pickle
import numpy as np
from oct2py import Oct2Py
import cyipopt
import warnings
from tqdm import trange
from pypower.ext2int import ext2int
from pypower.api import loadcase
from problemDefJITM import opfSocp
from problemDefJITMargin import opfSocpMargin
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
NUM_POINTS_INIT = 1000 # number of data points to generate initial set
NUM_POINTS_CONVEX = 3000 # number of convex hull points to generate
DIFF = args.diff # ratio variation of data

# random convex combination of weights of an array
def random_convex_combination(arr):
    weights = np.random.rand(arr.shape[0])
    weights /= weights.sum()
    return np.dot(weights, arr)

# main

if __name__ == "__main__":
    
    default_diffs = {
        'pglib_opf_case118_ieee':0.75,
        'pglib_opf_case793_goc':0.75,
        'pglib_opf_case1354_pegase':0.75,
        'pglib_opf_case2312_goc':0.75,
        'pglib_opf_case4601_goc':0.75,
        'pglib_opf_case10000_goc':0.75
    }
    
    # get mpi rank
    comm = MPI.COMM_WORLD
    mpi_rank = comm.Get_rank()
    mpi_size = comm.Get_size()
    NUM_POINTS = NUM_POINTS_INIT // mpi_size
    NUM_POINTS_2 = NUM_POINTS_CONVEX // mpi_size
    
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
        
        if args.diff < 0:
            args.diff = default_diffs[cn]
        
        optObj = opfSocp(this_case,cn)
        cub, clb = optObj.calc_cons_bounds()
        xub, xlb = optObj.calc_var_bounds()
        
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
        input_x_1, duals_1 = [], []
        for pt_idx in (t:=trange(NUM_POINTS)):
            
            # set random seed
            np.random.seed(int((pt_idx+1.21)*(mpi_rank+1.63)*(3745.99564)))
            
            # get pd, qd and perturb
            pd,qd = optObj.get_loads()
            
            dpd,dqd = (1-DIFF + 2*np.random.rand(*pd.shape)*DIFF)*pd, (1-DIFF + 2*np.random.rand(*qd.shape)*DIFF)*qd
            optObj.change_loads(dpd,dqd)

            # solve problem
            _, info = prob.solve(info_base['x'],lagrange=info_base['mult_g'].tolist(),zl=info_base['mult_x_L'].tolist(),zu=info_base['mult_x_U'].tolist())
            if info['status'] == 0:
                input_x_data = {'pd':dpd,'qd':dqd,'flow_lim_f':optObj.flow_lim,'flow_lim_t':optObj.flow_lim}
                input_x_1.append(np.concatenate([itm[1] for itm in input_x_data.items()],axis=0))
                dual = info['mult_g'][np.concatenate([optObj.cidx[consn] for consn in ['balance_real','balance_reac','flow_f','flow_t']])]
                duals_1.append(dual)
            
            # output status
            t.set_description(f"Initial solve, point: {pt_idx+1}/{NUM_POINTS}, status: {info['status']}, process: ({mpi_rank+1}/{mpi_size}).")
            
        # generate convex hull points
        concat_params = np.array(input_x_1)
        
        # generate convex hull points
        input_x_2, duals_2 = [], []
        if concat_params.shape[0] != 0:
            for pt_idx in (t:=trange(NUM_POINTS_2)):
                
                # set random seed
                np.random.seed(int((pt_idx+1.66)*(mpi_rank+1.21)*(3745.6634)))
                
                # convex combination of pd, qd and perturb
                d_rhs = random_convex_combination(concat_params)
                dpd, dqd = d_rhs[:optObj.n_bus], d_rhs[optObj.n_bus:2*optObj.n_bus]
                optObj.change_loads(dpd,dqd)

                # solve problem
                _, info = prob.solve(info_base['x'],lagrange=info_base['mult_g'].tolist(),zl=info_base['mult_x_L'].tolist(),zu=info_base['mult_x_U'].tolist())
                if info['status'] == 0:
                    input_x_data = {'pd':dpd,'qd':dqd,'flow_lim_f':optObj.flow_lim,'flow_lim_t':optObj.flow_lim}
                    input_x_2.append(np.concatenate([itm[1] for itm in input_x_data.items()],axis=0))
                    dual = info['mult_g'][np.concatenate([optObj.cidx[consn] for consn in ['balance_real','balance_reac','flow_f','flow_t']])]
                    duals_2.append(dual)
                
                # output status
                t.set_description(f"Convex solve, point: {pt_idx+1}/{NUM_POINTS}, status: {info['status']}, process: ({mpi_rank+1}/{mpi_size}).")
        
        # combine
        input_x = input_x_1 + input_x_2
        duals = duals_1 + duals_2
            
        # save data
        os.makedirs(os.getcwd()+f'/data2',exist_ok=True)
        if len(input_x) > 0:
            np.savez_compressed(os.getcwd()+f'/data2/{cn}_inp_{mpi_rank}.npz',data=np.array(input_x))
            np.savez_compressed(os.getcwd()+f'/data2/{cn}_dual_{mpi_rank}.npz',data=np.array(duals))