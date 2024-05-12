
from pypower.api import loadcase, runopf, ppoption
from pypower.idx_brch import *
from pypower.idx_bus import *
from pypower.idx_gen import *
from pypower.idx_cost import *
from pypower.ext2int import ext2int
import oct2py
import numpy as np
from numpy import r_
import math
import os, sys
from tqdm import tqdm
from oct2py import Oct2Py
octave = Oct2Py()
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import pickle
import shutil
import cvxpy as cp
from contextlib import contextmanager
import warnings
import cyipopt
from problemDefJITM2 import opfSocp
from problemDefJITMargin import opfSocpMargin
from copy import deepcopy
import cProfile
import cython
# suppress ComplexWarning
warnings.simplefilter("ignore", np.ComplexWarning)
# check for latex and configure matplotlib accordingly
if shutil.which('latex') is None:
    LATEX_AVAILABLE = False
else:
    LATEX_AVAILABLE = True
if LATEX_AVAILABLE:
    plt.rcParams['font.size'] = 14
    plt.rcParams['text.usetex'] = True
    plt.rcParams['text.latex.preamble'] = r'\usepackage{amsmath,amssymb}'

# to suppress output from functions    
@contextmanager
def suppress_stdout():
    with open(os.devnull, 'w') as devnull:
        old_stdout = sys.stdout
        sys.stdout = devnull
        try:  
            yield
        finally:
            sys.stdout = old_stdout    

# user options 
MAX_BUS = 40 # upper limit of number of buses in cases to be considered
RATE = 'A' # which line rating to use ('A','B','C')

if __name__ == "__main__":
    
    # get all cases in current directory
    current_directory = os.getcwd()+'/'
    # current_directory = '/home/sbose/pglib-opf/' # for running on BEBOP
    all_files_and_directories = os.listdir(current_directory)
    case_files = [os.path.join(current_directory, f) for f in all_files_and_directories if f.endswith('.m') and os.path.isfile(os.path.join(current_directory, f))]

    cases, casenames = [], []
    cases_full, casenames_full = [], []
    for cf in case_files:
        with suppress_stdout():
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
    
    # RECORDS FOR SOLVED AND UNSOLVED
    solved, unsolved = 0,0

    # FUNCTIONS TO COMPUTE ABSOLUTE VALUES # TEST
    absobj = lambda x,y: cp.sum(cp.hstack([cp.abs(cp.real(x)),cp.abs(cp.imag(x)),cp.abs(cp.real(y)),cp.abs(cp.imag(y))]))

    iterlist = [itm for itm in zip(casenames,cases)]
    for cn,this_case in iterlist:
        
        # Initialize
        print(f"\n--------\nSolving {cn}.\n--------\n",flush=True)
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
        prob.add_option('max_iter',2500)
        prob.add_option('mumps_mem_percent',25000)
        prob.add_option('mu_max',1e-0)
        prob.add_option('mu_init',1e-0)
        prob.add_option('nlp_lower_bound_inf',-optObj.LARGE_NUMBER+1)
        prob.add_option('nlp_upper_bound_inf',optObj.LARGE_NUMBER-1)
        prob.add_option('derivative_test','second-order')
        
        # Solve ipopt problem
        # cProfile.run('x,info = prob.solve(optObj.calc_x0_vanilla())')
        x, info = prob.solve(optObj.calc_x0_flatstart())    
        
        # evaluate solution
        if info['status_msg'].startswith(b'Algorithm terminated successfully'):
            solved += 1
        else:
            unsolved += 1
            
        # Now resolve with warm start
        
        print(f"\n--------\nSolving {cn} with warm start.\n--------\n",flush=True)
        
        prob.add_option('mu_init',1e-7)
        prob.solve(info['x'],lagrange=info['mult_g'].tolist(),zl=info['mult_x_L'].tolist(),zu=info['mult_x_U'].tolist())
        
        # print(f"\n--------\nSolving {cn} with warm start and reduced constr.\n--------\n",flush=True)
        
        # cub[18] = optObj.LARGE_NUMBER
        # prob = cyipopt.Problem(
        #     n = optObj.in_size,
        #     m = optObj.cons_size,
        #     problem_obj=optObj,
        #     lb=xlb,
        #     ub=xub,
        #     cl=clb,
        #     cu=cub
        # )
        
        # # Setup solver options
        # prob.add_option('tol',1e-6)
        # prob.add_option('max_iter',2500)
        # prob.add_option('mumps_mem_percent',25000)
        # prob.add_option('mu_max',1e-0)
        # prob.add_option('mu_init',1e-7)
        # prob.add_option('nlp_lower_bound_inf',-optObj.LARGE_NUMBER+1)
        # prob.add_option('nlp_upper_bound_inf',optObj.LARGE_NUMBER-1)
        # prob.add_option('derivative_test','second-order')
        
        # prob.solve(info['x'],lagrange=info['mult_g'].tolist(),zl=info['mult_x_L'].tolist(),zu=info['mult_x_U'].tolist())
        
        # # Find min margin
        
        # print(f"\n--------\nSolving {cn} min margin for pd, qd.\n--------\n",flush=True)
        # optObj = opfSocpMargin(this_case,cn,margin_sense=-1)
        # cub, clb = optObj.calc_cons_bounds()
        # xub, xlb = optObj.calc_var_bounds()
        
        # print(f"cidx: {optObj.cidx}, vidx: {optObj.vidx}")
        
        # # Define IPOPT problem
        # prob = cyipopt.Problem(
        #     n = optObj.in_size,
        #     m = optObj.cons_size,
        #     problem_obj=optObj,
        #     lb=xlb,
        #     ub=xub,
        #     cl=clb,
        #     cu=cub
        # )
        
        # # Setup solver options
        # prob.add_option('tol',1e-6)
        # prob.add_option('max_iter',2500)
        # prob.add_option('mumps_mem_percent',25000)
        # prob.add_option('mu_max',1e-0)
        # prob.add_option('mu_init',1e-0)
        # prob.add_option('nlp_lower_bound_inf',-optObj.LARGE_NUMBER+1)
        # prob.add_option('nlp_upper_bound_inf',optObj.LARGE_NUMBER-1)
        # prob.add_option('derivative_test','second-order')
        
        # # solve
        # x, info = prob.solve(optObj.calc_x0_flatstart())  
        # minMarginPd = x[optObj.vidx['mPd']]
        # minMarginQd = x[optObj.vidx['mQd']]
        
        # # Find max margin
        
        # print(f"\n--------\nSolving {cn} min margin for pd, qd.\n--------\n",flush=True)
        # optObj = opfSocpMargin(this_case,cn,margin_sense=1)
        # cub, clb = optObj.calc_cons_bounds()
        # xub, xlb = optObj.calc_var_bounds()
        
        # # Define IPOPT problem
        # prob = cyipopt.Problem(
        #     n = optObj.in_size,
        #     m = optObj.cons_size,
        #     problem_obj=optObj,
        #     lb=xlb,
        #     ub=xub,
        #     cl=clb,
        #     cu=cub
        # )
        
        # # Setup solver options
        # prob.add_option('tol',1e-6)
        # prob.add_option('max_iter',2500)
        # prob.add_option('mumps_mem_percent',25000)
        # prob.add_option('mu_max',1e-0)
        # prob.add_option('mu_init',1e-0)
        # prob.add_option('nlp_lower_bound_inf',-optObj.LARGE_NUMBER+1)
        # prob.add_option('nlp_upper_bound_inf',optObj.LARGE_NUMBER-1)
        # prob.add_option('derivative_test','second-order')
        
        # # solve
        # x, info = prob.solve(optObj.calc_x0_flatstart())  
        # maxMarginPd = x[optObj.vidx['mPd']]
        # maxMarginQd = x[optObj.vidx['mQd']]
        
        # # plotting
        # pd, qd = optObj.get_loads()
        # fig, axs = plt.subplots(nrows=2,ncols=1,figsize=(10,8))
        
        # axs[0].plot(pd+maxMarginPd,'g--',label=r'$P_d^{\max}$')
        # axs[0].plot(pd-minMarginPd,'b-.',label=r'$Q_d^{\min}$')
        # axs[0].plot(pd,'k',linewidth=2,label=r'$P_d$')
        # axs[0].set_xlim([0,pd.size-1])
        # axs[0].set_xlabel('Bus_index')
        # axs[0].set_ylabel('Real demand pu')
        # axs[0].legend()
        
        # axs[1].plot(qd+maxMarginQd,'g--',label=r'$Q_d^{\max}$')
        # axs[1].plot(qd-minMarginQd,'b-.',label=r'$Q_d^{\min}$')
        # axs[1].plot(qd,'k',linewidth=2,label=r'$Q_d$')
        # axs[1].set_xlim([0,pd.size-1])
        # axs[1].set_xlabel('Bus_index')
        # axs[1].set_ylabel('Reactive demand pu')
        # axs[1].legend()
        
        # plt.savefig(f'{cn}_margins.pdf',format='pdf',bbox_inches='tight')
        # plt.close()