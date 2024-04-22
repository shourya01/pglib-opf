import os, pickle
import numpy as np
# from oct2py import Oct2Py
import cyipopt
import warnings
from tqdm import trange
from pypower.ext2int import ext2int
from pypower.api import loadcase
from problemDefJITM import opfSocp
from problemDefJITMargin import opfSocpMargin
from mpi4py import MPI

# # get octave object
# octave = Oct2Py()
# filter warnings
warnings.simplefilter("ignore", np.ComplexWarning)

# user options 
MAX_BUS = 10000 # upper limit of number of buses in cases to be considered
NUM_POINTS = 5000 # number of data points to save
DIFF = 0.2 # ratio variation of data

# main

if __name__ == "__main__":
    
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
    todo_cases =  ['pglib_opf_case118_ieee',"pglib_opf_case10000_goc"]
    
    with open(os.getcwd()+'/allcases.pkl','rb') as file:
        data = pickle.load(file)
        
    cases = data['cases']
    casenames = data['casenames']
    cases_new, casenames_new = [], []
    for cs,cn in zip(cases,casenames):
        if cn in todo_cases:
            cases_new.append(cs)
            casenames_new.append(cn)
    cases, casenames = cases_new, casenames_new
    # for cf in case_files:
    #     octave.source(current_directory+os.path.basename(cf))
    #     cname = os.path.basename(cf).split('.')[0]
    #     num_buses = None
    #     # determine number of buses in the case from its name
    #     for ci in cname.split('_'):
    #         if 'case' in ci:
    #             num_buses = int(''.join([chr for chr in ci.replace('case','',1) if chr.isdigit()]))
    #     # fitler out cases with more buses than MAX_BUS
    #     if num_buses <= MAX_BUS:
    #         # convert to internal indexing
    #         case_correct_idx = ext2int(loadcase(octave.feval(cname)))
    #         # append
    #         cases.append(case_correct_idx)
    #         casenames.append(cname)
            
    # generate upper and lower bound
            
    # solve
    for cn,this_case in zip(casenames,cases):
        
        # # generate upper bound of power demand margin
        # optObj = opfSocpMargin(this_case,cn,margin_sense=1)
        # cub, clb = optObj.calc_cons_bounds()
        # xub, xlb = optObj.calc_var_bounds()
        # # Define IPOPT problem
        # probMargin = cyipopt.Problem(
        #     n = optObj.in_size,
        #     m = optObj.cons_size,
        #     problem_obj=optObj,
        #     lb=xlb,
        #     ub=xub,
        #     cl=clb,
        #     cu=cub
        # )
        # # Setup solver options
        # probMargin.add_option('tol',1e-6)
        # probMargin.add_option('max_iter',2500)
        # probMargin.add_option('mumps_mem_percent',25000)
        # probMargin.add_option('mu_max',1e-0)
        # probMargin.add_option('mu_init',1e-0)
        # # probMargin.add_option('nlp_lower_bound_inf',-optObj.LARGE_NUMBER+1)
        # # probMargin.add_option('nlp_upper_bound_inf',optObj.LARGE_NUMBER-1)
        # # solve
        # x, info = probMargin.solve(optObj.calc_x0_flatstart())  
        # maxMarginPd = x[optObj.vidx['mPd']]
        # maxMarginQd = x[optObj.vidx['mQd']]
        
        # # generate lower bound of power demand margin
        # optObj = opfSocpMargin(this_case,cn,margin_sense=-1)
        # cub, clb = optObj.calc_cons_bounds()
        # xub, xlb = optObj.calc_var_bounds()
        # # Define IPOPT problem
        # probMargin = cyipopt.Problem(
        #     n = optObj.in_size,
        #     m = optObj.cons_size,
        #     problem_obj=optObj,
        #     lb=xlb,
        #     ub=xub,
        #     cl=clb,
        #     cu=cub
        # )
        # # Setup solver options
        # probMargin.add_option('tol',1e-6)
        # probMargin.add_option('max_iter',2500)
        # probMargin.add_option('mumps_mem_percent',25000)
        # probMargin.add_option('mu_max',1e-0)
        # probMargin.add_option('mu_init',1e-0)
        # # probMargin.add_option('nlp_lower_bound_inf',-optObj.LARGE_NUMBER+1)
        # # probMargin.add_option('nlp_upper_bound_inf',optObj.LARGE_NUMBER-1)
        # # solve
        # x, info = probMargin.solve(optObj.calc_x0_flatstart())  
        # minMarginPd = x[optObj.vidx['mPd']]
        # minMarginQd = x[optObj.vidx['mQd']]
        
        # # filter the min/max
        # # maxMarginPd = np.minimum(_minMarginPd,_maxMarginPd)
        # # minMarginPd = maxMarginPd
        # # maxMarginQd = np.minimum(_minMarginQd,_maxMarginQd)
        # # minMarginQd = maxMarginQd
        
        # solve base problem
        # print(f"\n--------\nSolving {cn}.\n--------\n",flush=True)
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
        prob.add_option('mu_max',1e-1) 
        prob.add_option('mu_init',1e-1)
        prob.add_option('max_iter',1000)
        
        # Solve ipopt problem
        _, info_base = prob.solve(optObj.calc_x0_flatstart())
        
        # data generation
        print(f"\n--------\nSolving {cn}.\n--------\n",flush=True)
        
        # Setup solver options
        prob.add_option('tol',1e-6)
        prob.add_option('mumps_mem_percent',25000)
        prob.add_option('mu_max',1e-1) 
        prob.add_option('mu_init',1e-1)
        prob.add_option('print_level',0) 
        prob.add_option('max_iter',1000)
        
        # generate points
        input, duals = [], []
        for pt_idx in (t:=trange(NUM_POINTS)):
            
            # set random seed
            np.random.seed(pt_idx)
            
            # get pd, qd and perturb
            pd,qd = optObj.get_loads()
            
            dpd,dqd = (1-DIFF + np.random.rand(*pd.shape)*DIFF)*pd, (1-DIFF + np.random.rand(*qd.shape)*DIFF)*qd
            optObj.change_loads(dpd,dqd)

            # solve problem
            _, info = prob.solve(info_base['x'],lagrange=info_base['mult_g'].tolist(),zl=info_base['mult_x_L'].tolist(),zu=info_base['mult_x_U'].tolist())
            if info['status'] == 0:
                input_data = {'pd':dpd,'qd':dqd,'flow_lim_f':np.zeros_like(optObj.flow_lim),'flow_lim_t':np.zeros_like(optObj.flow_lim),'angmin':np.zeros_like(optObj.angmin),'angmax':np.zeros_like(optObj.angmax)}
                input.append(np.concatenate([itm[1] for itm in input_data.items()],axis=0))
                dual = info['mult_g'][np.concatenate([optObj.cidx[consn] for consn in ['balance_real','balance_reac','flow_f','flow_t','angmin','angmax']])]
                duals.append(dual)
            
            # output status
            t.set_description(f"Status of point {pt_idx} is {info['status']}. Process ({mpi_rank}/{mpi_size}).")
            
        # save data
        with open(os.getcwd()+f'/data2/{cn}_data_rank_{mpi_rank}.pkl','wb') as file:
            pickle.dump(data,file)
        if len(input_data) > 0:
            np.savez_compressed(os.getcwd()+f'/data2/{cn}_inp_{mpi_rank}.npz',data=np.array(input))
            np.savez_compressed(os.getcwd()+f'/data2/{cn}_dual_{mpi_rank}.npz',data=np.array(duals))