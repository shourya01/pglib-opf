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
from tqdm import tqdm, trange

octave = Oct2Py()
MAX_BUS = 10000
NUM_FILES_FOR_DATA = 23

dir_name = 'data/'

if __name__ == "__main__":

    # get all cases in current directory
    current_directory = os.getcwd()+'/'
    # current_directory = '/home/sbose/pglib-opf/' # for running on BEBOP
    all_files_and_directories = os.listdir(current_directory)
    # three specific cases
    case_files = [current_directory+i for i in ['pglib_opf_case2312_goc.m',"pglib_opf_case4601_goc.m","pglib_opf_case10000_goc.m"]]

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
        
        # generate upper bound of power demand margin
        optObj = opfSocpMargin(this_case,cn,margin_sense=1)
        cub, clb = optObj.calc_cons_bounds()
        xub, xlb = optObj.calc_var_bounds()
        # Define IPOPT problem
        probMargin = cyipopt.Problem(
            n = optObj.in_size,
            m = optObj.cons_size,
            problem_obj=optObj,
            lb=xlb,
            ub=xub,
            cl=clb,
            cu=cub
        )
        input, output_duals, output_cost = [], [], []
        # load pickle
        for nfiles in tqdm(range(NUM_FILES_FOR_DATA)):
            with open(os.getcwd()+'/'+dir_name+cn+f'_data_rank_{nfiles}.pkl','rb') as file:
                data = pickle.load(file)
            for dat in data:
                # save input
                input.append(np.concatenate([v for _,v in dat[0].items()]))
                # convert duals to negative and save
                dual = dat[1]['mult_g'][np.concatenate([optObj.cidx[consn] for consn in ['balance_real','balance_reac','flow_f','flow_t','angmin','angmax']])]
                # dual = np.where(dual<0,dual,-dual)
                output_duals.append(dual)
                # append cost
                output_cost.append(dat[1]['obj_val'])
        
        input = np.array(input)
        output_duals = np.array(output_duals)
        output_cost = np.array(output_cost)
        
        np.savez(os.getcwd()+'/'+dir_name+f'{cn}_inp.npz',data=input)
        np.savez(os.getcwd()+'/'+dir_name+f'{cn}_dual.npz',data=output_duals)
        np.savez(os.getcwd()+'/'+dir_name+f'{cn}_costs.npz',data=output_cost)
                

                
        
            
            
        
