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

# get octave object
octave = Oct2Py()
# filter warnings
warnings.simplefilter("ignore", np.ComplexWarning)

# user options 
MAX_BUS = 10000 # upper limit of number of buses in cases to be considered

if __name__ == "__main__":
    
    # get all cases in current directory
    current_directory = os.getcwd()+'/'
    # current_directory = '/home/sbose/pglib-opf/' # for running on BEBOP
    all_files_and_directories = os.listdir(current_directory)
    # case_files = [os.path.join(current_directory, f) for f in all_files_and_directories if f.endswith('.m') and os.path.isfile(os.path.join(current_directory, f))]
    case_files = case_files = [current_directory+i for i in ['pglib_opf_case118_ieee.m','pglib_opf_case2312_goc.m',"pglib_opf_case4601_goc.m","pglib_opf_case10000_goc.m"]] # case specific

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
        
    data = {'casenames':casenames,'cases':cases}
            
    with open(os.getcwd()+f'/allcases.pkl','wb') as file:
            pickle.dump(data,file)
            
    print("Done!")

