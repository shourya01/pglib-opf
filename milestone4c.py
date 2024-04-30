import numpy as np
import glob as glob
from tqdm import tqdm

cases = ['pglib_opf_case118_ieee','pglib_opf_case793_goc','pglib_opf_case1354_pegase','pglib_opf_case2312_goc','pglib_opf_case4601_goc','pglib_opf_case10000_goc']
read_dir = '/home/exx/shourya/presolve/pglib-opf/data2'
target_dir = '/home/exx/shourya/presolve/pglib-opf/data'

if __name__ == "__main__":
    
    for case in cases:
        
        inp_matrices = []
        for file in tqdm(glob.glob(read_dir+'/'+case+'_inp_*.npz')):
            inp_data = np.load(file)['data']
            if inp_data.size == 0:
                print(f"Input data for case {case} and file {file} has 0 elements! Continuing.")
            else:
                inp_matrices.append(inp_data)
        inp_matrix = np.vstack(inp_matrices)
        print(f"Case {case} has {inp_matrix.shape[0]} examples.")
        np.savez_compressed(target_dir+'/'+case+'_inp.npz',data=inp_matrix)
        
        dual_matrices = []
        for file in tqdm(glob.glob(read_dir+'/'+case+'_dual_*.npz')):
            dual_data = np.load(file)['data']
            if dual_data.size == 0:
                print(f"Dual data for case {case} and file {file} has 0 elements! Continuing.")
            else:
                dual_matrices.append(dual_data)
        dual_matrix = np.vstack(dual_matrices)
        np.savez_compressed(target_dir+'/'+case+'_dual.npz',data=dual_matrix)
