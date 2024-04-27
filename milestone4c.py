import numpy as np
import glob as glob

cases = ['pglib_opf_case118_ieee','pglib_opf_case2312_goc','pglib_opf_case4601_goc','pglib_opf_case10000_goc']
read_dir = '/home/exx/shourya/presolve/pglib-opf/data2'
target_dir = '/home/exx/shourya/presolve/pglib-opf/data'

if __name__ == "__main__":
    
    for case in cases:
        
        inp_matrices = []
        for file in glob.glob(read_dir+'/'+case+'_inp_*.npz'):
            inp_matrices.append(np.load(file)['data'])
        inp_matrix = np.vstack(inp_matrices)
        np.savez_compressed(target_dir+'/'+case+'_inp.npz',data=inp_matrix)
        
        dual_matrices = []
        for file in glob.glob(read_dir+'/'+case+'_dual_*.npz'):
            dual_matrices.append(np.load(file)['data'])
        dual_matrix = np.vstack(dual_matrices)
        np.savez_compressed(target_dir+'/'+case+'_dual.npz',data=dual_matrix)
