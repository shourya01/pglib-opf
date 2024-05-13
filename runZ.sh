conda activate PRESOLVE

mpiexec -np 10 python milestone8.py --case pglib_opf_case118_ieee
mpiexec -np 10 python milestone8.py --case pglib_opf_case793_goc
mpiexec -np 10 python milestone8.py --case pglib_opf_case1354_pegase
mpiexec -np 10 python milestone8.py --case pglib_opf_case2869_pegase
mpiexec -np 10 python milestone8.py --case pglib_opf_case4601_goc
mpiexec -np 10 python milestone8.py --case pglib_opf_case10000_goc