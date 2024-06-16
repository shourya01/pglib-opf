# assumes env with ipopt is already activated

mpiexec -np 10 python milestone4b.py --case pglib_opf_case2312_goc
mpiexec -np 10 python milestone4b.py --case pglib_opf_case4601_goc
mpiexec -np 10 python milestone4b.py --case pglib_opf_case10000_goc