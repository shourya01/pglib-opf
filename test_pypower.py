from pypower.api import loadcase
import numpy as np
import os

# get all cases in current directory
current_directory = os.getcwd()
all_files_and_directories = os.listdir(current_directory)
case_files = [os.path.abspath(f) for f in all_files_and_directories if f.endswith('.m') and os.path.isfile(os.path.join(current_directory, f))]
cases = []
for cf in case_files:
    cases.append(loadcase(cf))
    print(f"Locaded case: {cases[-1]}")