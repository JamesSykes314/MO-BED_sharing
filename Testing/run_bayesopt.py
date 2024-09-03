import time
import numpy as np
import pandas as pd
from experiment_9_10_dense import run_experiment

# make dictionary with keys and associated functions so then in line 18 can get the right function from dictionary based on key in arguments_list (Google if key is right word or maybe id??)

arguments_list = [[1,200,0],[1,100,100]]
n_tests = len(arguments_list)
iterations = 150
results = np.empty(shape=(n_tests, iterations+3))
objective_functions_name = "9_10_dense"

for j in range(n_tests):
    start_time = time.time()
    results[j, :3] = arguments_list[j]
    for i in range(iterations):
        _, fh = run_experiment(arguments_list[j][1], arguments_list[j][2])
        results[j,i+3] = fh
        print('Test number {} completed!'.format(i+1))
    end_time = time.time()
    print('Running the trials took {:.2f} minutes.'.format((end_time-start_time)/60))

try:
    results_df = pd.read_csv("results_df")
    results_df = pd.concat(results_df, pd.DataFrame(results))
# make sure pd.concat works when number of columns is different. If not find fix!!
except:
    results_df = pd.DataFrame(results)

results_df.to_csv("results_df")


