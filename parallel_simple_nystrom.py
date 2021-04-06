#===================================================================#
# this works very well for the cmd in windows but not for WSL Ubuntu
#===================================================================#

import numpy as np
import matplotlib.pyplot as plt
import sys
# import seaborn as sns
from matplotlib.colors import ListedColormap
from tqdm import tqdm

from utils import is_pos_def, is_pos_semi_def, viz_eigenvalues, is_real_eig, read_file, read_mat_file
# from Nystrom import simple_nystrom
# from recursiveNystrom import wrapper_for_recNystrom as simple_nystrom
from copy import deepcopy
import scipy.misc as scm
from scipy.io import savemat

import multiprocessing
from joblib import Parallel, delayed

import time


start_time = time.time()

num_cores = multiprocessing.cpu_count()

step = 10
norm_type = "original"
expand_eigs = True
mode = "eigI"
runs_ = 5

"""
20ng2_new_K_set1.mat  oshumed_K_set1.mat  recipe_K_set1.mat  recipe_trainData.mat  twitter_K_set1.mat  twitter_set1.mat
"""
# approximation_type = "leverage_"
approximation_type = "uniform_"

# method = "WMD"

if approximation_type == "leverage_":
    from recursiveNystrom import wrapper_for_recNystrom as simple_nystrom
    pass
if approximation_type == "uniform_":
    from Nystrom import simple_nystrom
    pass

filename = "rte"
id_count = 450 #len(similarity_matrix) #1000
#similarity_matrix = read_file(pred_id_count=id_count, file_=filename+".npy")
print("Reading file ...")
# similarity_matrix = read_mat_file(file_="WordMoversEmbeddings/mat_files/oshumed_K_set1.mat", version="v7.3")
similarity_matrix = read_file("../GYPSUM/"+filename+"_predicts_0.npy")

print("File read. Beginning preprocessing ...")
#number_of_runs = id_count / step
error_list = []
abs_error_list = []
avg_min_eig_vec = []

# check for similar rows or columns
unique_rows, indices = np.unique(similarity_matrix, axis=0, return_index=True)
similarity_matrix_O = similarity_matrix[indices][:, indices]
similarity_matrix = (similarity_matrix_O + similarity_matrix_O.T) / 2.0
if filename == "rte":
    # pass
    similarity_matrix = 1-similarity_matrix
print("Preprocessing done.")
# print()

#similarity_matrix_O = deepcopy(similarity_matrix)
# similarity_matrix_O = similarity_matrix
def top_level_function(id_count):
    error, abs_error, avg_min_eig, rank_l_k = simple_nystrom\
                (similarity_matrix, similarity_matrix_O, id_count, runs=runs_, \
                mode='eigI', normalize=norm_type, expand=expand_eigs)
    # processed_list = [error, abs_error, avg_min_eig, rank_l_k]
    return error, abs_error, avg_min_eig

inputs = tqdm(range(10, id_count, 10))

print("Beginning approximation parallely ...")
e = Parallel(n_jobs=num_cores, backend="threading")(map(delayed(top_level_function), inputs))
print("Approximation done. Beginning write out to files.")

for i in range(len(e)):
    tuple_out = e[i]
    error_list.append(tuple_out[0])
    abs_error_list.append(tuple_out[1])
    avg_min_eig_vec.append(tuple_out[2])    

# print(len(error_list), len(abs_error_list), len(avg_min_eig_vec))
error_list = np.array(error_list)
abs_error_list = np.array(abs_error_list)
avg_min_eig_vec = np.array(avg_min_eig_vec)
min_eig = np.real(np.min(np.linalg.eigvals(similarity_matrix)))
min_eig = round(min_eig,4)

# display
x_axis = list(range(10, id_count, 10))

plt.rc('axes', titlesize=13)
plt.rc('axes', labelsize=13)
plt.rc('xtick', labelsize=13)
plt.rc('ytick', labelsize=13)
plt.rc('legend', fontsize=11)

STYLE_MAP = {"symmetrized error": {"color": "#4d9221",  "marker": ".", "markersize": 7, 'label': 'Symmetrized', 'linewidth': 1},
             "true error": {"color": "#7B3294",  "marker": ".", "markersize": 7, 'label': 'Original', 'linewidth': 1},
             "min eig": {"color": "#f1a340", "marker": ".", "markersize": 7, 'label': "True minimum eigenvalue = "+str(min_eig), 'linewidth': 1},
            }

plt.gcf().clear()
scale_ = 0.55
new_size = (scale_ * 10, scale_ * 8.5)
plt.gcf().set_size_inches(new_size)

title_name = "RTE"

sim_error_pairs = [(x, y) for x, y in zip(x_axis, error_list)]
true_error_pairs = [(x, y) for x, y in zip(x_axis, abs_error_list)]
arr1 = np.array(sim_error_pairs)
arr2 = np.array(true_error_pairs)
print(arr1.shape, arr2.shape)
plt.plot(arr1[:, 0], arr1[:, 1], **STYLE_MAP['symmetrized error'])
plt.plot(arr2[:, 0], arr2[:, 1], **STYLE_MAP['true error'])
plt.locator_params(axis='x', nbins=6)
plt.xlabel("Number of landmark samples")
plt.ylabel("Average approximation error")
plt.title(title_name, fontsize=13)
plt.tight_layout(rect=[0, 0.03, 1, 0.95])
plt.legend(loc='upper right')
plt.savefig("figures/final_"+approximation_type+"_nystrom_errors_"+filename+".pdf")
# plt.savefig("./test1.pdf")
plt.gcf().clear()

scale_ = 0.55
new_size = (scale_ * 10, scale_ * 8.5)
plt.gcf().set_size_inches(new_size)
eigval_estimate_pairs = [(x, y) for x, y in zip(x_axis, list(np.squeeze(avg_min_eig_vec)))]
arr1 = np.array(eigval_estimate_pairs)
plt.plot(arr1[:, 0], arr1[:, 1], **STYLE_MAP['min eig'])
plt.locator_params(axis='x', nbins=6)
plt.xlabel("Number of landmark samples")
plt.ylabel("Minimum eigenvalue estimate")
plt.tight_layout(rect=[0, 0.03, 1, 0.95])
plt.legend(loc='upper right')
plt.title(title_name, fontsize=13)
# plt.savefig("./test2.pdf")
plt.savefig("figures/final_"+approximation_type+filename+"_min_eigenvalue_estimate.pdf")
plt.close()

end_time = time.time()
print("total time for execution:", end_time-start_time)
# total time for execution: 0.0001415879996784497