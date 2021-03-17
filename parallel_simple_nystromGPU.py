#===================================================================#
# this works very well for the cmd in windows but not for WSL Ubuntu
#===================================================================#

import numpy as np
import matplotlib.pyplot as plt
import sys
import seaborn as sns
from matplotlib.colors import ListedColormap
from tqdm import tqdm

from utils import is_pos_def, is_pos_semi_def, viz_eigenvalues, is_real_eig
# from Nystrom import simple_nystrom
# from recursiveNystrom import wrapper_for_recNystrom as simple_nystrom
from copy import deepcopy
import scipy.misc as scm
from scipy.io import savemat

import multiprocessing
from joblib import Parallel, delayed

import timeit

start_time = timeit.timeit()

num_cores = multiprocessing.cpu_count()

step = 10
norm_type = "original"
expand_eigs = True
mode = "eigI"
runs_ = 10
code_mode = "GPU"
from utils import read_file, read_mat_file

"""
20ng2_new_K_set1.mat  oshumed_K_set1.mat  recipe_K_set1.mat  recipe_trainData.mat  twitter_K_set1.mat  twitter_set1.mat
"""
# approximation_type = "leverage_"
approximation_type = "leverage_"

if approximation_type == "leverage_":
    from recursiveNystromGPU import wrapper_for_recNystrom as simple_nystrom
    pass
if approximation_type == "uniform_":
    from Nystrom import simple_nystrom
    pass

filename = "mrpc"
#similarity_matrix = read_file(pred_id_count=id_count, file_=filename+".npy")
print("Reading file ...")
# similarity_matrix = read_mat_file(file_="WordMoversEmbeddings/mat_files/20ng2_new_K_set1.mat", version="v7.3")
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
    similarity_matrix = 1-similarity_matrix

if code_mode == "GPU":
    import torch
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    similarity_matrix = torch.from_numpy(similarity_matrix).to(device)
print("Preprocessing done.")
# print()

similarity_matrix_O = deepcopy(similarity_matrix)
# similarity_matrix_O = similarity_matrix

# """
def top_level_function(id_count):
    error, abs_error, avg_min_eig, rank_l_k = simple_nystrom(\
                similarity_matrix, similarity_matrix_O, id_count, runs=runs_, \
                mode='eigI', normalize=norm_type, expand=expand_eigs)
    # processed_list = [error, abs_error, avg_min_eig, rank_l_k]
    return error, abs_error, avg_min_eig

id_count = 400 #len(similarity_matrix) #1000
inputs = tqdm(range(10, id_count, 10))


# for k in tqdm(range(10, id_count, 10)):
#     error, abs_error, avg_min_eig, _ = simple_nystrom(similarity_matrix, similarity_matrix_O, k, runs=runs_, mode='eigI', normalize=norm_type, expand=expand_eigs)
#     error_list.append(error)
#     abs_error_list.append(abs_error)
#     avg_min_eig_vec.append(avg_min_eig)
#     del _
#     pass

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
# min_eig = np.real(np.min(np.linalg.eigvals(similarity_matrix)))
min_eig = torch.lobpcg(similarity_matrix.expand(1, -1, -1), k=1, largest=False, method="ortho")[0].cpu().numpy()

avg_min_eig_vec = [x.cpu().numpy() for x in avg_min_eig_vec]

# display
sns.set()
flatui = ["#9b59b6", "#3498db", "#95a5a6", "#e74c3c", "#34495e", "#2ecc71"]
cmap = ListedColormap(sns.color_palette(flatui).as_hex())

x = list(range(10, id_count, 10))
fig, ax = plt.subplots(figsize=(15, 8))
for label in (ax.get_xticklabels() + ax.get_yticklabels()):
  label.set_fontsize(16)

plt.plot(x, error_list, label="average errors")
plt.plot(x, abs_error_list, label="average errors wrt original similarity_matrix")
plt.xlabel("number of reduced samples", fontsize=20)
plt.ylabel("error score", fontsize=20)
plt.legend(loc="upper right", fontsize=20)
plt.title("plot of average errors using Nystrom on "+filename+" BERT", fontsize=20)
if mode == "eigI":
    plt.savefig("figures/"+approximation_type+"nystrom_errors_new_"+mode+"_"+norm_type+"_"+str(int(expand_eigs))+"_"+filename+".pdf")
else:
    plt.savefig("figures/"+approximation_type+"nystrom_errors_new_"+mode+"_"+norm_type+"_"+filename+".pdf")
plt.clf()

plt.plot(x, np.squeeze(avg_min_eig_vec), label="average min eigenvalues")
plt.xlabel("number of reduced samples", fontsize=20)
plt.ylabel("minimum eigenvalues", fontsize=20)
plt.legend(loc="upper right", fontsize=20)
plt.title("plot of average eigenvalues for original values: "+str(min_eig), fontsize=20) 
plt.savefig("figures/"+approximation_type+filename+"_min_eigenvalue_estimate.pdf")
#"""

end_time = timeit.timeit()
print("total time for execution:", end_time-start_time)
# total time for execution: 4.214799992041662e-05