import numpy as np
import matplotlib.pyplot as plt
import sys
# import seaborn as sns
from matplotlib.colors import ListedColormap
from tqdm import tqdm

from utils import read_file, is_pos_def, is_pos_semi_def, viz_eigenvalues, is_real_eig, read_mat_file
# from Nystrom import simple_nystrom
from recursiveNystrom import wrapper_for_recNystrom as simple_nystrom
from copy import deepcopy
import scipy.misc as scm
from scipy.io import savemat

step = 50
norm_type = "original"
expand_eigs = True
print(expand_eigs)
mode = "eigI"
runs_ = 3

"""
20ng2_new_K_set1.mat  oshumed_K_set1.mat  recipe_K_set1.mat  recipe_trainData.mat  twitter_K_set1.mat  twitter_set1.mat
"""
filename = "stsb"
#similarity_matrix = read_file(pred_id_count=id_count, file_=filename+".npy")
# similarity_matrix = read_mat_file(file_="WordMoversEmbeddings/mat_files/recipe_trainData.mat", version="v7.3")
similarity_matrix = read_file("../GYPSUM/"+filename+"_predicts_0.npy")
#number_of_runs = id_count / step
error_list = []
abs_error_list = []

# check for similar rows or columns
unique_rows, indices = np.unique(similarity_matrix, axis=0, return_index=True)
similarity_matrix_O = similarity_matrix[indices][:, indices]
similarity_matrix = (similarity_matrix_O + similarity_matrix_O.T) / 2.0

if filename == "rte":
	similarity_matrix = similarity_matrix
	similarity_matrix_O = similarity_matrix_O

# print()

#similarity_matrix_O = deepcopy(similarity_matrix)
# similarity_matrix_O = similarity_matrix

id_count = 450 #len(similarity_matrix) #1000

avg_min_eig_vec = []
for k in tqdm(range(10, id_count, 10)):
    error, abs_error, avg_min_eig, _ = simple_nystrom(similarity_matrix, similarity_matrix_O, k, runs=runs_, mode='eigI', normalize=norm_type, expand=expand_eigs)
    error_list.append(error)
    abs_error_list.append(abs_error)
    avg_min_eig_vec.append(avg_min_eig)
    del _
    pass
# print(error_list)

# error_list = [x/np.linalg.norm(similarity_matrix) for x in error_list]
min_eig = np.min(np.linalg.eigvals(similarity_matrix))
min_eig = round(min_eig,4)
# display

# sns.set()
# flatui = ["#9b59b6", "#3498db", "#95a5a6", "#e74c3c", "#34495e", "#2ecc71"]
# cmap = ListedColormap(sns.color_palette(flatui).as_hex())

x_axis = list(range(10, id_count, 10))
# fig, ax = plt.subplots(figsize=(15, 8))
# for label in (ax.get_xticklabels() + ax.get_yticklabels()):
	# label.set_fontsize(16)
#"""

plt.rc('axes', titlesize=13)
plt.rc('axes', labelsize=13)
plt.rc('xtick', labelsize=13)
plt.rc('ytick', labelsize=13)
plt.rc('legend', fontsize=11)

STYLE_MAP = {"symmetrized error": {"color": "#4d9221",  "marker": ".", "markersize": 7, 'label': 'avg error wrt similarity matrix', 'linewidth': 1},
             "true error": {"color": "#7B3294",  "marker": ".", "markersize": 7, 'label': 'avg error wrt true matrix', 'linewidth': 1},
             "min eig": {"color": "#f1a340", "marker": ".", "markersize": 7, 'label': "True minimum eigenvalue = "+str(min_eig), 'linewidth': 1},
            }

plt.gcf().clear()
scale_ = 0.55
new_size = (scale_ * 10, scale_ * 8.5)
plt.gcf().set_size_inches(new_size)

sim_error_pairs = [(x, y) for x, y in zip(x_axis, error_list)]
true_error_pairs = [(x, y) for x, y in zip(x_axis, abs_error_list)]
arr1 = np.array(sim_error_pairs)
arr2 = np.array(true_error_pairs)
print(arr1.shape, arr2.shape)
plt.plot(arr1[:, 0], arr1[:, 1], **STYLE_MAP['symmetrized error'])
plt.plot(arr2[:, 0], arr2[:, 1], **STYLE_MAP['true error'])
plt.locator_params(axis='x', nbins=6)
plt.xlabel("Number of landmark samples")
plt.ylabel("Approximation error")
plt.title("Plot of average errors using Nystrom on "+filename+" BERT", fontsize=13)
plt.tight_layout()
plt.legend(loc='upper right')
# plt.savefig("./test1.pdf")
plt.savefig("figures/final_"filename+"_approx_errors.pdf")

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
plt.title("Plot of minimum eigenvalue estimate", fontsize=13)
# plt.savefig("./test2.pdf")
plt.savefig("figures/final_"filename+"_min_eigenvalue_estimate.pdf")
plt.close()

# plt.plot(x, error_list, label="average errors")
# plt.plot(x, abs_error_list, label="average errors wrt original similarity_matrix")
# plt.xlabel("number of reduced samples", fontsize=20)
# plt.ylabel("error score", fontsize=20)
# plt.legend(loc="upper right", fontsize=20)
# plt.title("plot of average errors using Nystrom on "+filename+" BERT", fontsize=20)
# if mode == "eigI":
#     plt.savefig("test_old1.pdf")
# else:
#     plt.savefig("test_old1.pdf")
# plt.clf()
# # """
# plt.plot(x, avg_min_eig_vec, label="average min eigen values")
# plt.xlabel("number of reduced samples")
# plt.ylabel("minimum eigenvalues")
# plt.legend(loc="upper right")
# plt.title("plot of average eigenvalues for original values: "+str(min_eig)) 
# plt.savefig("test_old2.pdf")
# # """
