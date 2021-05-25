import numpy as np
import matplotlib.pyplot as plt
import sys
# import seaborn as sns
from matplotlib.colors import ListedColormap
from tqdm import tqdm

from utils import read_file, is_pos_def, is_pos_semi_def, viz_eigenvalues, is_real_eig, read_mat_file

from copy import deepcopy
import scipy.misc as scm
from scipy.io import savemat
import random
import os

from plotter import plot_errors
from recursiveNystrom import wrapper_for_recNystrom



def CUR_alt(similarity_matrix, k, return_type="error"):
    """
    compute CUR approximation
    versions:
    U = S2^T K S1
    """
    list_of_available_indices = range(len(similarity_matrix))
    sample_indices_rows = np.sort(random.sample(\
                     list_of_available_indices, k))
    sample_indices_cols = np.sort(random.sample(\
                     list_of_available_indices, k))
    A = similarity_matrix[sample_indices_rows][:, sample_indices_cols]
    
    similarity_matrix_x = deepcopy(similarity_matrix)
    KS_cols = similarity_matrix_x[:, sample_indices_cols]
    KS_rows = similarity_matrix_x[sample_indices_rows]
    if return_type == "error":
        return np.linalg.norm(\
                similarity_matrix - \
                KS_cols @ np.linalg.pinv(A) @ KS_rows)\
                / np.linalg.norm(similarity_matrix)


step = 50
runs_ = 3
"""
20ng2_new_K_set1.mat  oshumed_K_set1.mat  recipe_K_set1.mat  recipe_trainData.mat  twitter_K_set1.mat  twitter_set1.mat
"""
filetype = None
dataset = sys.argv[1]
if dataset == "PSD":
    feats = np.random.random((1000,1000))
    similarity_matrix = feats @ feats.T
    filetype = "numpy"
if dataset == "mrpc" or dataset == "rte" or dataset == "stsb":
    filename = "../GYPSUM/"+dataset+"_predicts_0.npy"
    filetype = "python"
if dataset == "twitter":
    similarity_matrix = read_mat_file(file_="./WordMoversEmbeddings/mat_files/twitter_K_set1.mat")
if filetype == "python":
    similarity_matrix = read_file(filename)

# check for similar rows or columns
if dataset != "PSD":
    unique_rows, indices = np.unique(similarity_matrix, axis=0, return_index=True)
    similarity_matrix_O = similarity_matrix[indices][:, indices]
    # symmetrization
    similarity_matrix = (similarity_matrix_O + similarity_matrix_O.T) / 2.0
    # print("is the current matrix PSD? ", is_pos_def(similarity_matrix))
id_count = 1500#len(similarity_matrix)-1
print(dataset)

CUR_alt_error_list = []
print("alt nyst")
for k in tqdm(range(10, id_count, 10)):
    err = 0
    for j in range(runs_):
        error = CUR_alt(similarity_matrix, k)
        err += error
    error = err/np.float(runs_)
    CUR_alt_error_list.append(error)
    pass

plt.plot(CUR_alt_error_list)
plt.show()