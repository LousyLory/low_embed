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

from recursiveNystrom import wrapper_for_recNystrom

def nystrom(similarity_matrix, k, min_eig=0.0, min_eig_mode=False, return_type="error", correct_outer=False):
    """
    compute nystrom approximation
    versions:
    1. True nystrom with min_eig_mode=False
    2. Eigen corrected nystrom with min_eig_mode=True
    2a. KS can be eigencorrected with correct_outer=True
    2b. KS not eigencorrected with correct_outer=False
    """
    list_of_available_indices = range(len(similarity_matrix))
    sample_indices = np.sort(random.sample(\
                     list_of_available_indices, k))
    A = similarity_matrix[sample_indices][:, sample_indices]
    if min_eig_mode == True:
        A = A - min_eig*np.eye(len(A))
        if correct_outer == False:
            similarity_matrix_x = deepcopy(similarity_matrix)
        else:
            similarity_matrix_x = deepcopy(similarity_matrix)\
                                  - min_eig*np.eye(len(similarity_matrix))
    else:
        similarity_matrix_x = deepcopy(similarity_matrix)
    KS = similarity_matrix_x[:, sample_indices]
    if return_type == "error":
        return np.linalg.norm(\
                similarity_matrix - \
                KS @ np.linalg.pinv(A) @ KS.T)\
                / np.linalg.norm(similarity_matrix)


def ratio_nystrom(similarity_matrix, k, min_eig=0.0, min_eig_mode=False, return_type="error"):
    """
    compute nystrom approximation
    versions:
    1. True nystrom with min_eig_mode=False
    2. Eigen corrected nystrom with min_eig_mode=True
    2a. KS can be eigencorrected with correct_outer=True
    2b. KS not eigencorrected with correct_outer=False
    """
    eps=1e-16
    list_of_available_indices = range(len(similarity_matrix))
    sample_indices = np.sort(random.sample(\
                     list_of_available_indices, k))
    A = similarity_matrix[sample_indices][:, sample_indices]
    if min_eig_mode == True:
        A = A - min_eig*np.eye(len(A))
        similarity_matrix_x = deepcopy(similarity_matrix)
    elif min_eig_mode == False:
        similarity_matrix_x = deepcopy(similarity_matrix)
    else:
        local_min_eig = min(0, np.min(np.linalg.eigvals(A))) - eps
        ratio = min_eig / local_min_eig
        A = (1.0/ratio)*A - np.eye(len(A))
        similarity_matrix_x = deepcopy(similarity_matrix)

    KS = similarity_matrix_x[:, sample_indices]
    if return_type == "error":
        if min_eig_mode == True or min_eig_mode == False:
            return np.linalg.norm(\
                similarity_matrix - \
                KS @ np.linalg.pinv(A) @ KS.T)\
                / np.linalg.norm(similarity_matrix)
        else:
            return np.linalg.norm(\
                similarity_matrix - \
                (1/ratio)*KS @ np.linalg.pinv(A) @ KS.T)\
                / np.linalg.norm(similarity_matrix)


def nystrom_with_eig_estimate(similarity_matrix, k, return_type="error", scaling=False, mult=0):
    """
    compute eigen corrected nystrom approximations
    versions:
    1. Eigen corrected without scaling (scaling=False)
    2. Eigen corrected with scaling (scaling=True)
    """
    eps=1e-16
    list_of_available_indices = range(len(similarity_matrix))
    sample_indices = np.sort(random.sample(\
                     list_of_available_indices, k))
    A = similarity_matrix[sample_indices][:, sample_indices]
    # estimating min eig in the following block
    if mult == 0:
        large_k = np.int(np.sqrt(k*len(similarity_matrix)))
    else:
        large_k = min(mult*k, len(similarity_matrix)-1)
    larger_sample_indices = np.sort(random.sample(\
                            list_of_available_indices, large_k))
    Z = similarity_matrix[larger_sample_indices][:, larger_sample_indices]
    min_eig = min(0, np.min(np.linalg.eigvals(Z))) - eps
    if scaling == True:
        min_eig = min_eig*np.float(len(similarity_matrix))/np.float(large_k)

    A = A - min_eig*np.eye(len(A))
    similarity_matrix_x = deepcopy(similarity_matrix)
    KS = similarity_matrix_x[:, sample_indices]
    
    if return_type == "error":
        return np.linalg.norm(\
                similarity_matrix - \
                KS @ np.linalg.pinv(A) @ KS.T)\
                / np.linalg.norm(similarity_matrix), min_eig


def CUR(similarity_matrix, k, eps=1e-3, delta=1e-14, return_type="error", same=False):
    """
    implementation of Linear time CUR algorithm of Drineas2006 et. al.

    input:
    1. similarity matrix in R^{n,d}
    2. integers c, r, and k

    output:
    1. either C, U, R matrices
    or
    1. CU^+R
    or
    1. error = similarity matrix - CU^+R

    """
    n,d = similarity_matrix.shape
    # setting up c, r, eps, and delta for error bound
    # c = (64*k*((1+8*np.log(1/delta))**2) / (eps**4)) + 1
    # r = (4*k / ((delta*eps)**2)) + 1
    # c = 4*k
    c = k
    r = k
    if c > n:
        c= n
    # r = 4*k
    if r > n:
        r = n
    # print("chosen c, r:", c,r)
    try:
        assert 1 <= c and c <= d
    except AssertionError as error:
        print("1 <= c <= m is not true")
    try:
        assert 1 <= r and r <= n
    except AssertionError as error:
        print("1 <= r <= n is not true")
    try:
        assert 1 <= k and k <= min(c,r)
    except AssertionError as error:
        print("1 <= k <= min(c,r)")

    # using uniform probability instead of row norms
    pj = np.ones(d).astype(float) / float(d)
    qi = np.ones(n).astype(float) / float(n)

    # choose samples
    samples_c = np.random.choice(range(d), c, replace=False, p = pj)
    if same:
        samples_r = samples_c
    else:
        samples_r = np.random.choice(range(n), r, replace=False, p = qi)

    # grab rows and columns and scale with respective probability
    samp_pj = pj[samples_c]
    samp_qi = qi[samples_r]
    C = similarity_matrix[:, samples_c] / np.sqrt(samp_pj*c)
    rank_k_C = C
    # modification works only because we assume similarity matrix is symmetric
    R = similarity_matrix[:, samples_r] / np.sqrt(samp_qi*r)
    R = R.T
    psi = C[samples_r, :].T / np.sqrt(samp_qi*r)
    psi = psi.T

    U = np.linalg.inv(rank_k_C.T @ rank_k_C)
    # i chose not to compute rank k reduction of U
    U = U @ psi.T
    
    if return_type == "decomposed":
        return C, U, R
    if return_type == "approximate":
        return (C @ U) @ R
    if return_type == "error":
        # print(np.linalg.norm((C @ U) @ R))
        relative_error = np.linalg.norm(similarity_matrix - ((C @ U) @ R)) / np.linalg.norm(similarity_matrix)
        return relative_error

def Lev_samples(similarity_matrix, k, KS_correction=False):
    K = similarity_matrix
    num_imp_samples = k
    error, _,_,_ = wrapper_for_recNystrom(similarity_matrix, K, num_imp_samples, \
        runs=1, mode="normal", normalize="rows", \
        expand=True, KS_correction=KS_correction)
    return error

step = 50
runs_ = 3
"""
20ng2_new_K_set1.mat  oshumed_K_set1.mat  recipe_K_set1.mat  recipe_trainData.mat  twitter_K_set1.mat  twitter_set1.mat
"""
filename = "mrpc"
id_count = 450 #len(similarity_matrix) #1000
# similarity_matrix = read_mat_file(file_="WordMoversEmbeddings/mat_files/twitter_K_set1.mat")
similarity_matrix = read_file("../GYPSUM/"+filename+"_predicts_0.npy")

KS_corrected_error_list = []
KS_ncorrected_error_list = []

scaling_error_list = []
nscaling_error_list = []

min_eig_scaling = []
min_eig_nscaling = []

SKS_corrected_error_list = []
SKS_ncorrected_error_list = []
SKS_rcorrected_error_list = []
ZKZ_multiplier_error_list = []

CUR_diff_error_list = []
CUR_same_error_list = []

Lev_corrected_error_list = []
Lev_ncorrected_error_list = []

# check for similar rows or columns
unique_rows, indices = np.unique(similarity_matrix, axis=0, return_index=True)
similarity_matrix_O = similarity_matrix[indices][:, indices]
# symmetrization
similarity_matrix = (similarity_matrix_O + similarity_matrix_O.T) / 2.0
# print("is the current matrix PSD? ", is_pos_def(similarity_matrix))

################# uniform sampling #####################################
# eps=1e-16
# min_eig = min(0, np.min(np.linalg.eigvals(similarity_matrix))) - eps
# for k in tqdm(range(10, id_count, 10)):
#     err = 0
#     for j in range(runs_):
#         error = nystrom(similarity_matrix, k, min_eig_mode=True, min_eig=min_eig)
#         err += error
#     error = err/np.float(runs_)
#     KS_ncorrected_error_list.append(error)
#     pass

#     err = 0
#     for j in range(runs_):
#         error = nystrom(similarity_matrix, k, min_eig_mode=True, min_eig=min_eig, correct_outer=True)
#         err += error
#     error = err/np.float(runs_)
#     KS_corrected_error_list.append(error)
#     pass    

######################## eigen corrected uniform sampling ###################
# eps=1e-16
# min_eig_val = min(0, np.min(np.linalg.eigvals(similarity_matrix))) - eps
# for k in tqdm(range(10, id_count, 10)):
#     err = 0
#     for j in range(runs_):
#         error = nystrom(similarity_matrix, k, min_eig_mode=True, min_eig=min_eig_val)
#         err += error
#     error = err/np.float(runs_)
#     KS_ncorrected_error_list.append(error)
#     pass


# for k in tqdm(range(10, id_count, 10)):
#     err = 0
#     min_eig_agg = 0
#     for j in range(runs_):
#         error, min_eig = nystrom_with_eig_estimate(similarity_matrix, k, return_type="error", scaling=True)
#         err += error
#         min_eig_agg += min_eig
#     error = err/np.float(runs_)
#     min_eig_scaling.append(min_eig_agg/np.float(runs_))
#     scaling_error_list.append(error)
#     pass

for k in tqdm(range(10, id_count, 10)):
    err = 0
    min_eig_agg = 0
    for j in range(runs_):
        error, min_eig = nystrom_with_eig_estimate(similarity_matrix, k, return_type="error")
        err += error
        min_eig_agg += min_eig
    error = err/np.float(runs_)
    # min_eig_nscaling.append(min_eig_agg/np.float(runs_))
    nscaling_error_list.append(error)
    pass    

for k in tqdm(range(10, id_count, 10)):
    err = 0
    min_eig_agg = 0
    for j in range(runs_):
        error, min_eig = nystrom_with_eig_estimate(similarity_matrix, k, return_type="error", mult=10)
        err += error
        min_eig_agg += min_eig
    error = err/np.float(runs_)
    # min_eig_nscaling.append(min_eig_agg/np.float(runs_))
    ZKZ_multiplier_error_list.append(error)
    pass   

################################## RATIO CHECK ################################
# eps=1e-16
# min_eig = min(0, np.min(np.linalg.eigvals(similarity_matrix))) - eps
# for k in tqdm(range(10, id_count, 10)):
    # err = 0
    # for j in range(runs_):
    #     error = ratio_nystrom(similarity_matrix, k, min_eig_mode=True, min_eig=min_eig)
    #     err += error
    # error = err/np.float(runs_)
    # SKS_corrected_error_list.append(error)
    # pass

    # err = 0
    # for j in range(runs_):
    #     error = ratio_nystrom(similarity_matrix, k, min_eig_mode=False, min_eig=min_eig)
    #     err += error
    # error = err/np.float(runs_)
    # SKS_ncorrected_error_list.append(error)
    # pass    

    # err = 0
    # for j in range(runs_):
    #     error = ratio_nystrom(similarity_matrix, k, min_eig_mode="ratio", min_eig=min_eig)
    #     err += error
    # error = err/np.float(runs_)
    # SKS_rcorrected_error_list.append(error)
    # pass 

################################ CUR decomposition ##########################
# for k in tqdm(range(10, id_count, 10)):
#     err = 0
#     for j in range(runs_):
#         error = CUR(similarity_matrix, k)
#         err += error
#     error = err/np.float(runs_)
#     CUR_diff_error_list.append(error)
#     pass

# for k in tqdm(range(10, id_count, 10)):
#     err = 0
#     for j in range(runs_):
#         error = CUR(similarity_matrix, k, same=True)
#         err += error
#     error = err/np.float(runs_)
#     CUR_same_error_list.append(error)
#     pass


################################ leverage scores #############################
# for k in tqdm(range(10, id_count, 10)):
#     err = 0
#     for j in range(runs_):
#         error = Lev_samples(similarity_matrix, k)
#         err += error
#     error = err/np.float(runs_)
#     Lev_ncorrected_error_list.append(error)
#     # Lev_error_list.append(error)
#     pass

# for k in tqdm(range(10, id_count, 10)):
#     err = 0
#     for j in range(runs_):
#         error = Lev_samples(similarity_matrix, k, KS_correction=True)
#         err += error
#     error = err/np.float(runs_)
#     Lev_corrected_error_list.append(error)
#     # Lev_error_list.append(error)
#     pass

#######################################################################
# PLOTS
x_axis = list(range(10, id_count, 10))

plt.rc('axes', titlesize=13)
plt.rc('axes', labelsize=13)
plt.rc('xtick', labelsize=13)
plt.rc('ytick', labelsize=13)
plt.rc('legend', fontsize=11)

STYLE_MAP = {"With true eigen correction": {"color": "#4d9221",  "marker": ".", "markersize": 7, 'label': 'True eig', 'linewidth': 1},
             "With estimated eigen corrected": {"color": "#EF4026",  "marker": ".", "markersize": 7, 'label': 'Estimated eig', 'linewidth': 1},
             "With scaled estimated eigen correction": {"color": "#7B3294",  "marker": ".", "markersize": 7, 'label': 'Scaled estimated eig', 'linewidth': 1},
             "Estimated eigenvalue": {"color": "#7B3294",  "marker": ".", "markersize": 7, 'label': 'Estimated Eig', 'linewidth': 1},
             "Scaled estimated eigenvalue": {"color": "#9ACD32",  "marker": ".", "markersize": 7, 'label': 'Scaled estimated eig', 'linewidth': 1},
             "True eigenvalue": {"color": "#EE82DE",  'label': 'True eig', 'linewidth': 1},
             "KS corrected": {"color": "#4d9221",  "marker": ".", "markersize": 7, 'label': 'KS corrected', 'linewidth': 1},
             "KS not corrected": {"color": "#EF4026",  "marker": ".", "markersize": 7, 'label': 'KS not corrected', 'linewidth': 1},
             "SKS corrected": {"color": "#EF4026",  "marker": ".", "markersize": 7, 'label': 'SKS corrected', 'linewidth': 1},
             "SKS not corrected": {"color": "#4d9221",  "marker": ".", "markersize": 7, 'label': 'SKS not corrected', 'linewidth': 1},
             "SKS ratio corrected": {"color": "#EE82DE",  "marker": ".", "markersize": 7, 'label': 'SKS ratio corrected', 'linewidth': 1},
             "CUR diff": {"color": "#EE82DE",  "marker": ".", "markersize": 7, 'label': 'CUR', 'linewidth': 1},
             "CUR same": {"color": "#4d9221",  "marker": ".", "markersize": 7, 'label': 'CUR same c and r', 'linewidth': 1},
             "RLS KS not corrected": {"color": "#9ACD32",  "marker": ".", "markersize": 7, 'label': 'RLS KS not corrected', 'linewidth': 1},
             "RLS KS corrected": {"color": "#4d9221",  "marker": ".", "markersize": 7, 'label': 'RLS KS corrected', 'linewidth': 1},
             "multiplied Z": {"color": "#4d9221",  "marker": ".", "markersize": 7, 'label': 'z = 10s', 'linewidth': 1},
            }

plt.gcf().clear()
scale_ = 0.55
new_size = (scale_ * 10, scale_ * 8.5)
plt.gcf().set_size_inches(new_size)

title_name = "MRPC"
directory = "figures/comparison_with_sample_multiplier/"
if not os.path.isdir(directory):
    os.mkdir(directory)
path = os.path.join(directory, filename+".pdf")

# KS_ncorrected_error_pairs = [(x, y) for x, y in zip(x_axis, KS_ncorrected_error_list)]
# arr1 = np.array(KS_ncorrected_error_pairs)
# plt.plot(arr1[:, 0], arr1[:, 1], **STYLE_MAP["With true eigen correction"])
# plt.plot(arr1[:, 0], arr1[:, 1], **STYLE_MAP["KS not corrected"])

# KS_corrected_error_pairs = [(x, y) for x, y in zip(x_axis, KS_corrected_error_list)]
# arr1 = np.array(KS_corrected_error_pairs)
# plt.plot(arr1[:, 0], arr1[:, 1], **STYLE_MAP["KS corrected"])

# scaling_error_pairs = [(x, y) for x, y in zip(x_axis, scaling_error_list)]
# arr2 = np.array(scaling_error_pairs)
# plt.plot(arr2[:, 0], arr2[:, 1], **STYLE_MAP["With scaled estimated eigen correction"])

nscaling_error_pairs = [(x, y) for x, y in zip(x_axis, nscaling_error_list)]
arr1 = np.array(nscaling_error_pairs)
plt.plot(arr1[:, 0], arr1[:, 1], **STYLE_MAP["With estimated eigen corrected"])

ZKZ_multiplier_error_pairs = [(x, y) for x, y in zip(x_axis, ZKZ_multiplier_error_list)]
arr1 = np.array(ZKZ_multiplier_error_pairs)
plt.plot(arr1[:, 0], arr1[:, 1], **STYLE_MAP["multiplied Z"])

# SKS_corrected_error_pairs = [(x, y) for x, y in zip(x_axis, SKS_corrected_error_list)]
# arr1 = np.array(SKS_corrected_error_pairs)
# plt.plot(arr1[:, 0], arr1[:, 1], **STYLE_MAP["SKS corrected"])

# SKS_ncorrected_error_pairs = [(x, y) for x, y in zip(x_axis, SKS_ncorrected_error_list)]
# arr1 = np.array(SKS_ncorrected_error_pairs)
# plt.plot(arr1[:, 0], arr1[:, 1], **STYLE_MAP["SKS not corrected"])

# SKS_rcorrected_error_pairs = [(x, y) for x, y in zip(x_axis, SKS_rcorrected_error_list)]
# arr1 = np.array(SKS_rcorrected_error_pairs)
# plt.plot(arr1[:, 0], arr1[:, 1], **STYLE_MAP["SKS ratio corrected"])

# CUR_diff_error_pairs = [(x, y) for x, y in zip(x_axis, CUR_diff_error_list)]
# arr1 = np.array(CUR_diff_error_pairs)
# plt.plot(arr1[:, 0], arr1[:, 1], **STYLE_MAP["CUR diff"])

# CUR_same_error_pairs = [(x, y) for x, y in zip(x_axis, CUR_same_error_list)]
# arr1 = np.array(CUR_same_error_pairs)
# plt.plot(arr1[:, 0], arr1[:, 1], **STYLE_MAP["CUR same"])

# lev_ncorrected_error_pairs = [(x, y) for x, y in zip(x_axis, Lev_ncorrected_error_list)]
# arr1 = np.array(lev_ncorrected_error_pairs)
# plt.plot(arr1[:, 0], arr1[:, 1], **STYLE_MAP["RLS KS not corrected"])

# lev_corrected_error_pairs = [(x, y) for x, y in zip(x_axis, Lev_corrected_error_list)]
# arr1 = np.array(lev_corrected_error_pairs)
# plt.plot(arr1[:, 0], arr1[:, 1], **STYLE_MAP["RLS KS corrected"])

plt.locator_params(axis='x', nbins=6)
# plt.ylim(bottom=0.0, top=0.3)
plt.xlabel("Number of landmark samples")
plt.ylabel("Average approximation error")
plt.title(title_name, fontsize=13)
plt.tight_layout(rect=[0, 0.03, 1, 0.95])
plt.legend(loc='upper right')
plt.savefig(path)
# plt.show()
plt.gcf().clear()


################################### EIGENVALUE PLOTS ###############################################################
# path = os.path.join(directory, filename+"_eigenvalue.pdf")

# min_eig_vec = min_eig_val*np.ones(len(x_axis))
# min_eig_pairs = [(x, y) for x, y in zip(x_axis, list(min_eig_vec))]
# arr1 = np.array(min_eig_pairs)
# plt.plot(arr1[:, 0], arr1[:, 1], **STYLE_MAP["True eigenvalue"])

# min_eig_scaling_pairs = [(x, y) for x, y in zip(x_axis, min_eig_scaling)]
# arr1 = np.array(min_eig_scaling_pairs)
# plt.plot(arr1[:, 0], arr1[:, 1], **STYLE_MAP["Scaled estimated eigenvalue"])

# min_eig_nscaling_pairs = [(x, y) for x, y in zip(x_axis, min_eig_nscaling)]
# arr1 = np.array(min_eig_nscaling_pairs)
# plt.plot(arr1[:, 0], arr1[:, 1], **STYLE_MAP["Estimated eigenvalue"])

# plt.locator_params(axis='x', nbins=6)
# # plt.ylim(bottom=0.0, top=2)
# plt.xlabel("Number of landmark samples")
# plt.ylabel("Average estimated eigenvalues")
# plt.title(title_name, fontsize=13)
# plt.tight_layout(rect=[0, 0.03, 1, 0.95])
# plt.legend(loc='upper right')
# plt.savefig(path)
# # plt.show()
# plt.gcf().clear()
