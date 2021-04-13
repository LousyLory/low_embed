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


step = 50
norm_type = "original"
expand_eigs = True
# print(expand_eigs)
mode = "eigI"
runs_ = 3

"""
20ng2_new_K_set1.mat  oshumed_K_set1.mat  recipe_K_set1.mat  recipe_trainData.mat  twitter_K_set1.mat  twitter_set1.mat
"""
filename = "mrpc"
id_count = 500 #len(similarity_matrix) #1000
# similarity_matrix = read_mat_file(file_="WordMoversEmbeddings/mat_files/twitter_K_set1.mat")
similarity_matrix = read_file("../GYPSUM/"+filename+"_predicts_0.npy")
# similarity_matrix = read_file("/mnt/nfs/work1/elm/ray/old_predicts_0.npy")

uni_norm_error_list = []
uni_eig_error_list = []
lev_error_list = []
uni_CUR_error_list = []

# check for similar rows or columns
unique_rows, indices = np.unique(similarity_matrix, axis=0, return_index=True)
similarity_matrix_O = similarity_matrix[indices][:, indices]
# symmetrization
similarity_matrix = (similarity_matrix_O + similarity_matrix_O.T) / 2.0
# eigenshift
# min_eig = np.min(np.linalg.eigvals(similarity_matrix))
# similarity_matrix = similarity_matrix - min_eig*np.eye(len(similarity_matrix))
# check if current matrix is PSD
print("is the current matrix PSD? ", is_pos_def(similarity_matrix))


# if filename == "rte":
# 	similarity_matrix = 1-similarity_matrix
# 	similarity_matrix_O = 1-similarity_matrix_O

################# uniform sampling #####################################
from Nystrom import simple_nystrom

for k in tqdm(range(10, id_count, 10)):
    error, _, _, _ = simple_nystrom(similarity_matrix, similarity_matrix_O, \
        k, runs=runs_, mode='eigI', normalize=norm_type, expand=expand_eigs)
    uni_eig_error_list.append(error)
    del _
    pass



#################### leverage sampling ################################
"""
from recursiveNystrom import wrapper_for_recNystrom

for k in tqdm(range(10, id_count, 10)):
    error, _, _, _ = wrapper_for_recNystrom(similarity_matrix, similarity_matrix_O, k, runs=runs_, mode='eigI', normalize=norm_type, expand=expand_eigs)
    lev_error_list.append(error)
    del _
    pass
"""

#################### CUR decomposition no eig ###############################
#"""
from CUR import simple_CUR
for k in tqdm(range(10, id_count, 10)):
    error, _, _, _ = simple_CUR(similarity_matrix, similarity_matrix_O, \
        k, runs=runs_, mode='eigI', normalize=norm_type, expand=expand_eigs)
    uni_CUR_error_list.append(error)
    del _
    pass
#"""

########################## anchornet #########################################
# anchor_error_list = []
# from anchor import simple_anchor

# for k in tqdm(range(10, id_count, 10)):
#     error, _, _, _ = simple_anchor(xy, similarity_matrix, similarity_matrix_O, \
#         k, runs=runs_, mode='eigI', normalize=norm_type, expand=expand_eigs)
#     anchor_error_list.append(error)
#     del _
#     pass


#######################################################################
# PLOTS
x_axis = list(range(10, id_count, 10))

plt.rc('axes', titlesize=13)
plt.rc('axes', labelsize=13)
plt.rc('xtick', labelsize=13)
plt.rc('ytick', labelsize=13)
plt.rc('legend', fontsize=11)

STYLE_MAP = {"uniform normal error": {"color": "#4d9221",  "marker": ".", "markersize": 7, 'label': 'Uniform normal', 'linewidth': 1},
             "uniform eig error": {"color": "#EF4026",  "marker": ".", "markersize": 7, 'label': 'Uniform eig', 'linewidth': 1},
             "leverage error": {"color": "#7B3294",  "marker": ".", "markersize": 7, 'label': 'Leverage', 'linewidth': 1},
             "uniform CUR error": {"color": "#7B3294",  "marker": ".", "markersize": 7, 'label': 'Uniform CUR', 'linewidth': 1},
             "anchor net": {"color": "#9ACD32",  "marker": ".", "markersize": 7, 'label': 'Anchor Net', 'linewidth': 1},
            }

plt.gcf().clear()
scale_ = 0.55
new_size = (scale_ * 10, scale_ * 8.5)
plt.gcf().set_size_inches(new_size)

title_name = "MRPC"

uniform_eig_error_pairs = [(x, y) for x, y in zip(x_axis, uni_eig_error_list)]
#uniform_norm_error_pairs = [(x, y) for x, y in zip(x_axis, uni_norm_error_list)]
#leverage_error_pairs = [(x, y) for x, y in zip(x_axis, lev_error_list)]
uniform_CUR_error_pairs = [(x, y) for x, y in zip(x_axis, uni_CUR_error_list)]
# anchor_error_pairs = [(x, y) for x, y in zip(x_axis, anchor_error_list)]
arr1 = np.array(uniform_eig_error_pairs)
#arr2 = np.array(leverage_error_pairs)
#arr3 = np.array(uniform_norm_error_pairs)
arr4 = np.array(uniform_CUR_error_pairs)
# arr5 = np.array(anchor_error_pairs)
plt.plot(arr1[:, 0], arr1[:, 1], **STYLE_MAP['uniform eig error'])
#plt.plot(arr2[:, 0], arr2[:, 1], **STYLE_MAP['leverage error'])
#plt.plot(arr3[:, 0], arr3[:, 1], **STYLE_MAP['uniform normal error'])
plt.plot(arr4[:, 0], arr4[:, 1], **STYLE_MAP['uniform CUR error'])
# plt.plot(arr5[:, 0], arr5[:, 1], **STYLE_MAP['anchor net'])
plt.locator_params(axis='x', nbins=6)
# plt.ylim(bottom=0.0, top=100)
plt.xlabel("Number of landmark samples")
plt.ylabel("Average approximation error")
plt.title(title_name, fontsize=13)
plt.tight_layout(rect=[0, 0.03, 1, 0.95])
plt.legend(loc='upper right')
plt.savefig("figures/unyst_v_cur_v_anchor_errors_"+filename+".pdf")
# plt.savefig("./test1.pdf")
plt.gcf().clear()

