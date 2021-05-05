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

def nystrom_with_eig_estimate(similarity_matrix, k, return_type="error", mult=0, \
    z_scale=0, min_eig_val=0):
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
    if min_eig_val == 0:
        if z_scale == 0:
            large_k = np.int(np.sqrt(k*len(similarity_matrix)))
        else:
            large_k = min(z_scale*k, len(similarity_matrix))
        larger_sample_indices = np.sort(random.sample(\
                                list_of_available_indices, large_k))
        Z = similarity_matrix[larger_sample_indices][:, larger_sample_indices]
        min_eig = min(0, np.min(np.linalg.eigvals(Z))) - eps
    else:
        min_eig = min_eig_val
    # use the multiplier to multiply to the true eigenvalue estimate
    if min_eig == 0:
        min_eig = mult*min_eig
    
    A = A - min_eig*np.eye(len(A))
    similarity_matrix_x = deepcopy(similarity_matrix)
    KS = similarity_matrix_x[:, sample_indices]
    
    if return_type == "error":
        return np.linalg.norm(\
                similarity_matrix - \
                KS @ np.linalg.pinv(A) @ KS.T)\
                / np.linalg.norm(similarity_matrix), min_eig

##########################################################################################
step = 50
runs_ = 3
"""
20ng2_new_K_set1.mat  oshumed_K_set1.mat  recipe_K_set1.mat  recipe_trainData.mat  twitter_K_set1.mat  twitter_set1.mat
"""
filename = "stsb"
id_count = 750 #len(similarity_matrix) #1000
# similarity_matrix = read_mat_file(file_="WordMoversEmbeddings/mat_files/twitter_K_set1.mat")
similarity_matrix = read_file("../GYPSUM/"+filename+"_predicts_0.npy")
# check for similar rows or columns
unique_rows, indices = np.unique(similarity_matrix, axis=0, return_index=True)
similarity_matrix_O = similarity_matrix[indices][:, indices]
# symmetrization
similarity_matrix = (similarity_matrix_O + similarity_matrix_O.T) / 2.0

multipliers = list(np.arange(1.0, 2.3, 0.5))

list_of_list_of_errors = []
list_of_min_eig_scaling = []

z_range = [1,2,5,10]

# eps=1e-16
# min_eig_val = np.real(min(0, np.min(np.linalg.eigvals(similarity_matrix)))) - eps
# min_eigs = np.arange(0.001, min_eig_val, min_eig_val/10)

for j in range(len(z_range)):
    z_val = z_range[j]
    for i in tqdm(range(len(multipliers))):
        multiplier = multipliers[i]
        list_of_errors = []
        min_eig_scaling = []
        for k in range(10, id_count, 10):
            err = 0
            min_eig_agg = 0
            for j in range(runs_):
                error, min_eig = nystrom_with_eig_estimate\
                    (similarity_matrix, k, return_type="error", mult=multiplier,\
                        z_scale=z_val)
                err += error
                min_eig_agg += min_eig
            error = err/np.float(runs_)
            min_eig_scaling.append(min_eig_agg/np.float(runs_))
            list_of_errors.append(error)
            pass
        list_of_list_of_errors.append(list_of_errors)
        list_of_min_eig_scaling.append(min_eig_scaling)
        pass

# for i in tqdm(range(len(min_eigs))):
#     list_of_errors = []
#     min_eig_scaling = []
#     for k in range(10, id_count, 10):
#         err = 0
#         min_eig_agg = 0
#         for j in range(runs_):
#             error, min_eig = nystrom_with_eig_estimate\
#                 (similarity_matrix, k, return_type="error", min_eig_val=min_eigs[i])
#             err += error
#             min_eig_agg += min_eig
#         error = err/np.float(runs_)
#         min_eig_scaling.append(min_eig_agg/np.float(runs_))
#         list_of_errors.append(error)
#         pass
#     list_of_list_of_errors.append(list_of_errors)
#     list_of_min_eig_scaling.append(min_eig_scaling)
#     pass


###############################PLOT##########################################SSS
x_axis = list(range(10, id_count, 10))

plt.gcf().clear()
scale_ = 0.55
new_size = (scale_ * 10, scale_ * 8.5)
plt.gcf().set_size_inches(new_size)

fig1 = plt.figure()
ax1 = fig1.add_subplot(111)

plt.rc('axes', titlesize=13)
plt.rc('axes', labelsize=13)
plt.rc('xtick', labelsize=13)
plt.rc('ytick', labelsize=13)
plt.rc('legend', fontsize=11)

STYLE_MAP = {"plot":{"marker":".", "markersize":7, "linewidth":1}}

for j in range(len(z_range)):
    for i in range(len(multipliers)):
        id_ = j*len(multipliers)+i
        # error_pairs = [(x, y) for x, y in zip(x_axis, list_of_list_of_errors[i])]
        error_pairs = list_of_list_of_errors[id_]
        arr1 = np.array(error_pairs)
        ax1.plot(np.array(x_axis),arr1,\
            label=str(round(multipliers[i],2))+", *z="+str(z_range[j]),**STYLE_MAP["plot"])


# for i in range(len(min_eigs)):
#     # error_pairs = [(x, y) for x, y in zip(x_axis, list_of_list_of_errors[i])]
#     error_pairs = list_of_list_of_errors[i]
#     arr1 = np.array(error_pairs)
#     ax1.plot(np.array(x_axis),arr1,\
#         label=min_eigs[i], **STYLE_MAP["plot"])

colormap = plt.cm.cool
colors1 = [colormap(i) for i in np.linspace(0, 1,int(len(ax1.lines)/len(z_range)))]
colormap = plt.cm.copper
colors2 = [colormap(i) for i in np.linspace(0, 1,int(len(ax1.lines)/len(z_range)))]
colormap = plt.cm.autumn
colors3 = [colormap(i) for i in np.linspace(0, 1,int(len(ax1.lines)/len(z_range)))]
colormap = plt.cm.winter
colors4 = [colormap(i) for i in np.linspace(0, 1,int(len(ax1.lines)/len(z_range)))]

for i,j in enumerate(ax1.lines):
    if int(i / 3) == 0:
        j.set_color(colors1[i%3])
    if int(i / 3) == 1:
        j.set_color(colors2[i%3])
    if int(i / 3) == 2:
        j.set_color(colors3[i%3])
    if int(i / 3) == 3:
        j.set_color(colors4[i%3])

title_name = "STS-B"

directory = "figures/comparison_among_eigenvalues_and_z/"
if not os.path.isdir(directory):
    os.mkdir(directory)
path = os.path.join(directory, filename+".pdf")


plt.locator_params(axis='x', nbins=6)
plt.ylim(bottom=0.0, top=1.35)
plt.xlabel("Number of landmark samples")
plt.ylabel("Average approximation error")
plt.title(title_name, fontsize=13)
plt.tight_layout(rect=[0, 0.03, 1, 0.95])
ax1.legend(loc='upper right')
plt.savefig(path)
# plt.show()
plt.gcf().clear()


################################# EIGENVALUE PLOTS ###############################################################
path = os.path.join(directory, filename+"_eigenvalue.pdf")

scale_ = 0.55
new_size = (scale_ * 10, scale_ * 8.5)
plt.gcf().set_size_inches(new_size)

fig1 = plt.figure()
ax1 = fig1.add_subplot(111)

plt.rc('axes', titlesize=13)
plt.rc('axes', labelsize=13)
plt.rc('xtick', labelsize=13)
plt.rc('ytick', labelsize=13)
plt.rc('legend', fontsize=11)

STYLE_MAP = {"plot":{"marker":".", "markersize":7, "linewidth":1}}

eps=1e-16
min_eig_val = np.real(min(0, np.min(np.linalg.eigvals(similarity_matrix)))) - eps
min_eig_vec = min_eig_val*np.ones(len(x_axis))
arr1 = np.array(min_eig_vec)

ax1.plot(np.array(x_axis),arr1,label="True",**STYLE_MAP["plot"])
for j in range(len(z_range)):
    for i in range(len(multipliers)):
        id_ = j*len(multipliers)+i
        # error_pairs = [(x, y) for x, y in zip(x_axis, list_of_list_of_errors[i])]
        eigenvalues = list_of_min_eig_scaling[id_]
        arr1 = np.array(eigenvalues)
        ax1.plot(np.array(x_axis),arr1,\
            label=str(round(multipliers[i],2))+", *z="+str(z_range[j]),**STYLE_MAP["plot"])

# colormap = plt.cm.copper
# colors = [colormap(i) for i in np.linspace(0, 1,len(ax1.lines))]

# for i,j in enumerate(ax1.lines):
#     j.set_color(colors[i])

colormap = plt.cm.cool
colors1 = [colormap(i) for i in np.linspace(0, 1,int(len(ax1.lines)/len(z_range)))]
colormap = plt.cm.copper
colors2 = [colormap(i) for i in np.linspace(0, 1,int(len(ax1.lines)/len(z_range)))]
colormap = plt.cm.autumn
colors3 = [colormap(i) for i in np.linspace(0, 1,int(len(ax1.lines)/len(z_range)))]
colormap = plt.cm.winter
colors4 = [colormap(i) for i in np.linspace(0, 1,int(len(ax1.lines)/len(z_range)))]

for i,j in enumerate(ax1.lines):
    if int(i / 3) == 0:
        j.set_color(colors1[i%3])
    if int(i / 3) == 1:
        j.set_color(colors2[i%3])
    if int(i / 3) == 2:
        j.set_color(colors3[i%3])
    if int(i / 3) == 3:
        j.set_color(colors4[i%3])

plt.locator_params(axis='x', nbins=6)
# plt.ylim(bottom=0.0, top=2)
plt.xlabel("Number of landmark samples")
plt.ylabel("Average estimated eigenvalues")
plt.title(title_name, fontsize=13)
plt.tight_layout(rect=[0, 0.03, 1, 0.95])
ax1.legend(loc='upper right')
plt.savefig(path)
# plt.show()
plt.gcf().clear()


######################################### Z Values ######################################
# path = os.path.join(directory, filename+"_zvalues.pdf")

# scale_ = 0.55
# new_size = (scale_ * 10, scale_ * 8.5)
# plt.gcf().set_size_inches(new_size)

# fig1 = plt.figure()
# ax1 = fig1.add_subplot(111)

# plt.rc('axes', titlesize=13)
# plt.rc('axes', labelsize=13)
# plt.rc('xtick', labelsize=13)
# plt.rc('ytick', labelsize=13)
# plt.rc('legend', fontsize=11)

# STYLE_MAP = {"plot":{"marker":".", "markersize":7, "linewidth":1}}
 
# z_vals = np.sqrt(np.array(x_axis)*len(similarity_matrix)).astype(int)
# arr1 = np.array(z_vals)
# ax1.plot(np.array(x_axis),arr1,label="z values",**STYLE_MAP["plot"])

# plt.locator_params(axis='x', nbins=6)
# # plt.ylim(bottom=0.0, top=2)
# plt.xlabel("Number of landmark samples")
# plt.ylabel("Average estimated eigenvalues")
# plt.title(title_name, fontsize=13)
# plt.tight_layout(rect=[0, 0.03, 1, 0.95])
# ax1.legend(loc='upper right')
# plt.savefig(path)
# # plt.show()
# plt.gcf().clear()
