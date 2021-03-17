import numpy as np
import matplotlib.pyplot as plt
import sys
import seaborn as sns
from matplotlib.colors import ListedColormap
from tqdm import tqdm

from utils import read_file, is_pos_def, is_pos_semi_def, viz_eigenvalues, is_real_eig, read_mat_file
from Nystrom import simple_nystrom
from copy import deepcopy
import scipy.misc as scm
from scipy.io import savemat

step = 10
norm_type = "original"
expand_eigs = True
print(expand_eigs)
mode = "eigI"
runs_ = 5

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

# print()

#similarity_matrix_O = deepcopy(similarity_matrix)
# similarity_matrix_O = similarity_matrix

id_count = 1000 #len(similarity_matrix) #1000

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
# display

sns.set()
flatui = ["#9b59b6", "#3498db", "#95a5a6", "#e74c3c", "#34495e", "#2ecc71"]
cmap = ListedColormap(sns.color_palette(flatui).as_hex())

x = list(range(10, id_count, 10))
fig, ax = plt.subplots(figsize=(15, 8))
for label in (ax.get_xticklabels() + ax.get_yticklabels()):
	label.set_fontsize(16)
#"""
plt.plot(x, error_list, label="average errors")
plt.plot(x, abs_error_list, label="average errors wrt original similarity_matrix")
plt.xlabel("number of reduced samples", fontsize=20)
plt.ylabel("error score", fontsize=20)
plt.legend(loc="upper right", fontsize=20)
plt.title("plot of average errors using Nystrom on "+filename+" BERT", fontsize=20)
if mode == "eigI":
    plt.savefig("figures/nystrom_errors_new_"+mode+"_"+norm_type+"_"+str(int(expand_eigs))+"_"+filename+".pdf")
else:
    plt.savefig("figures/nystrom_errors_new_"+mode+"_"+norm_type+"_"+filename+".pdf")
plt.clf()
# """
plt.plot(x, avg_min_eig_vec, label="average min eigen values")
plt.xlabel("number of reduced samples")
plt.ylabel("minimum eigenvalues")
plt.legend(loc="upper right")
plt.title("plot of average eigenvalues for original values: "+str(min_eig)) 
plt.savefig("figures/"+filename+"_min_eigenvalue_estimate.pdf")
# """
