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
mode = "normal"

"""
20ng2_new_K_set1.mat  oshumed_K_set1.mat  recipe_K_set1.mat  recipe_trainData.mat  twitter_K_set1.mat  twitter_set1.mat
"""
filename = "twitter"
# similarity_matrix = read_mat_file(file_="WordMoversEmbeddings/mat_files/twitter_K_set1.mat")
# similarity_matrix = read_file("../GYPSUM/"+filename+"_predicts_0.npy")
similarity_matrix = read_file("../GYPSUM/predicts_0.npy")
error_list = []
abs_error_list = []

# symmetrize
similarity_matrix = (similarity_matrix + similarity_matrix.T) / 2.0
# eigen correct
min_eig = np.real(np.min(np.linalg.eigvals(similarity_matrix)))
print(min_eig, similarity_matrix[0,5])
similarity_matrix = similarity_matrix - min_eig*np.eye(len(similarity_matrix))
print("matrix is PSD=", is_pos_semi_def(similarity_matrix))

error, abs_error, avg_min_eig, _ = simple_nystrom(similarity_matrix, similarity_matrix, 100, runs=1, mode='normal', normalize=norm_type, expand=expand_eigs)
