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

filename = "mrpc"
# similarity_matrix = read_mat_file(file_="WordMoversEmbeddings/mat_files/twitter_K_set1.mat")
similarity_matrix = read_file("../GYPSUM/"+filename+"_predicts_0.npy")


# check for similar rows or columns
unique_rows, indices = np.unique(similarity_matrix, axis=0, return_index=True)
similarity_matrix_O = similarity_matrix[indices][:, indices]
# symmetrization
similarity_matrix = (similarity_matrix_O + similarity_matrix_O.T) / 2.0
# print("is the current matrix PSD? ", is_pos_def(similarity_matrix))

# true min eig
eps = 1e-16
min_eig = np.min(np.linalg.eigvals(similarity_matrix)) - eps

print(np.linalg.norm(min_eig*np.eye(len(similarity_matrix))) / \
	np.linalg.norm(similarity_matrix), min_eig)