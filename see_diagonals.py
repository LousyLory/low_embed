import numpy as np
import matplotlib.pyplot as plt
import sys
import seaborn as sns
from matplotlib.colors import ListedColormap
from tqdm import tqdm

from utils import read_file, is_pos_def, is_pos_semi_def, viz_eigenvalues, is_real_eig, norm_diag, viz_diagonal
from Nystrom import simple_nystrom
from copy import deepcopy
import scipy.misc as scm
from scipy.io import savemat

id_count = 1118
similarity_matrix = read_file(pred_id_count=id_count, file_="predicts_1.npy")

# check for similar rows or columns
unique_rows, indices = np.unique(similarity_matrix, axis=0, return_index=True)
similarity_matrix = similarity_matrix[indices][:, indices]
sym_similarity_matrix = similarity_matrix + similarity_matrix.T
diag = np.diagonal(sym_similarity_matrix)

viz_diagonal(sym_similarity_matrix, mat_type="symmetrized_matrix_1")

# normalize diagonal
K = norm_diag(sym_similarity_matrix)
viz_diagonal(K, mat_type="normalized_symmetrized_matrix_1")

