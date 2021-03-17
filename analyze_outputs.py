import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from matplotlib.colors import ListedColormap
import matplotlib as mpl
import sys
import scipy

from utils import read_file, read_mat_file
import numpy as np

# from sympy import Matrix

def check_symmetric(a, tol=1e-8):
    return np.all(np.abs(a-a.T) < tol)

def is_pos_def(x, tol=1e-8):
    return np.all(np.linalg.eigvals(x) > -tol)

dataset = "mrpc"

reshaped_preds = read_file(file_="../GYPSUM/"+dataset+"_predicts_0.npy")
reshaped_preds = reshaped_preds.astype('float64') 
print(reshaped_preds.shape, reshaped_preds.dtype)
# if dataset == "rte":
	# reshaped_preds = 1- reshaped_preds

# reshaped_preds = Matrix(reshaped_preds)

"""
20ng2_new_K_set1.mat  oshumed_K_set1.mat  recipe_K_set1.mat  recipe_trainData.mat  twitter_K_set1.mat  twitter_set1.mat
"""
# reshaped_preds = read_mat_file(file_="WordMoversEmbeddings/mat_files/20ng2_new_K_set1.mat", version="v7.3")

# X = np.linalg.norm(reshaped_preds - reshaped_preds.T)
# print("error for transpose subtraction (check for similarity)", X)

# unique_rows, indices = np.unique(reshaped_preds, axis=0, return_index=True)
# reshaped_preds = reshaped_preds[indices][:, indices]

reshaped_preds = (reshaped_preds + reshaped_preds.T) / 2

print("shape of output matrix:", reshaped_preds.shape)
print("symmetric check", check_symmetric(reshaped_preds))

# info for the dataset
#print("are the matrices positive semidefinite: ", is_pos_def(reshaped_preds))
#print("rank of the matrix: ", np.linalg.matrix_rank(reshaped_preds))

# eigenvals = np.linalg.eigvals(reshaped_preds)
# eigenvals, eigvecs = scipy.sparse.linalg.eigs(reshaped_preds, k=len(reshaped_preds)-2)
# eigenvals = reshaped_preds.eigenvals()
s = 1
k = 200
eigenvals, eigvecs = scipy.sparse.linalg.eigs(reshaped_preds, k=k)
print(eigenvals.shape)

abs_eigvals = np.absolute(eigenvals)
real_eigvals = np.real(eigenvals)

abs_eigvals = abs_eigvals[s:k]
real_eigvals = real_eigvals[s:k]

# create the plot
sns.set()
flatui = ["#9b59b6", "#3498db", "#95a5a6", "#e74c3c", "#34495e", "#2ecc71"]
cmap = ListedColormap(sns.color_palette(flatui).as_hex())
# x = list(range(1,len(abs_eigvals)+1))
x = list(range(0,len(abs_eigvals)))
# fig = plt.gcf()
fig, ax = plt.subplots(figsize=(15, 8))
# fig.set_size_inches(15, 8, forward=True)
for label in (ax.get_xticklabels() + ax.get_yticklabels()):
	label.set_fontsize(16)
plt.scatter(x, abs_eigvals, label="absolute values", c='#e74c3c')
plt.scatter(x, real_eigvals, label="real part", c='#34495e')
plt.figtext(.7, .7, "PSD? = "+str(is_pos_def(reshaped_preds)), fontsize=20)
plt.figtext(.7, .65, "Rank = "+str(np.linalg.matrix_rank(reshaped_preds)), fontsize=20)
plt.xlabel("eigenvalue indices", fontsize=20)
plt.ylabel("values", fontsize=20)
plt.legend(loc='upper right', fontsize=20)
plt.title("plot of the distribution of norm and real part of the eigenvalues", fontsize=20)
plt.savefig("figures/sympy_"+dataset+"_eigenvalues_122_full.pdf")
#"""
