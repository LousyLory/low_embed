import numpy as np
import sys
from matplotlib import pyplot as plt
import itertools
from utils import read_mat_file, read_file

def compute_min_eig(Z):
    min_eigs = np.linalg.eigvals(Z)
    return min_eigs

# read mat
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


sample_size = int(sys.argv[2])
runs = 50
n_bins = 50

# check for similar rows or columns
if dataset != "PSD":
    unique_rows, indices = np.unique(similarity_matrix, axis=0, return_index=True)
    similarity_matrix_O = similarity_matrix[indices][:, indices]
    # symmetrization
    similarity_matrix = (similarity_matrix_O + similarity_matrix_O.T) / 2.0

n,d = similarity_matrix.shape

SMS_eig_vals = []
for r in range(runs):
    sample_indices = np.random.choice(range(n), sample_size, replace=False)
    A = similarity_matrix[sample_indices][:, sample_indices]
    SMS_eig_vals.append(compute_min_eig(A))

merged_all_eig_vals = list(itertools.chain(*SMS_eig_vals))
print(len(merged_all_eig_vals))

range_low = -1
range_high = 1
clipped_eigvals = []
for i in range(len(merged_all_eig_vals)):
    if merged_all_eig_vals[i] > range_low and merged_all_eig_vals[i] < range_high:
        clipped_eigvals.append(merged_all_eig_vals[i])
# plot
if dataset == "twitter":
    title_name = "Twitter"
if dataset == "mrpc":
    title_name = "MRPC"
if dataset == "stsb":
    title_name = "STS-B"
path = "figures/eigenvalue_histogram/"+dataset+".pdf"
plt.gcf().clear()
scale_ = 0.55
new_size = (scale_ * 10, scale_ * 8.5)
plt.gcf().set_size_inches(new_size)

fig1 = plt.figure()
ax1 = fig1.add_subplot(111)

plt.rc('axes', titlesize=21)
plt.rc('axes', labelsize=25)
plt.rc('xtick', labelsize=18)
plt.rc('ytick', labelsize=18)
plt.rc('legend', fontsize=21)

# This is  the colormap I'd like to use.
cm = plt.cm.get_cmap('RdYlBu_r')

# Plot histogram.
n, bins, patches = ax1.hist(clipped_eigvals, n_bins, color='green')
bin_centers = 0.5 * (bins[:-1] + bins[1:])
# scale values to interval [0,1]
col = bin_centers - min(bin_centers)
col /= max(col)

for c, p in zip(col, patches):
    plt.setp(p, 'facecolor', cm(c))

# ax1.hist(merged_all_eig_vals, bins=n_bins)
plt.locator_params(axis='x', nbins=20)
# plt.xlim(left=-5, right=20)
plt.xlabel("Bin ranges", fontsize=18)
plt.ylabel("Number of eigenvalues in range", fontsize=18)
plt.title(title_name, fontsize=21)
plt.tight_layout(rect=[0, 0.03, 1, 0.95])
plt.savefig(path)
plt.gcf().clear()
# plt.show()