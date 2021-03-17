import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from matplotlib.colors import ListedColormap
import matplotlib as mpl
import sys
import scipy

from utils import read_file, read_mat_file
import numpy as np

def dist_mat(A, B):
	dist = np.linalg.norm(A-B)
	return dist

reshaped_preds = read_file(file_="../GYPSUM/rte_predicts_0.npy")
reshaped_preds1 = (reshaped_preds + reshaped_preds.T) / 2

print(dist_mat(reshaped_preds, reshaped_preds.T), dist_mat(reshaped_preds1, reshaped_preds1.T))

plt.matshow(reshaped_preds)
plt.colorbar()
plt.savefig("figures/rte_full.pdf")
plt.clf()


plt.matshow(reshaped_preds1)
plt.colorbar()
plt.savefig("figures/sym_rte_full.pdf")
plt.clf()
