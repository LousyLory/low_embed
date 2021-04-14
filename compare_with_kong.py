# this is a dataset file for the donkey kong image
import numpy as np
import matplotlib.pyplot as plt
import skimage.io
from skimage import feature
# from utils import is_pos_semi_def as is_psd
from measures import sigmoid, tps
# import numpy as np
# import matplotlib.pyplot as plt
import sys
# import seaborn as sns
from matplotlib.colors import ListedColormap
from tqdm import tqdm

from utils import read_file, is_pos_def, is_pos_semi_def, viz_eigenvalues, is_real_eig, read_mat_file

from copy import deepcopy
import scipy.misc as scm
from scipy.io import savemat

# get the donkey kong file
# !wget 'https://www.coloringpages4u.com/page/mario/donkeykong.tn768.png' 
# saved as donkeykong.tn768.png

imagedrawing = skimage.io.imread('donkeykong.tn768.png')
# plt.imshow(imagedrawing, cmap='gray')

edges = imagedrawing

xy = np.stack(np.where(edges == 0), axis=1)
n_samples = 3000
xy_sampled_idxs = np.random.randint(low=0, high=xy.shape[0], size=n_samples)
xy = xy[xy_sampled_idxs, :]
xy[:,0] = -xy[:,0]
y_min = np.min(xy[:,0])
xy[:,0] = xy[:,0]-y_min
xy = xy.astype(np.float)
# add noise
# noise_matrix = np.random.random(xy.shape)
# xy = xy+noise_matrix
#normalize between 0 and 1
# print(xy.dtype, xy[0,:], np.float(np.max(xy[:,0])), np.float(np.max(xy[:,1])))
xy[:, 0] = xy[:,0] / np.max(xy[:,0])
xy[:, 1] = xy[:,1] / np.max(xy[:,1])
# print(xy[0,:])
# print(xy)
# print(np.max(xy[:,0]), np.max(xy[:,1]))
# plt.scatter(xy[:,1], xy[:,0])
# plt.show()


##############################################################################
######################### create similarity matrix ###########################
# define the measure first
similarity_matrix = sigmoid(xy, xy)
# np.save("../GYPSUM/kong_sigmoid_similarity.npy", similarity_matrix)
similarity_matrix_O = similarity_matrix
# print("rank:",np.linalg.matrix_rank(similarity_matrix))
# print(is_psd(similarity_matrix))
# plt.imshow(similarity_matrix)
# plt.show()
# print(np.min(similarity_matrix), np.max(similarity_matrix))

##############################################################################
######################### set up #############################################
step = 50
norm_type = "original"
expand_eigs = True
runs_ = 10
id_count = 500

##############################################################################
############################ uniform sampling ################################
uni_eig_error_list = []
from Nystrom import simple_nystrom

for k in tqdm(range(10, id_count, 10)):
    error, _, _, _ = simple_nystrom(similarity_matrix, similarity_matrix_O, \
    	k, runs=runs_, mode='eig', normalize=norm_type, expand=expand_eigs)
    uni_eig_error_list.append(error)
    del _
    pass


##############################################################################
#################### CUR decomposition with eig ##############################
# uni_CUR_error_list = []
# from CUR import simple_CUR

# for k in tqdm(range(10, id_count, 10)):
#     error, _, _, _ = simple_CUR(similarity_matrix, similarity_matrix_O, \
#     	k, runs=runs_, mode='normal', normalize=norm_type, expand=expand_eigs)
#     uni_CUR_error_list.append(error)
#     del _
#     pass

##############################################################################
########################## anchornet #########################################
anchor_error_list = []
from anchor import simple_anchor

for k in tqdm(range(10, id_count, 10)):
    error, _, _, _ = simple_anchor(xy, similarity_matrix, similarity_matrix_O, \
    	k, runs=runs_, mode='normal', normalize=norm_type, expand=expand_eigs)
    anchor_error_list.append(error)
    del _
    pass

#######################################################################
# PLOTS
x_axis = list(range(10, id_count, 10))

plt.rc('axes', titlesize=13)
plt.rc('axes', labelsize=13)
plt.rc('xtick', labelsize=13)
plt.rc('ytick', labelsize=13)
plt.rc('legend', fontsize=11)

STYLE_MAP = {"uniform normal error": {"color": "#9ACD32",  "marker": ".", "markersize": 7, 'label': 'Uniform normal', 'linewidth': 1},
             "uniform eig error": {"color": "#EF4026",  "marker": ".", "markersize": 7, 'label': 'Uniform eig', 'linewidth': 1},
             "leverage error": {"color": "#7B3294",  "marker": ".", "markersize": 7, 'label': 'Leverage', 'linewidth': 1},
             "uniform CUR error": {"color": "#7B3294",  "marker": ".", "markersize": 7, 'label': 'Uniform CUR', 'linewidth': 1},
             "anchor net": {"color": "#FAC205",  "marker": ".", "markersize": 7, 'label': 'Anchor Net', 'linewidth': 1},
            }

plt.gcf().clear()
scale_ = 0.55
new_size = (scale_ * 10, scale_ * 8.5)
plt.gcf().set_size_inches(new_size)

title_name = "DonkeyKong Sigmoid"

uniform_eig_error_pairs = [(x, y) for x, y in zip(x_axis, uni_eig_error_list)]
#uniform_norm_error_pairs = [(x, y) for x, y in zip(x_axis, uni_norm_error_list)]
#leverage_error_pairs = [(x, y) for x, y in zip(x_axis, lev_error_list)]
# uniform_CUR_error_pairs = [(x, y) for x, y in zip(x_axis, uni_CUR_error_list)]
anchor_error_pairs = [(x, y) for x, y in zip(x_axis, anchor_error_list)]
arr1 = np.array(uniform_eig_error_pairs)
#arr2 = np.array(leverage_error_pairs)
#arr3 = np.array(uniform_norm_error_pairs)
# arr4 = np.array(uniform_CUR_error_pairs)
arr5 = np.array(anchor_error_pairs)
plt.plot(arr1[:, 0], arr1[:, 1], **STYLE_MAP['uniform eig error'])
#plt.plot(arr2[:, 0], arr2[:, 1], **STYLE_MAP['leverage error'])
#plt.plot(arr3[:, 0], arr3[:, 1], **STYLE_MAP['uniform normal error'])
# plt.plot(arr4[:, 0], arr4[:, 1], **STYLE_MAP['uniform CUR error'])
plt.plot(arr5[:, 0], arr5[:, 1], **STYLE_MAP['anchor net'])
plt.locator_params(axis='x', nbins=6)
plt.ylim(bottom=0.0, top=0.002)
plt.xlabel("Number of landmark samples")
plt.ylabel("Average approximation error")
plt.title(title_name, fontsize=13)
plt.tight_layout(rect=[0, 0.03, 1, 0.95])
plt.legend(loc='upper right')
plt.savefig("figures/comparison_with_kong_sigmoid_errors.pdf")
# plt.savefig("./test1.pdf")
plt.gcf().clear()