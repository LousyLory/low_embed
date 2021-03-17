import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from matplotlib.colors import ListedColormap
import matplotlib as mpl
import sys
from copy import copy
from Nystrom import simple_nystrom
from copy import deepcopy
from utils import read_file, read_labels
import numpy as np
from scipy.stats.stats import pearsonr, spearmanr
from sklearn.metrics import f1_score
from sklearn.metrics import accuracy_score

def is_pos_def(x):
    return np.all(np.linalg.eigvals(x) > 0)

def error_compute(x1, x2):
    return np.linalg.norm(x1-x2)

def add_rows_back(predictions, duplicate_ids, matched_ids):
    # first add rows and columns
    complete_predictions = copy(predictions)
    for i in range(len(duplicate_ids)):
        # print("duplicate id:", duplicate_ids[i])
        current_id = duplicate_ids[i]
        complete_predictions = np.insert(complete_predictions, current_id, np.zeros(complete_predictions.shape[1]), 0)
        complete_predictions = np.insert(complete_predictions, current_id, np.zeros(complete_predictions.shape[0]), 1)
    # print(complete_predictions.shape)
    for i in range(len(duplicate_ids)):
        current_id = duplicate_ids[i]
        matching_id = matched_ids[i]
        complete_predictions[current_id, :] = complete_predictions[matching_id, :]
        complete_predictions[:, current_id] = complete_predictions[:, matching_id]
    return complete_predictions

def find_matches(reshaped_preds, indices):
    # needs to be filled in
    all_indices = list(range(len(reshaped_preds)))
    missing_indices = list(set(all_indices) - set(indices))
    missing_indices = np.sort(missing_indices)
    # first find the exact copies 
    exact_copies = []
    for i in range(len(missing_indices)):
        to_find = reshaped_preds[missing_indices[i],:]
        match_ids = (reshaped_preds == to_find).all(axis=1).nonzero()
        exact_copies.append(match_ids[0])
    # remove the IDs which occur in missing indices for each copies 
    # (this will leave out the ones left in reshaped_preds)
    for i in range(len(exact_copies)):
        removed_duplicates = list(set(exact_copies[i]) - set(missing_indices))[0]
        exact_copies[i] = removed_duplicates
    exact_copies = np.array(exact_copies)
    return missing_indices, exact_copies

def compare_results(A, original_score, method="pearson"):
    true_value = original_score[:,2]
    rows = original_score[:,0].astype(int)
    cols = original_score[:,1].astype(int)
    computed_value = A[rows, cols]

    if method == "pearson":
        results = pearsonr(true_value, computed_value)[0]
    if method == "spearman":
        results = spearmanr(true_value, computed_value)[0]
    if method == "f1":
        computed_value = np.floor(computed_value)
        results = f1_score(true_value, computed_value)
    if method == "accuracy":
        computed_value = 1 - computed_value
        computed_value = np.floor(computed_value)
        # print(true_value, computed_value)
        results = accuracy_score(true_value, computed_value)
    return results

def scale_scores(score1, score2, original_score):
    rows = original_score[:,0].astype(int)
    cols = original_score[:,1].astype(int)
    computed_value = score2[rows, cols]
    min_val_o = np.min(computed_value)
    max_val_o = np.max(computed_value)
    
    min_val_n = np.min(score1[rows, cols])
    max_val_n = np.max(score1[rows, cols])
    # normalize between 0 and 1
    score1[rows, cols] = (score1[rows, cols] - min_val_n) / max_val_n
    score1[rows, cols] = max_val_o*score1[rows, cols] + min_val_o
    return score1

def evaluate(A, original_score, database="stsb"):
    if database == "stsb":
        print("pearson score:", \
            compare_results(A, original_score, method="pearson"))
        print("spearman score:", \
            compare_results(A, original_score, method="spearman"))
        pass
    if database == "mrpc":
        print("F1 score:", \
            compare_results(A, original_score, method="f1"))
        pass
    if database == "rte":
        print("accuracy score:", \
            compare_results(restored_similarity_matrix, original_scores, method="accuracy"))
        pass
    return None

def fix_vals_first(A, original_score):
    rows = original_score[:,0].astype(int)
    cols = original_score[:,1].astype(int)
    computed_scores = A[rows, cols]
    D = np.eye(len(A))
    D[rows, cols] = computed_scores
    D[cols, rows] = computed_scores
    return D

#############################################################################################
#================================= Data extraction =========================================#
# READ THE FILE
dataset = "rte"
num_samples = 200

reshaped_preds = read_file(file_="../GYPSUM/"+dataset+"_predicts_0.npy")
# READ LABELS FROM FILE
original_scores = read_labels("../GYPSUM/"+dataset+"_label_ids.txt")

# reshaped_preds = reshaped_preds
reshaped_preds = fix_vals_first(reshaped_preds, original_scores)
# print(reshaped_preds.shape)

# FIND OUT DUPLICATE ROWS AND ELIMINATE!!
unique_rows, unique_indices = np.unique(reshaped_preds, axis=0, return_index=True)

# FIND DUPLICATE ROWS AND ID THEM WITH MATCHING ROWS
# Duplicate ids == rows in reshaped_preds which has been removed
# Matched ids = rows in reshaped_preds which has a duplicate in duplcate_ids
unique_indices = np.sort(unique_indices)
# checked, and the following line works fine
duplicate_ids, matched_ids = find_matches(reshaped_preds, unique_indices)


# Reshape the predictions to remove duplicate rows
similarity_matrix = copy(reshaped_preds[unique_indices][:, unique_indices])
similarity_matrix = (similarity_matrix+similarity_matrix.T) / 2.0
# restore symmetrized matrix to original shape
restored_similarity_matrix = add_rows_back(similarity_matrix, duplicate_ids, matched_ids)
#############################################################################################

#============================ compute the required scores ================================#
# print("true scores")
# evaluate(reshaped_preds, original_scores, database=dataset)
# print("symmetrized scores")
# evaluate(restored_similarity_matrix, original_scores, database=dataset)
# print(np.min(restored_similarity_matrix), np.max(restored_similarity_matrix), \
#     np.min(original_scores[:,2]), np.max(original_scores[:,2]))
# print("original error")
# print(np.linalg.norm(reshaped_preds-restored_similarity_matrix)/np.linalg.norm(reshaped_preds))
#########################################################################################

#================================== approximation =====================================#
# error, abs_error, avg_min_eig, approx_sim_mat = simple_nystrom(\
#                                         deepcopy(similarity_matrix),\
#                                         deepcopy(similarity_matrix),\
#                                         num_samples,\
#                                         runs=10,\
#                                         mode="eigI",\
#                                         normalize="original",\
#                                         expand=True)
# print(error, abs_error)
#########################################################################################

#============================ compute approximation error ==============================#
# approx_sim_mat = add_rows_back(approx_sim_mat, duplicate_ids, matched_ids)
# # approx_sim_mat = scale_scores(approx_sim_mat, reshaped_preds, original_scores)
# print("approximate similarity scores "+str(num_samples))
# print(np.min(approx_sim_mat), np.max(approx_sim_mat), \
#     np.min(original_scores[:,2]), np.max(original_scores[:,2]))
# evaluate(approx_sim_mat, original_scores, database=dataset)
#########################################################################################

#================================= checking for RTE ====================================#
# grab the scores
rows = original_scores[:,0].astype(int)
cols = original_scores[:,1].astype(int)
computed_scores_forward = reshaped_preds[rows, cols]
computed_scores_backward = reshaped_preds[cols, rows]
print(pearsonr(computed_scores_forward, computed_scores_backward)[0])
print(spearmanr(computed_scores_forward, computed_scores_backward)[0])
#########################################################################################