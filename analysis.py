import numpy as np
from utils import read_file

filename = "mrpc"
print("Reading file ...")
# similarity_matrix = read_mat_file(file_="WordMoversEmbeddings/mat_files/oshumed_K_set1.mat", version="v7.3")
similarity_matrix = read_file("../GYPSUM/"+filename+"_predicts_0.npy")

print("File read. Beginning preprocessing ...")

# check for similar rows or columns
unique_rows, indices = np.unique(similarity_matrix, axis=0, return_index=True)
similarity_matrix_O = similarity_matrix[indices][:, indices]
similarity_matrix = (similarity_matrix_O + similarity_matrix_O.T) / 2.0
print("Preprocessing done.")

# compute svd
s,v,d = np.linalg.svd(similarity_matrix, full_matrices=False)
eigs,_ = np.linalg.eigs(similarity_matrix)
print(v.shape)
print(eigs.shape)
v = v[v>=0]
# print(v)
s = s[:,v>=0]
d = d[v>=0,:]

new_mat = s@np.diag(v)@d

print("error:", np.linalg.norm(similarity_matrix-new_mat))