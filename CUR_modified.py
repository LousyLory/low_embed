import numpy as np

def CUR(similarity_matrix, c, r, k, return_type="error"):
	"""
	implementation of Linear time CUR algorithm of Drineas2006 et. al.

	input:
	1. similarity matrix in R^{n,d}
	2. integers c, r, and k

	output:
	1. either C, U, R matrices
	or
	1. CU^+R
	or
	1. error = similarity matrix - CU^+R

	"""
	n,d = similarity_matrix.shape
	try:
		assert 1 <= c and c <= d
	except AssertionError as error:
		print("1 <= c <= m is not true")
	try:
		assert 1 <= r and r <= n
	except AssertionError as error:
		print("1 <= r <= n is not true")
	try:
		assert 1 <= k and k <= min(c,r)
	except AssertionError as error:
		print("1 <= k <= min(c,r)")

	# using uniform probability instead of row norms
	pj = np.ones(d).astype(float) / float(d)
	qi = np.ones(n).astype(float) / float(n)

	# choose samples
	samples_c = np.random.choice(range(d), c, replace=False, p = pj)
	samples_r = np.random.choice(range(n), r, replace=False, p = qi)

	# grab rows and columns and scale with respective probability
	samp_pj = pj[samples_c]
	samp_qi = qi[samples_r]
	C = similarity_matrix[:, samples_c] / np.sqrt(samp_pj*c)
	# C = C.T
	k = min(k, np.linalg.matrix_rank(C.T @ C))
	s,v,d = np.linalg.svd(C.T @ C, full_matrices=False)
	s = s[:, range(k)]
	v = np.diag(v[range(k)])
	d = d[range(k), :]
	# print(s.shape, v.shape, d.shape)
	rank_k_C = (s @ v) @ d
	# modification works only because we assume similarity matrix is symmetric
	R = similarity_matrix[:, samples_r] / np.sqrt(samp_qi*r)
	R = R.T
	psi = C[samples_r, :].T / np.sqrt(samp_qi*r)
	psi = psi.T

	# U = np.linalg.pinv(C.T @ C)
	U = np.linalg.pinv(rank_k_C.T @ rank_k_C)
	# i chose not to compute rank k reduction of U
	U = U @ psi.T

	if return_type == "decomposed":
		return C, U, R
	if return_type == "approximate":
		return (C @ U) @ R
	if return_type == "error":
		relative_error = np.linalg.norm(similarity_matrix - ((C @ U) @ R)) / np.linalg.norm(similarity_matrix)
		return relative_error


# X = np.random.random((500,500))
# X = X.T @ X
# print("error:", CUR(X, 100, 100, 100, return_type="error"))