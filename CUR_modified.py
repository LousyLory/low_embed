import numpy as np

def CUR(similarity_matrix, k, eps=1e-3, delta=1e-14, return_type="error"):
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
	# setting up c, r, eps, and delta for error bound
	c = (64*k*((1+8*np.log(1/delta))**2) / (eps**4)) + 1
	r = (4*k / ((delta*eps)**2)) + 1
	# c = 4*k
	# c = k
	# r = k
	if c > n:
		c= n
	# r = 4*k
	if r > n:
		r = n
	print("chosen c, r:", c,r)
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
	# U = np.linalg.inv(rank_k_C.T @ rank_k_C)
	U = rank_k_C.T @ rank_k_C
	# i chose not to compute rank k reduction of U
	U = U @ psi.T
	U = np.linalg.pinv(U)

	if return_type == "decomposed":
		return C, U, R
	if return_type == "approximate":
		return (C @ U) @ R
	if return_type == "error":
		# print(np.linalg.norm((C @ U) @ R))
		relative_error = np.linalg.norm(similarity_matrix - ((C @ U) @ R)) / np.linalg.norm(similarity_matrix)
		return relative_error


X = np.random.random((500,500))
X = X.T @ X
k = 200
eps = 1e-7
print("error:", CUR(X, k, eps=eps, return_type="error"))
s,v,d = np.linalg.svd(X, full_matrices=False)
samples = range(k)
s = s[:, samples]
v = np.diag(v[samples])
d = d[samples, :]
rank_k_X = (s @ v) @ d
print("true error: ", np.linalg.norm(X - rank_k_X)+eps*np.linalg.norm(X))