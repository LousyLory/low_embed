"""
Copyright (c) 2021 Archan Ray

Permission is hereby granted, free of charge, to any person obtaining a copy of 
this software and associated documentation files (the "Software"), to deal in the 
Software without restriction, including without limitation the rights to use, copy, 
modify, merge, publish, distribute, sublicense, and/or sell copies of the Software, 
and to permit persons to whom the Software is furnished to do so, subject to the 
following conditions:

The above copyright notice and this permission notice shall be included in all 
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR IMPLIED, 
INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY, FITNESS FOR A PARTICULAR 
PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR COPYRIGHT HOLDERS BE 
LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT, 
TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE 
USE OR OTHER DEALINGS IN THE SOFTWARE.
"""

import numpy as np
from sklearn.metrics import pairwise_distances as euclid

def sigmoid(data1, data2, sigma=1):
	similarity_matrix = np.matrix(data1) * np.matrix(data2.T)
	similarity_matrix = (similarity_matrix / sigma) + 1.0
	similarity_matrix = np.tanh(similarity_matrix)
	return similarity_matrix

def tps(data1, data2, sigma=1.0):
	eps=1e-16
	similarity_matrix = np.square(euclid(data1, data2))+eps
	# print(similarity_matrix.shape, np.min(similarity_matrix), np.max(similarity_matrix))
	similarity_matrix = similarity_matrix / (sigma**2)
	similarity_matrix = similarity_matrix * np.log(similarity_matrix)
	return similarity_matrix