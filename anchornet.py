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
from measures import sigmoid
import itertools
from statsmodels.tools.sequences import halton
import random
import matplotlib.pyplot as plt

def AnchorNet_construction(data, s, q):
    """
    implementation of anchornet algorithm 5.1
    inputs: 
    1. data in R^{n x d}
    2. integer s
    3. integer q

    output:
    1. AnchorNet: A_x_p
    """
    [n,d] = data.shape
    # create low discrepancy set with s points in the smallest box B0 that contains X
    # (line 1 of algorithm 5.1)
    T = halton(d, s)
    # rescale the sequences
    Omega_end = np.max(data, axis=0)
    Omega_begin = np.min(data, axis=0)
    T = T*(Omega_end-Omega_begin)+Omega_begin

    # plt.scatter(data[:,1], data[:,0], marker="o")
    # plt.scatter(T[:,1], T[:,0], marker="^")
    # plt.show()

    # create groups (lines 2-6 algorithm 5.1)
    G = {}
    for k in range(len(T)):
        G[k] = []
    for j in range(n):
        index = np.argmin(np.max(np.abs(0-(T-data[j,:])), axis=1))
        # print(index)
        G[index].append(j)

    # get the indices of non empty groups (line 7 algoithm 5.1)
    set_of_nonempty_indices = []
    for i in range(len(T)):
        if G[i] == []:
            # print(i)
            pass
        else:
            set_of_nonempty_indices.append(i)
    Q = len(set_of_nonempty_indices)

    # find smallest box bi that contains gi and compute lebesgue bi (lines 8-10 algorithm 5.1)
    lebesgue_Q = []
    # B = []
    for i in set_of_nonempty_indices:
        # B.append(np.max(data[G[i],:], axis=1))
        m = np.prod(np.max(data[G[i],:], axis=0))
        lebesgue_Q.append(m)

    # Choose Q low discrepancy sets parameterized by p1 to pQ such that pi <= q (line 11 algorithm 5.1)
    A = []
    for i in range(Q):
        pi = len(G[i])
        if pi <= q:
            A.append(G[i])
        else:
            A.append(random.sample(G[i], k=q))

    # union of anchor nets A (line 12 algorithm 5.1)        
    anchornet_x_p = list(itertools.chain.from_iterable(A))
    return anchornet_x_p

def AnchorNet(data, s, q):
    """
    Implementation of anchornet algorithm 5.2 
    
    inputs: 
    1. data in R^{n x d}
    2. integer s
    3. integer q

    output:
    1. Samples: S
    """

    # create anchornet for data (line 1 of alorithm 5.2)
    # print(s,q)
    anchornet_data_p = AnchorNet_construction(data, s, q)
    # print(len(anchornet_data_p))
    # grab the samples (line 2-5 of algorithm 5.2)
    S = []
    for i in range(len(anchornet_data_p)):
        index = np.argmin(np.max(np.abs(0-(data-data[anchornet_data_p[i],:])), axis=1))
        S.append(index)
    return S
