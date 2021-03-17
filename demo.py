import numpy as np
from utils import read_file
from numpy import linalg as LA
from tqdm import tqdm
import random
import seaborn as sns
from matplotlib.colors import ListedColormap
import matplotlib.pyplot as plt

K = read_file(1000)

# remove exact copies of rows
unique_rows, indices = np.unique(K, axis=0, return_index=True)
KO = K[indices][:, indices]

# symmetrize the matrix
K = (KO+KO.T)/2.0

# eigenvalues and eigenveectors
D, V = LA.eig(K)

# len of samples
n = len(K)

errors = []
errors1 = []
for i in tqdm(range(10, n, 10)):
    list_of_available_indices = list(range(n))
    s = np.sort(random.sample(list_of_available_indices, i))
    z = np.sort(random.sample(list_of_available_indices, int(np.ceil(np.sqrt(i*n)))))
    K_hat = K[z][:, z]
    Ds, Vs = LA.eig(K_hat)
    min_eig = min(0, np.min(Ds)) - 0.0001
    K_bar = K - min_eig*np.eye(n)
    err = LA.norm(K - K_bar[:][:, s] @ LA.inv(K_bar[s][:, s]) @ K_bar[s])
    errors.append(err / LA.norm(K))
    errors1.append(err / LA.norm(KO))

# display
sns.set()
flatui = ["#9b59b6", "#3498db", "#95a5a6", "#e74c3c", "#34495e", "#2ecc71"]
cmap = ListedColormap(sns.color_palette(flatui).as_hex())
x = list(range(10, n, 10))
fig = plt.gcf()
fig.set_size_inches(15, 9, forward=True)
plt.plot(x, errors, label="average errors normalized with symmetrized K")
plt.plot(x, errors1, label="average errors normalized with K")
plt.xlabel("number of reduced samples")
plt.ylabel("error score")
plt.legend(loc="upper right")
plt.title("plot of average errors using Nystrom on a non-PSD similarity matrix")
plt.savefig("figures/nystrom_errors_new.pdf")
