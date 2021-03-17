import multiprocessing
from joblib import Parallel, delayed
from tqdm import tqdm
import numpy as np


from joblib import Parallel, delayed

def myfun(multiplier = 5, arg=2, adder=3):
    result = multiplier*(arg**2)+adder
    return result

def top(arg):
    multiplier = 2
    adder = 5
    result = myfun(multiplier = multiplier, arg=arg, adder=adder)
    return result

inputs = tqdm(range(10, 20000, 10))

results = Parallel(n_jobs=-1, backend="threading")(
             map(delayed(top), inputs))

print(results)