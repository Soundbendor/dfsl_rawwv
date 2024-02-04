import torch
import numpy as np
# https://stackoverflow.com/questions/66832716/how-to-quickly-inverse-a-permutation-by-using-pytorch answer from BinChen
def rev_perm(perm):
    inv = torch.empty_like(perm)
    inv[perm] = np.arange(0,perm.shape(0))
    return inv
