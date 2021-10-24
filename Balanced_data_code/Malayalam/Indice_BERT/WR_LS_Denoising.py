from scipy.sparse import spdiags
#import numpy as np
import cupy as cp

def WR_LS_Denoise(y,lamda=0.01):
    N = len(y)
    lam = lamda
    x = cp.zeros((N-2,N), dtype=int)
    for i in range(N-2):
        x[i,i] = 1
        x[i,i+1] = -2
        x[i, i+2] = 1
    z = cp.eye(N,dtype = int) + lam*cp.dot(cp.transpose(x),x)
    return cp.dot(cp.linalg.inv(z),y)