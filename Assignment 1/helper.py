import numpy as np
import cv2 as cv
import matplotlib.pyplot as plt
import math


# padding function
def padd(target, padding):
    #image size
    n, m = target.shape # n x m
    newTarget = np.zeros((target.shape[0] + 2*padding,\
                          target.shape[1] + 2*padding))
    newTarget[padding:n + padding, padding:m + padding] = target
    return newTarget


# covolution function
def covolv(target, filtr):
    #assuming the filter is odd x odd
    padding = int(int(filtr.shape[0])/2)
    filterSize = filtr.shape[0]
    filtr = filtr.T
    padTarget = padd(target, padding)
    NewTarget = padd(target, padding)
    
    n, m = target.shape # n x m
    for i in range(padding, n + padding):
        for j in range(padding, m + padding):
            s = 0
            for l in range(filterSize):
                for k in range(filterSize):
                    s += filtr[l, k] * padTarget[i + l - padding,\
                                                 j + k - padding]
            NewTarget[i,j] = s
    return NewTarget[padding:n + padding, padding:m + padding]


# gaussian filter generator function
def gaussian_eq(dim, std):
    filtr = np.zeros((dim, dim))
    padding = int(dim/2)

    for x in range(dim - padding):
        for y in range(dim - padding):
            # using the gaussian equation to find the value of pixle
            g = 1./(2*math.pi*std**2) * math.exp(-(x**2 + y**2)/(2*std**2))
            filtr[x + padding, y + padding] = g * 1e0
            
    '''
    Our matrix till now is of the form(assuming dim=5):
    
    0 | 0 | 0 | 0 | 0
    0 | 0 | 0 | 0 | 0
    0 | 0 | O | v | x
    0 | 0 | v | x | w
    0 | 0 | x | w | z
    
    Using some flipping functions to populate the
    rest of the matrix (filter) to finally get
    
    z | w | x | w | z
    w | x | v | x | w
    x | v | O | v | x
    w | x | v | x | w
    z | w | x | w | z
    '''

    filtr[:padding, :] = np.flip(filtr[dim - padding:, :], axis=0)
    filtr[:, :padding] = np.flip(filtr[:, dim - padding:], axis=1)
    
    # normalizing the filter
    sumx = np.sum(filtr, axis=1)
    sumy = np.sum(sumx, axis=0)
    return filtr


# printing in scientific notations
def prettyPrint(ary):
    n,m = ary.shape
    for i in range(n):
        for j in range(m - 1):
            print('{:.3e}'.format(ary[i,j]) + ' | ', end='')
        print ('{:.3e}'.format(ary[i,m-1]))

# white balance the resulting matrix