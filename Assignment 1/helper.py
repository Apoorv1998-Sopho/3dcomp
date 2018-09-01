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
    n, m = target.shape # n x m

    # gave extra padding to original image and made a big matrix to return
    padTarget = padd(target, padding)
    NewTarget = np.zeros((n + padding, m + padding))
    
    #move filter over all points in matrix
    for trxx in range(padding, n + padding):
        for tryy in range(padding, m + padding):
            s = 0
            
            #find the dot product of filter + submatrix
            for fx in range(filterSize):
                for fy in range(filterSize):
                    srx = trxx + fx - padding
                    sry = tryy + fy - padding
                    s += filtr[fx, fy] * padTarget[srx, sry]
            NewTarget[trxx,tryy] = s

    # returning trimmed big matrix and typecated unsignedint8
    return NewTarget[padding:n + padding, padding:m + padding].astype(np.uint8)


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
    return filtr / sumy


# printing in scientific notations
def prettyPrint(ary):
    n,m = ary.shape
    for i in range(n):
        for j in range(m - 1):
            print('{:.3e}'.format(ary[i,j]) + ' | ', end='')
        print ('{:.3e}'.format(ary[i,m-1]))

def readImg_Grey_Resize(file, scale=1):
    I1 = cv.imread('imgs/3_1.jpg')
    height, width, channels = I1.shape
    
    # show raw img
    cv.namedWindow('raw', cv.WINDOW_NORMAL)
    cv.imshow('raw',I1)
    cv.waitKey(0)
    cv.destroyAllWindows()
    # convert to grey
    I1_gray = cv.cvtColor(I1,cv.COLOR_BGR2GRAY);
    
    # show greyed image
    cv.namedWindow('grey', cv.WINDOW_NORMAL)
    cv.imshow('grey',I1_gray)
    cv.waitKey(0)
    cv.destroyAllWindows()
    
    # resize img
    if (scale != 1):
        I1_resized = cv.resize(I1_gray,None, fx = scale, fy = scale, \
                         interpolation = cv.INTER_CUBIC)
        cv.namedWindow('resized', cv.WINDOW_NORMAL)
        cv.imshow('resized',I1_resized)
        cv.waitKey(0)
        cv.destroyAllWindows()
        return I1_resized
    
    else:
        return I1_gray
        
# white balance the resulting matrix