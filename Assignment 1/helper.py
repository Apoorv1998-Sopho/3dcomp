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
def covolv(target, filtr, typ=None):
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

    # returning trimmed big matrix and typecated to type provided
    if typ == None:
        typ = filtr.dtype
    return NewTarget[padding:n + padding, padding:m + padding].astype(typ)


# gaussian filter generator function
def gaussian_eq(dim, std):
    filtr = np.zeros((dim, dim))
    padding = int(dim/2)

    for fx in range(dim):
        for fy in range(dim):
            x = fx - padding
            y = fy - padding
            # using the gaussian equation to find the value at pixel x,y
            g = 1./(2*math.pi*std**2) * math.exp(-(x**2 + y**2)/(2*std**2))
            filtr[fx, fy] = g

    # normalizing the filter
    normalizer = 0
    for i in range(dim):
        for k in range(dim):
            normalizer += filtr[i,k]
            
    # dividing the filter elements with the sum of all the elements.
    for i in range(dim):
        for k in range(dim):
            filtr[i,k] /= normalizer
    
    return filtr


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

''' 
brief:
detecting Zero Crossings, in a given array.

@param arry: input matrix
@param method: among the possible methods
  choose a specific method to find zero
  crossings
  
  (0) - see for sign change (+ to -) in 4
   neigbours.
  
  (1) - make any pixel within epsilon 255 (edge)
  
@param epsilon corresponds to method = 1
'''
def detectZeroCrossings(arry, method = 0, epsilon = 1):
    n, m = arry.shape
    r = np.zeros((n,m), dtype=np.uint8)
    
    if (method == 0):
        for i in range(n):
            for k in range(m):
                try:
                    s = min(arry[i,k-1], arry[i,k+1], \
                            arry[i-1,k], arry[i+1,k])
                    if(arry[i,k]>=0 and s < 0):
                        r[i,k] = 255
                except IndexError:
                    r[i,k] = 255
    else: #(method == 1)
        for i in range(n):
            for k in range(m):
                if (abs(arry[i, k]) <= epsilon):
                    r[i,k] = 255
                else:
                    r[i,k] = 0
    
    return r