'''
Author: apoorv agnihotri
16110020

Assignment 2: Sift
'''

import numpy as np
import ../Assignment1/helper as hp

'''
here dim is the dimention of gaussian filter
'''
class SIFT(object):
    def __init__ (self, sigma, k=2**.5, dim=5\
                  numofoctaves=4, numofscales=5):
        self.sigma = sigma
        self.k = k
        self.numofoctaves = numofoctaves
        self.numofscales = numofscales
        self.dim = dim



def generateScales(img):
    scales=[]
    for i in range(self.numofscales):
        #find the related sigam
        newsigma = self.sigma*(self.k**i)
        scales.append(self.blur(img, newsigma))
    return scales

def generateOctaves(img):
    octaves=[]
    for i in range(self.numofoctaves):
        newsigma = (self.k ** (2*i - 1)) * self.sigma
        newbaseimage= cv.resize(img, None, \
                            fx = scale, \
                            fy = scale, \
                            interpolation = cv.INTER_CUBIC)
        octaves.append(generateScales(img,newsigma))

        

'''
Runs a normalized gaussian filter over the image
and returns the result with same datatype
'''
def blur(img, newsigma)
    gauss = hp.gaussian_nor(newsigma)
    return hp.covolv(img, gauss)

