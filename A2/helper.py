'''
Author: apoorv agnihotri
16110020

Assignment 2: Sift
'''

import numpy as np
import cv2 as cv
from old_helper import *

'''
here dim is the dimention of gaussian filter
'''
class SIFT(object):
    def __init__ (self, sigma, k=2**.5, dim=5,\
                  numofoctaves=4, numofscales=5):
        self.sigma = sigma
        self.k = k
        self.numofoctaves = numofoctaves
        self.numofscales = numofscales
        self.dim = dim

    def generateScales(self, img):
        scales=[]
        for i in range(self.numofscales):
            #find the related sigam
            newsigma = self.sigma*(self.k**i)
            scales.append(self.blur(img, newsigma))
        return scales

    def generateOctaves(self, img):
        octaves=[]
        for i in range(self.numofoctaves):
            newsigma = (self.k ** (2*i - 1)) * self.sigma
            scale = 1./(2**i)
            newbaseimage= cv.resize(img, None, \
                                fx = scale, \
                                fy = scale, \
                                interpolation = cv.INTER_CUBIC)
            octaves.append(generateScales(img,newsigma))
        return octaves


    '''
    Runs a normalized gaussian filter over the image
    and returns the result with uint8 datatype
    '''
    def blur(self, img, newsigma):
        gauss = hp.gaussian_nor(self.dim, newsigma)
        return hp.covolv(img, gauss, np.uint8)

