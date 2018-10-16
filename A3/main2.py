import cv2 as cv
import numpy as np
import sys
import matplotlib.pyplot as plt
from helper import *
##########################################################
#Reading files
##########################################################
path = './RGBD dataset/'
imagesNames = ['a.jpg', 'b.jpg']
depthNames = ['d'+img for img in imagesNames]
scale = (1, 1)
images = {} # will have 3 channel color imgs
dimages = {} # will have 1 chnl imgs
imageNos = len(imagesNames)
m = 3
k = 4

##########################################################
#Rescaling
##########################################################
for img in imagesNames:
    print(path + img)
    temp = cv.imread(path + img)
    temp = cv.resize(temp, None, fx=scale[0], 
                     fy=scale[1], interpolation=cv.INTER_CUBIC)
    images[img] = temp
for dimg in depthNames:
    print(path + dimg)
    temp = cv.imread(path + dimg)
    temp = cv.resize(temp, None, fx=scale[0], 
          	         fy=scale[1], interpolation=cv.INTER_CUBIC)
    temp = cv.cvtColor(temp, cv.COLOR_BGR2GRAY)
    dimages[dimg] = temp
del temp
# print('images.shape',images[imagesNames[0]].shape)
# print('dimages.shape',dimages[depthNames[0]].shape)

##########################################################
#Finding KeyPoints and Discriptors
##########################################################
print('finding keypoints and discriptors')
imageKeyPoints, imageDescriptors = keyPoints(images, imagesNames)
print('done keypoints and discriptors')
# retured dictionaries with keys as imageNames

##########################################################
#Finding matchings for best 'm' matching images for each image
##########################################################
print('finding keymatches')
dirr = './result/Part2'
imgA = imagesNames[0]
imgB = imagesNames[1]
lowsR = 111000 # low's ratio, taking big, cz not many matchings
keyPointMatchings = keyPointMatching(images, imageKeyPoints,
                        imageDescriptors,
                        imgA, imgB, dirr, lowsR)
print('done keymatches')

##########################################################
#Quantize the depth image
##########################################################
dlevels = 10
dimages, depth_quantum = Quantize(dimages,depthNames, dlevels)
# print(dimages[depthNames[0]])
cv.imshow("Quantized image", dimages[depthNames[0]])
cv.waitKey(0)

'''
Returns a dlevel length dictionary with keys as depthlevel
Works to divide the keypoints in the first image of dimages
'''
# print(keyPointMatchings[0][0])
# sys.exit()
def keypt_divide_depth(dimages, depthNames, keyPointMatchings, dlevels, depth_quantum):
    dname = depthNames[0]
    keyPtsDivided = {}
    length = len(keyPointMatchings)
    print ('# of KeyPoints matchings', length)
    for i in range(length):
        x,y = keyPointMatchings[0][i] # selecting all keypoints in first img
        xi = int(x)
        yi = int(y)
        depthVal=dimages[dname][yi,xi]
        dlevel = int(depthVal/depth_quantum)
        # print('value of depth:', depthVal)
        # print('dlevel of coordinate:', dlevel)
        
        try:
            keyPtsDivided[dlevel].append([keyPointMatchings[0][i], keyPointMatchings[1][i]])
        except:
            keyPtsDivided[dlevel] = [keyPointMatchings[0][i], keyPointMatchings[1][i]]
    return keyPtsDivided


keyPtsDivided = keypt_divide_depth(dimages, depthNames, keyPointMatchings, dlevels, depth_quantum)
print(keyPtsDivided)
sys.exit()
for i in range(dlevels):
    pass

for i in range(dlevels):
    dname = depthNames[i]
    dimg = dimages[dname]



sys.exit()