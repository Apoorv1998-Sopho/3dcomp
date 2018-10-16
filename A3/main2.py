import cv2 as cv
import numpy as np
import sys
import matplotlib.pyplot as plt
from helper import *
##########################################################
#Reading files
##########################################################
path = './RGBD dataset/'
imagesNames = ['a.jpg', 'b.jpg', 'c.jpg', 'd.jpg']#, 'e.jpg', 'f.jpg']
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
goodMatchings={}
for i in range(imageNos-1):
    imgA = imagesNames[i]
    imgB = imagesNames[i+1]
    goodMatchings[(imgA,imgB)]= keyPointMatching(images, 
                        imageKeyPoints, imageDescriptors,
                        imgA, imgB, dirr)
print('done keymatches')

##########################################################
#Quantize the depth image
##########################################################
'''
Quantized is a dict
Quantized['da.jpg']=[imgdpt1, imgdpt2, imgdpt3...]
'''
def Quantize(dimages, depthNames, dlevels=5):
    # find the max depth
    for i in range(len(dimages)):
        if i == 0:
            max_depth = np.max(dimages[depthNames[i]])
        max_depth = max(max_depth, np.max(dimages[depthNames[i]]))
    # print ('max_depth',max_depth) = 255
    depth_quantum = int(max_depth/dlevels)
    print(depth_quantum)
    for i in range(len(dimages)):
        dimages[depthNames[i]] = (dimages[depthNames[i]]/dlevels).astype(np.uint8) #only dlevels
        dimages[depthNames[i]] *= depth_quantum
    return dimages

dimages = Quantize(dimages,depthNames)
# print(dimages[depthNames[0]])
# cv.imshow("Quantized image", dimages[depthNames[0]])
# cv.waitKey(0)



sys.exit()