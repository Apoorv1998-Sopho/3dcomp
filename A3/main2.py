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
lowsR = .95 # low's ratio, taking big, cz not many matchings
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
keyPtsDivided = keypt_divide_depth(dimages, depthNames, keyPointMatchings, dlevels, depth_quantum)
# print(len(keyPtsDivided[0]))

##########################################################
#Find HomoGraphies
##########################################################
n = 1000
r = 4
t = 2
Tratio = 0.95

# Dict containing HomoGraphies
print('Finding HomoGraphies')
Hs = {}
for i in range(dlevels):
    print(type(keyPtsDivided[i]))
    list_kp = keyPtsDivided[i]
    H, S = findHomoRanSac(n, r, list_kp, t, Tratio)
    Hs[i] = H
print('Done HomoGraphies')
print('HomoGraphies:', Hs)


##########################################################
#Warp
##########################################################

# create canvas
factor = [int(imageNos*3), int(imageNos*5)] # dy,dx
offset = [[3000,1000]] # x,y
canvas2 = createCanvas(images[imagesNames[0]], factor)

#drawing unblended
print('drawing ', end= '')
for i in range(0, imageNos):
    print(imagesNames[i], end=' ')
    drawOnCanvas(canvas2, images[imagesNames[i]], Hss[i], offset, abs(int(imageNos/2)-i)+1, weightDic=None)
canvas2 = canvas2.astype(np.uint8)

# stripping
print ('stripping')
true_points = np.argwhere(canvas2)
top_left = true_points.min(axis=0)
bottom_right = true_points.max(axis=0)
out = canvas2[top_left[0]:bottom_right[0]+1,  # plus 1 because slice isn't
             top_left[1]:bottom_right[1]+1]  # inclusive
print('done stripping')

# spitting
cv.imshow("./result/"+pathI+"unblended.jpg", out)
cv.waitKey(0)
sys.exit()
for i in range(dlevels):
    dname = depthNames[i]
    dimg = dimages[dname]



sys.exit()