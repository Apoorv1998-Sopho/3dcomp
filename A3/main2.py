import cv2 as cv
import numpy as np
import sys
import matplotlib.pyplot as plt
from helper import *
##########################################################
#Reading files
##########################################################
# setting some variables
warp_usual = False
dlevels = 5
pathI = 'D'
path = './RGBD dataset/' + pathI + '/'
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
    dimages[dimg] = temp # depth imgs are rowxcolx3 with 3 values being same
del temp


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

dimages, depth_quantum = Quantize(dimages,depthNames, dlevels)
keyPtsDivided = keypt_divide_depth(dimages, depthNames, keyPointMatchings, dlevels, depth_quantum)

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
    list_kp = keyPtsDivided[i]
    try:
        H, S = findHomoRanSac(n, r, list_kp, t, Tratio)
    except ValueError: # when not enough points
        H = None
    Hs[i] = H
print('Done HomoGraphies')
print('HomoGraphies:', Hs)

# interpolating Those H which couldn't be calculated
for i in range(dlevels):
    if Hs[i].tolist() == None:
        try:
            Hs[i] = (Hs[i-1] + Hs[i+1])/2
        except KeyError:
            try:
                Hs[i] = Hs[i-1]
            except KeyError:
                Hs[i] = Hs[i+1]

##########################################################
#Warp
##########################################################

# create canvas
factor = [int(imageNos*3), int(imageNos*5)] # dy,dx
offset = [[3000,1000]] # x,y
canvas2 = createCanvas(images[imagesNames[0]], factor)

# making regions for first img
regions = {}
dimg = dimages[depthNames[0]]
print('Finding regions')
for i in range(dlevels): # only want to warm one img
    depth = depth_quantum * i
    truevals = dimg == depth
    regions[i] = np.multiply(images[imagesNames[0]], truevals)
    # cv.imshow('r', regions[i])
    # cv.waitKey(0)
print('done regions')

# printing ref img
print('drawing ', end= '')
# drawOnCanvas(canvas2, images[imagesNames[1]], np.eye(3), offset, fill=1, weightDic=None)
for i in range(dlevels):
    print('dlevel:', i, end=' ')
    drawOnCanvas(canvas2, regions[i], Hs[i], offset, 
                 fill=2, weightDic=None, blackPixelPrint=False)
canvas2 = canvas2.astype(np.uint8)

# stripping
print ('stripping')
out = strip(canvas2)
print('done stripping')

# spitting
cv.imshow("./result/"+pathI+"homographed.jpg", out)
cv.imwrite("./result/"+pathI+"homographed.jpg", out)
cv.waitKey(0)
sys.exit()
