'''
Author: Apoorv Agnihotri

Code samples from:
https://kushalvyas.github.io/stitching.html
'''
import cv2 as cv
from src import *
import numpy as np
import matplotlib.pyplot as plt


##########################################################
# Hyperparameters
##########################################################
width_epipolar = 3 # width/2 of the epiline to chck correspondences
method = 'sift' # when we use sift or basic discriptor
channel = 'RGB' # when we use basic dicriptor
lowsR = 0.7 # low's ratio
width = 200
pathI = 'I3' # image path

##########################################################
#Reading files
##########################################################
path = '../data/' + pathI + '/'
imagesNames = ['a.png', 'b.png']
images = {} # will have 3 channel color imgs
imageNos = len(imagesNames)
imgB = 1

##########################################################
#Rescaling
##########################################################
images = {}
for i in range(len(imagesNames)):
    img = imagesNames[i]
    print(path + img)
    temp = cv.imread(path + img)
    scale_percent = width*100/temp.shape[1]
    height = int(temp.shape[0] * scale_percent / 100)
    dim = (width, height) 
    temp = cv.resize(temp, dim, interpolation=cv.INTER_CUBIC)
    images[i]=temp
del temp

##########################################################
#Finding KeyPoints and Discriptors
##########################################################
print('finding keypoints and discriptors')
imageKeyPoints, imageDescriptors = keyPoints(images)
print('done keypoints and discriptors')

##########################################################
#Finding matchings
##########################################################
print('finding keymatches')
goodMatchings={}
for i in range(imageNos-1):
    imgA = i
    goodMatchings[(imgA,imgB)]= keyPointMatching(images, 
                              imageKeyPoints, imageDescriptors, 
                              imgA, imgB, lowsR)
print('done keymatches')

##########################################################
# Finding F the pair of image
##########################################################
print('finding Fundamentals')
for i in range(imageNos -1):
    imgA = i
    list_kp = goodMatchings[(imgA, imgB)]
    # finding the fundamental matrices, using ransac
    F, S = cv.findFundamentalMat(np.array(list_kp[0]), np.array(list_kp[1]))
    F = F.T
print('done Fundamentals')
print('Fundamental Matrix:', F)

##########################################################
# Change to LAB if wanted
##########################################################

##########################################################
# For each point in I1, find epiline
##########################################################
print('Finding the epilines for each point')
I1 = images[0]
I2 = images[1]
h, w, chl = I1.shape
# make a list of all the points
listOfPoints = []
for y in range(h):
    for x in range(w):
        pt = (x, y)
        listOfPoints.append(pt)
# compute epilines for each pt
epilines = findEpilines(np.array(listOfPoints), F)
print('done computing epilines')

##########################################################
# For each point in I1, find the corresp I2 point
##########################################################
print('Find the point corresp to every point in,', pathI,', please wait..')
h, w, chl = I2.shape
Iout = np.zeros((h, w, chl))
d2 = {} # for storing sift discriptors for I2
d1 = {} # for storing sift discriptors for I1
for i in range(len(epilines)):
	# if i*10 == len(epilines)print (i/len(epilines))
    epiline = epilines[i]
    vPts = findValidPoints(epiline, (h, w), width_epipolar)
    if len(vPts) == 0: # no valid point
        # print('no valid point')
        # print('epiline, pt', epiline, listOfPoints[i])
        continue

    # returns discriptors in rows
    discriptors = findCustomDiscriptor(I2, vPts, d2, listOfPoints, channel=channel, method=method)
    if discriptors == None:
    	continue
    if discriptors != None:
        ptI1 = listOfPoints[i]
        I1ptDiscriptor = findCustomDiscriptor(I1, [ptI1], d1, listOfPoints, channel=channel, method=method) # can return empty array
        if I1ptDiscriptor != None:
            keys = list(discriptors.keys())
            dis = np.array([discriptors[key] for key in keys])
            temp = dis - I1ptDiscriptor[ptI1].astype(float)
            temp = np.linalg.norm(temp, axis=1)
            min_index = np.argmin(temp)
            argminKey = keys[min_index]
            y_ = argminKey[1]
            x_ = argminKey[0]
            y = ptI1[1]
            x = ptI1[0]
            Iout[y_,x_] = I1[y,x]
Iout = Iout.astype(np.uint8)
# plt.imshow(Iout)
# plt.show()
cv.imwrite('../result/'+pathI+ 'channel' +channel+
    'scale' + str(width)+'width'
    + str(width_epipolar)+
    'lowsR' +str(lowsR) + 
    method+'.png', Iout)
print('check ../result/')
