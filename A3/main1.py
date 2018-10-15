'''
Author: Apoorv Agnihotri

Code samples from:
https://kushalvyas.github.io/stitching.html
'''
import cv2 as cv
import numpy as np
import sys
import matplotlib.pyplot as plt
from helper import *
##########################################################
#Reading files
##########################################################
path = './Images_Asgnmt3_1/I1/'
imagesNames = ['a.jpg', 'b.jpg', 'c.jpg', 'd.jpg']#, 'e.jpg', 'f.jpg']
scale = (0.2, 0.2)
images = {} # will have 3 channel color imgs
imageNos = len(imagesNames)
m = 3
k = 4

##########################################################
#Rescaling
##########################################################
for img in imagesNames:
    print(path + img)
    temp = cv.imread(path + img)
    temp = cv.resize(temp, None, fx=scale[0], fy=scale[1], interpolation=cv.INTER_CUBIC)
    images[img] = temp
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
goodMatchings={}
for i in range(imageNos-1):
    imgA = imagesNames[i]
    imgB = imagesNames[i+1]
    goodMatchings[(imgA,imgB)]= keyPointMatching(images, 
                              imageKeyPoints, imageDescriptors, 
                              imgA, imgB)
print('done keymatches')

##########################################################
#Finding H for each of the pairs of images
##########################################################
n = 3000 # iterations
r = 4 # no of point to calc homo
t = 2 # pixel threashold
Tratio = 0.95 # majority threashold

# currently for single
Hs = []
Ss = []
print('finding homographies')
for i in range(imageNos -1):
    imgA = imagesNames[i]
    imgB = imagesNames[i+1]
    list_kp = goodMatchings[(imgA, imgB)]
    H, S = findHomoRanSac(n, r, list_kp, t, Tratio)
    Hs.append(H)
    Ss.append(S)
print('done homographies')

##########################################################
#Wrapping the images together using H
##########################################################
# first factor multiplie height
factor = [int(imageNos*1.5), int(imageNos*3)]
# x,y
offset = [[3000,1000]]
canvas = createCanvas(images[imagesNames[0]], factor)
print('image.shape', images[imagesNames[0]].shape)
print('canvas.shape', canvas.shape)

print('finding realtive homographies')
# middle term is identity mat
Hss = {int(imageNos/2): np.eye(3)}
#forward terms filled
for i in range(int(imageNos/2), imageNos-1):
    Hss[i+1] = np.matmul(np.linalg.inv(Hs[i]), Hss[i])
#backward terms filled
for i in range(int(imageNos/2)-1, -1, -1):
    Hss[i] = np.matmul(Hs[i], Hss[i+1])
print('done realtive homographies')

# print ('Hss', Hss)
# print ('Hs', Hs)

# drawing
for i in range(0, imageNos):
    print('drawing', imagesNames[i])
    drawOnCanvas(canvas, images[imagesNames[i]], Hss[i], offset, abs(int(imageNos/2)-i)+1)
    print('drawn', imagesNames[i])

# print (canvas)
# cv.imshow("Stitched Panorama", canvas)

print ('stripping')
# argwhere will give you the coordinates of every non-zero point
true_points = np.argwhere(canvas)
# take the smallest points and use them as the top left of your crop
top_left = true_points.min(axis=0)
# take the largest points and use them as the bottom right of your crop
bottom_right = true_points.max(axis=0)
out = canvas[top_left[0]:bottom_right[0]+1,  # plus 1 because slice isn't
          top_left[1]:bottom_right[1]+1]  # inclusive
print('done stripping')

cv.imwrite("./result/stitched.jpg", out)
sys.exit()

'''
imageBox(images):
h = 0
w
for img in images:
    h,w,c = img.shape
'''
