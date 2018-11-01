'''
Author: Apoorv Agnihotri

Code samples from:
https://kushalvyas.github.io/stitching.html
'''
import cv2 as cv
import numpy as np
import sys
from src import *
import matplotlib.pyplot as plt
##########################################################
#Reading files
##########################################################
paths = ['I1', 'I2']
for pathI in paths:
    path = '../data/' + pathI + '/'
    imagesNames = ['a.jpg', 'b.jpg']
    scale = (0.1, 0.1)
    images = {} # will have 3 channel color imgs
    imageNos = len(imagesNames)
    imgB = 1
    width_epipolar = 3 # width of the epiline to chck correspondences

    ##########################################################
    #Rescaling
    ##########################################################
    images = {}
    for i in range(len(imagesNames)):
        img = imagesNames[i]
        print(path + img)
        temp = cv.imread(path + img)
        temp = cv.resize(temp, None, fx=scale[0], fy=scale[1], interpolation=cv.INTER_CUBIC)
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
    lowsR = 0.85 # low's ratio
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
    n = 1000 # iterations
    r = 8 # no of point to calc fundamental matrix
    t = 2 # pixel threashold
    Tratio = 0.95 # majority threashold
    print('finding Fundamentals')
    for i in range(imageNos -1):
        imgA = i
        list_kp = goodMatchings[(imgA, imgB)]
        # finding the fundamental matrices, using ransac
        F, S = cv.findFundamentalMat(np.array(list_kp[0]), np.array(list_kp[1]))
    print('done Fundamentals')
    print('Fundamental Matrix:', F)

    ##########################################################
    # Change to LAB if wanted
    ##########################################################
    channel = 'RGB'

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
    print('Find the point corresp to every point in I1, please wait..')
    h, w, chl = I2.shape
    Iout = np.zeros((h, w, chl))
    for i in range(len(epilines)):
        epiline = epilines[i]
        vPts = findValidPoints(epiline, (h, w), width_epipolar)
        if len(vPts) == 0: # no valid point
            continue

        # returns discriptors in rows
        discriptors = findCustomDiscriptor(I2, vPts, channel=channel)
        if discriptors == None:
        	continue
        if discriptors != None:
            ptI1 = listOfPoints[i]
            I1ptDiscriptor = findCustomDiscriptor(I1, [ptI1], channel=channel) # can return empty array
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
    np.save('out', Iout)
    plt.imshow('out', Iout)
    plt.imwrite('out.jpg', Iout)
    plt.show()
    sys.exit()
