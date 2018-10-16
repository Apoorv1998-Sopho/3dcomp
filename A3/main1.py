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
built_in = False # to switch between built-in or custom
paths = ['I1', 'I4', 'I5']
for pathI in paths:
    path = './Images_Asgnmt3_1/' + pathI + '/'
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
    dirr = './result/Part1'+pathI
    lowsR = 0.85 # low's ratio
    goodMatchings={}
    for i in range(imageNos-1):
        imgA = imagesNames[i]
        imgB = imagesNames[i+1]
        goodMatchings[(imgA,imgB)]= keyPointMatching(images, 
                                  imageKeyPoints, imageDescriptors, 
                                  imgA, imgB, dirr, lowsR)
    print('done keymatches')

    ##########################################################
    #Finding H for each of the pairs of images
    ##########################################################
    n = 1000 # iterations
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
        if (not built_in):
            H, S = findHomoRanSac(n, r, list_kp, t, Tratio)
        else:
            H, S = cv.findHomography(list_kp[1], list_kp[0], cv.RANSAC, 4)
        Hs.append(H)
        Ss.append(S)
    print('done homographies')
    print('HomoGraphies', Hs)

    ##########################################################
    #Wrapping the images together using H
    ##########################################################
    print('finding realtive homographies')
    Hss = {int(imageNos/2): np.eye(3)}
    for i in range(int(imageNos/2), imageNos-1):
        Hss[i+1] = np.matmul(np.linalg.inv(Hs[i]), Hss[i])
    for i in range(int(imageNos/2)-1, -1, -1):
        Hss[i] = np.matmul(Hs[i], Hss[i+1])
    print('done realtive homographies')

    if not built_in:
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
        cv.imwrite("./result/"+pathI+"unblended.jpg", out)

        # create canvas
        factor = [int(imageNos*3), int(imageNos*5)] # dy,dx
        offset = [[3000,1000]] # x,y
        canvas = createCanvas(images[imagesNames[0]], factor)

        print('stiching blended')
        #drawing blended
        weightDic = {}
        print('drawing ', end= '')
        for i in range(0, imageNos):
            print(imagesNames[i], end=' ')
            drawOnCanvas(canvas, images[imagesNames[i]], Hss[i], offset, abs(int(imageNos/2)-i)+1, weightDic)

        # divide by weights at each pixel
        divideWeight(canvas, weightDic)
        canvas = canvas.astype(np.uint8)

        # stripping
        print ('stripping')
        true_points = np.argwhere(canvas)
        top_left = true_points.min(axis=0)
        bottom_right = true_points.max(axis=0)
        out = canvas[top_left[0]:bottom_right[0]+1,  # plus 1 because slice isn't
                     top_left[1]:bottom_right[1]+1]  # inclusive
        print('done stripping')

        # spitting
        cv.imwrite("./result/"+pathI+"blended.jpg", out)


    else: # built_in
        pass
        

print('check "./result"')
sys.exit()