import cv2 as cv
import numpy as np
import sys
from helper import *

paths = ['A', 'B', 'C']
for pathI in paths:
    ##########################################################
    #Reading files
    ##########################################################
    # setting some variables
    warp_usual = False # Turn True to do warping without depths
    dlevels = 5 
    path = '../data/RGBD/' + pathI + '/'
    imagesNames = ['a.jpg', 'b.jpg']
    depthNames = ['d'+img for img in imagesNames]
    scale = (1, 1)
    images = {} # will have 3 channel color imgs
    dimages = {} # will have 1 chnl imgs
    imageNos = len(imagesNames)
    m = 3
    k = 4
    lowsR = .75 # low's ratio, taking big, cz not many matchings
    n = 1000
    r = 4
    t = 2
    Tratio = 0.95

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
    imgA = imagesNames[0]
    imgB = imagesNames[1]
    keyPointMatchings = keyPointMatching(images, imageKeyPoints,
                            imageDescriptors,
                            imgA, imgB, lowsR)
    print('done keymatches')

    ##########################################################
    #Quantize the depth image
    ##########################################################

    dimages, depth_quantum = Quantize(dimages,depthNames, dlevels)
    keyPtsDivided = keypt_divide_depth(dimages, depthNames, keyPointMatchings, dlevels, depth_quantum)

    ##########################################################
    #Find HomoGraphies
    ##########################################################

    if not warp_usual:
        # Dict containing HomoGraphies
        print('Finding HomoGraphies')
        Hs = {}
        for i in range(dlevels):
            try:
                list_kp = keyPtsDivided[i]
                try:
                    H, S = findHomoRanSac(n, r, list_kp, t, Tratio)
                    if len(S[0]) < 15: # if the number of inliers small, H unrealible
                        H=Hs[i-1] # interpolate with prev H
                except ValueError: # when not enough points
                    H=Hs[i-1]
            except KeyError: # not even one keypoint matching
                H = Hs[i-1]
            
            Hs[i] = H
        print('Done HomoGraphies')
        print('HomoGraphies:', Hs)

    else: # warp_usual
        H, S = findHomoRanSac(n, r, keyPointMatchings, t, Tratio)
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

    if not warp_usual:
        # printing ref img
        print('drawing dlevel: ', end= '')
        # drawOnCanvas(canvas2, images[imagesNames[1]], np.eye(3), offset, fill=1, weightDic=None)
        for i in range(dlevels):
            print(i, end=' ')
            drawOnCanvas(canvas2, regions[i], Hs[i], offset, 
                         fill=3, weightDic=None, blackPixelPrint=False)
        canvas2 = canvas2.astype(np.uint8)

    else : # usual warp
        drawOnCanvas(canvas2, images[imagesNames[0]], H, offset, fill=3)
        canvas2 = canvas2.astype(np.uint8)

    # stripping
    print ('stripping')
    out = strip(canvas2)
    print('done stripping')

    # spitting
    if not warp_usual:
        cv.imwrite("../result/"+pathI+"warped.jpg", out)
        print('check "../result/'+pathI+'warped.jpg"')

    else: # warp_usual
        cv.imwrite("../result/"+pathI+"usual_warped.jpg", out)
        print('check "../result/'+pathI+'usual_warped.jpg"')

sys.exit()
