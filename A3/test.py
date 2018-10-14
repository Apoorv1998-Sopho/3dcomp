'''
Author: Apoorv Agnihotri

Code samples from:
https://kushalvyas.github.io/stitching.html
'''

import cv2 as cv
import numpy as np
import sys

path = './Images_Asgnmt3_1/I1/'
imagesNames = ['a.jpg', 'b.jpg'] #'c.jpg', 'd.jpg', 'e.jpg', 'f.jpg']
scale = (0.3, 0.3)
images = [] # will have 3 channel color imgs
imageNos = len(imagesNames)

for i in range(imageNos):
    print(path + imagesNames[i])
    temp = cv.imread(path + imagesNames[i])
    temp = cv.resize(temp, None, fx=scale[0], fy=scale[1], interpolation=cv.INTER_CUBIC)
    images.append(temp)
del temp

def keyPointMatching(imgA, imgB): # add an option to send a list of strings, where keypoints return
    imageKeyPoints = []
    imageDescriptors = []
    sift = cv.xfeatures2d.SIFT_create()
    imgA = cv.cvtColor(imgA, cv.COLOR_BGR2GRAY)
    keyPoints, descriptors = sift.detectAndCompute(imgA, None)
    imageDescriptors.append(descriptors)
    imageKeyPoints.append(keyPoints)
    imgB = cv.cvtColor(imgB, cv.COLOR_BGR2GRAY)
    keyPoints, descriptors = sift.detectAndCompute(imgB, None)
    imageDescriptors.append(descriptors)
    imageKeyPoints.append(keyPoints)

    FLANN_INDEX_KDTREE = 0
    index_params = dict(algorithm=FLANN_INDEX_KDTREE, trees=5)
    search_params = dict(checks=50)   # or pass empty dictionary

    # print('1 ', len(imageDescriptors[0]))
    # print('2 ', len(imageDescriptors[1]))
    # print (imageDescriptors[0].shape)
    flann = cv.FlannBasedMatcher(index_params,search_params)
    matches = flann.knnMatch(imageDescriptors[1], imageDescriptors[0], k=2)
    # print (type(matches), matches[0],'type of matches[0][0]'+str(type(matches[0][0])) , matches[0][0], matches[0][0].trainIdx)
    good = []
    for i, (m, n) in enumerate(matches):
        if m.distance < 0.7 * n.distance:
            good.append((m.trainIdx, m.queryIdx))

    #print(type(matches))
    # print('3 ', len(matches))
    matchesMask = [[0,0] for i in range(len(matches))]
    # print('matchesMask',len(matchesMask))
    draw_params = dict(matchColor=(0,255,0),
                      singlePointColor=(255,0,0),
                      matchesMask=matchesMask,
                      flags=0)
    img3 = cv.drawMatchesKnn(images[0], imageKeyPoints[0], images[1], imageKeyPoints[1], matches, None, **draw_params)

    cv.imshow("correspondences", img3)
    # print('length goog ', len(good))
    cv.waitKey()
    if len(good) > 4:
        pointsCurrent = imageKeyPoints[1]
        pointsPrevious = imageKeyPoints[0]

        matchedPointsCurrent = np.float32(
            [pointsCurrent[i].pt for (__, i) in good]
        )
        matchedPointsPrev = np.float32(
            [pointsPrevious[i].pt for (i, __) in good]
        )
        # print(matchedPointsCurrent)
        H, s = cv.findHomography(matchedPointsPrev, matchedPointsCurrent, cv.RANSAC, 4)
        print (H, len(s))
    return (H, s)
def stitch(imgA, imgB, H, s):
    result = cv.warpPerspective(imgA, H, (imgA.shape[1] + imgB.shape[1], imgA.shape[0]))
    result[0:imgB.shape[0], 0:imgB.shape[1]] = imgB
    return result


# the number of images that can matched to a single image
m = 3
r = 4 # number of points to find homography

H, s = keyPointMatching(images[0], images[1])
result = stitch(images[1], images[0], H, s)
cv.imshow("correspondences", result)
cv.waitKey()
