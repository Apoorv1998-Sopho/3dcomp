'''
Author: Apoorv Agnihotri

Code samples from:
https://kushalvyas.github.io/stitching.html
'''

import cv2 as cv
import numpy as np
import sys

path = './Images_Asgnmt3_1/I1/'
imagesNames = ['a', 'b', 'c', 'd', 'e', 'f']
scale = (0.3, 0.3)
images = [] # will have 3 channel color imgs
imageNos = len(imagesNames)

for i in range(imageNos):
    print(path + imagesNames[i])
    temp = cv.imread(path + imagesNames[i])
    temp = cv.resize(temp, None, fx=scale[0], fy=scale[1], interpolation=cv.INTER_CUBIC)
    images.append(temp)
del temp

imageKeyPoints = []
imageDescriptors = []

sift = cv.xfeatures2d.SIFT_create()
for i in range(imageNos):

# images is a list of numpy arrays, containing images
def keyPoints(images): # add an option to send a list of strings, where keypoints return
    # for every image find keypoint discriptors    
    imageKeyPoints = {}
    imageDescriptors = {}
    for img in imagesNames:
        img = cv.cvtColor(img, cv.COLOR_BGR2GRAY)
        keyPoints, descriptors = sift.detectAndCompute(img, None)
        imageDescriptors[img] = descriptors
        imageKeyPoints[img] = keyPoints

    # compare each image with every other
    pass


def keyPointMatching(imgA, imgB)
    imageKeyPoints = []
    imageDescriptors = []
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

    flann = cv.FlannBasedMatcher(index_params,search_params)
    # print('1 ', len(imageDescriptors[0]))
    # print('2 ', len(imageDescriptors[1]))
    print (type(imageDescriptors[0]))
    matches = flann.knnMatch(imageDescriptors[1], imageDescriptors[0], k=2)
    print (type(matches), matches[0],'type of matches[0][0]'+str(type(matches[0][0])) , matches[0][0], matches[0][0].trainIdx)
    good = []
    for i, (m, n) in enumerate(matches):
        if m.distance < 0.7 * n.distance:
            good.append((m.trainIdx, m.queryIdx))

    #print(type(matches))
    print('3 ', len(matches))
    matchesMask = [[0,0] for i in range(len(matches))]
    print(matchesMask)
    draw_params = dict(matchColor=(0,255,0),
                      singlePointColor=(255,0,0),
                      matchesMask=matchesMask,
                      flags=0)
    img3 = cv.drawMatchesKnn(images[0], imageKeyPoints[0], images[1], imageKeyPoints[1], matches, None, **draw_params)

    cv.imshow("correspondences", img3)
    print('length goog ', len(good))
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
        #print(len(matchedPointsCurrent))
        H, s = cv.findHomography(matchedPointsCurrent, matchedPointsPrev, cv.RANSAC, 4)
    return (H, s)


def stitch(imgA, imgB, H, s, ratio=0.75, reporjThrest=4.0):
    result = cv.warpPerspective(imgA, H, (imgA.shape[1] + imgB.shape[1], imgA.shape[0]))
    result[0:imgB.shape[0], 0:imgB.shape[1]] = imgB
    return result


H, s = keyPointMatching(images[0], images[1])
result = stitch(images[1], images[0], H, s)
cv.imshow("correspondences", result)
cv.waitKey()
H, s = keyPointMatching(result, images[2])
result = stitch(images[2], result, H, s)
#result = cv.resize(result, (960, 540))
cv.imshow("correspondences", result)
cv.waitKey()


# How to find the coordinates
'''
# Initialize lists
list_kp1 = []
list_kp2 = []

# For each match...
for mat in matches:

    # Get the matching keypoints for each of the images
    img1_idx = mat.queryIdx
    img2_idx = mat.trainIdx

    # x - columns
    # y - rows
    # Get the coordinates
    (x1,y1) = kp1[img1_idx].pt
    (x2,y2) = kp2[img2_idx].pt

    # Append to each list
    list_kp1.append((x1, y1))
    list_kp2.append((x2, y2))
'''

'''
I assume that i recieve 2 lists, in which i have k points

def make_P(list_kp1, list_kp2):
    k = len(list_kp1)

    # making P matrix
    P = np.zeros((2k, 9))
    for i in range(0,2*k,2)
        pi = np.zeros((2,9))
        x = list_kp1[i/2][0]
        x_ = list_kp2[i/2][0]
        y = list_kp1[i/2][1]
        y_ = list_kp2[i/2][1]

        P[i+0,0] = -x
        P[i+0,1] = -y
        P[i+0,2] = -1
        P[i+0,6] = x*x_
        P[i+0,7] = y*x_
        P[i+0,8] = x_
        P[i+1,3] = -x
        P[i+1,4] = -y
        P[i+1,5] = -1
        P[i+1,6] = x*y_
        P[i+1,7] = y*y_
        P[i+1,8] = y_
'''

'''
def findH_Si(P, matches, tol):
    # do svd on P

    # get H

    # multiply all the matches and find if within tol

    # increase counter

    # return H and count
'''