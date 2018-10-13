'''
Author: Apoorv Agnihotri

Code samples from:
https://kushalvyas.github.io/stitching.html
'''
# images is a list of numpy arrays, containing images
def keyPoints(images, imagesNames): # add an option to send a list of strings, where keypoints return
    # for every image find keypoint discriptors
    sift = cv.xfeatures2d.SIFT_create()
    imageKeyPoints = {}
    imageDescriptors = {}
    for i in imagesNames:
        img = images[i]

        # finding dicriptors
        img = cv.cvtColor(img, cv.COLOR_BGR2GRAY)
        keyPoints, descriptors = sift.detectAndCompute(img, None)
        imageDescriptors[i] = descriptors
        imageKeyPoints[i] = keyPoints

    # compare each image with every other
    return (imageKeyPoints, imageDescriptors)

def keyPointMatching(imageDescriptors, imgA, imgB):
    FLANN_INDEX_KDTREE = 0
    index_params = dict(algorithm=FLANN_INDEX_KDTREE, trees=5)
    search_params = dict(checks=50)   # or pass empty dictionary

    flann = cv.FlannBasedMatcher(index_params,search_params)
    # print('1 ', len(imageDescriptors[0]))
    # print('2 ', len(imageDescriptors[1]))
    # print (type(imageDescriptors[imgA]))
    matches = flann.knnMatch(imageDescriptors[imgB],
                             imageDescriptors[imgA], k=2)
                             # matches 2 nearest neigbours

    #using lows ratio test
    good = []
    for i, (m, n) in enumerate(matches):
        if m.distance < 0.7 * n.distance: # if closest match is ratio 
                                          # closer than the second closest one,
                                          # then the match is good
            good.append((m.trainIdx, m.queryIdx))

    #print(type(matches))
    matchesMask = [[0,0] for i in range(len(matches))]
    draw_params = dict(matchColor=(0,255,0),
                      singlePointColor=(255,0,0),
                      matchesMask=matchesMask,
                      flags=0)
    img3 = cv.drawMatchesKnn(images[imgA], imageKeyPoints[imgA], images[imgB],
                             imageKeyPoints[imgB], matches, None, **draw_params)

    cv.imshow("correspondences", img3)
    # print('length of all matches ', len(matches))
    # print('length good matches ', len(good))
    cv.waitKey()
    return good
    # if len(good) > 4:
    #     pointsCurrent = imageKeyPoints[1]
    #     pointsPrevious = imageKeyPoints[0]

    #     matchedPointsCurrent = np.float32(
    #         [pointsCurrent[i].pt for (__, i) in good]
    #     )
    #     matchedPointsPrev = np.float32(
    #         [pointsPrevious[i].pt for (i, __) in good]
    #     )
    #     #print(len(matchedPointsCurrent))
    #     H, s = cv.findHomography(matchedPointsCurrent, matchedPointsPrev, cv.RANSAC, 4)
    # return (H, s)


def stitch(imgA, imgB, H, s, ratio=0.75, reporjThrest=4.0):
    result = cv.warpPerspective(imgA, H, (imgA.shape[1] + imgB.shape[1], imgA.shape[0]))
    result[0:imgB.shape[0], 0:imgB.shape[1]] = imgB
    return result



##########################################################
import cv2 as cv
import numpy as np
import sys


##########################################################
#Reading files
##########################################################
path = './Images_Asgnmt3_1/I1/'
imagesNames = ['a.jpg', 'b.jpg', 'c.jpg', 'd.jpg', 'e.jpg', 'f.jpg']
scale = (0.3, 0.3)
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
imageKeyPoints, imageDescriptors = keyPoints(images, imagesNames)
# retured dictionaries with keys as imageNames


##########################################################
#Finding matchings for best 'm' matching images for each image
##########################################################
goodMatchings={}
for i in range(imageNos-1):
    imgA = imagesNames[i]
    imgB = imagesNames[i+1]
    goodMatchings[(imgA,imgB)]= keyPointMatching(imageDescriptors, imgA, imgB)

for ke in goodMatchings.keys():
    findHomoRanSac(goodMatchings, ke)

sys.exit()
##########################################################
#Finding H for each of the pairs of images
##########################################################


##########################################################
#Wrapping the images together using H
##########################################################


##########################################################
#Finding matchings for best 'm' matching images for each image
##########################################################

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

# n is the number of times to repeat ransac
# t is the number of pixels tolerance allowed
# T is the ratio of the for which we terminate early.
def findHomoRanSac(n, m, list_kp1, list_kp2, matches, t, T):
    for i in range(n):
        P = make_P(list_kp1, list_kp2)
        H, Si = findH_Si(P, matches, t, T)

'''


'''
I assume that i recieve 2 lists, in which i have k points

def make_P(list_kp1, list_kp2):
    k = len(list_kp1)

    # making P matrix
    P = np.zeros((2k, 9))
    for i in range(0,2*k,2):
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
def findH_Si(P, matches, t, T):
    # do svd on P get perlimns H

    # multiply all the matches and find if within tol

    # increase counter if within t

    # if counter crosses T
        # recalculate H for the Si set
        # return H and count

    # else
        # recalculate H for the Si set
        # return H and count corresp to biggest set
'''