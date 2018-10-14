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

def keyPointMatching(imageKeyPoints, imageDescriptors, imgA, imgB):
    FLANN_INDEX_KDTREE = 0
    index_params = dict(algorithm=FLANN_INDEX_KDTREE, trees=5)
    search_params = dict(checks=50)   # or pass empty dictionary

    flann = cv.FlannBasedMatcher(index_params,search_params)
    # print('1 ', len(imageDescriptors[0]))
    # print('2 ', len(imageDescriptors[1]))
    # print (type(imageDescriptors[imgA]))
    matches = flann.knnMatch(imageDescriptors[imgA],
                             imageDescriptors[imgB], k=2)
                             # matches 2 nearest neigbours

    #using lows ratio test
    good = [[],[]]
    for i, (m, n) in enumerate(matches):
        if m.distance < 0.8 * n.distance: # if closest match is ratio 
                                          # closer than the second closest one,
                                          # then the match is good
            good[0].append(imageKeyPoints[imgA][m.queryIdx].pt)
            good[1].append(imageKeyPoints[imgB][m.trainIdx].pt)

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
    # print (len(good[0])) # length of good keyPointS
    return good

'''
n is the number of times to repeat ransac
t is the number of pixels tolerance allowed
T is the ratio of the for which we terminate early.
'''
def findHomoRanSac(n, r, list_kp, t, T):
    list_kp1 = list_kp[0]
    list_kp2 = list_kp[1]
    length = len(list_kp)
    print(len(list_kp[0])) # no of good matches
    print(list_kp1[0]) # random point

    for i in range(n):
        list_kp1r = []
        list_kp2r = []
        Sis = []
        Sisno = []
        # selecting ramdomly r points
        for i in range(r):
            key = np.random.choice(length)
            list_kp1r.append(list_kp1[key])
            list_kp2r.append(list_kp2[key])

        # find the homo, inlier set
        P = make_P(list_kp1r, list_kp2r)
        H, Si = findH_Si(P, list_kp, t)
        print ("length of the Si, line no 85:",len(Si))
        Sis.append(Si)
        Sisno.append(len(Si))

        # if majority return with new H
        if len(Si) >= T:
            P = make_P(Si[0], Si[1])
            H, Si = findH_Si(P, list_kp, t)
            return H

    Sisnoi = np.argmax(np.array(Sisno))[0] # taking the first index 
                                           # with global max cardinality
    Si = Sis[Sisnoi]
    P = make_P(Si[0], Si[1])
    H, Si = findH_Si(P, list_kp, t)
    return H


def findH_Si(P, list_kp, t):
    # do svd on P get perlimns H
    u, s, vh = np.linalg.svd(P, full_matrices=True)
    H = vh.T[:, -1] # taking the last singular vector
    H = np.reshape(H, (3,3))
    Si = [[],[]]

    # multiply all the matches and find if within tol
    initialPts = list_kp[0]
    finalPts = list_kp[1]
    for i in range(len(initialPts)):
        inPt = initialPts[i]
        fPt = finalPts[i]
        vc = np.array([[fPt[0]],[fPt[1]], [1]])
        vi = np.array([[inPt[0]],[inPt[1]], [1]])
        vf = np.matmul(H, vc)
        # print ('shape of vf', vf.shape)
        # making the last coordinate 1
        vf /= vf[2,0]

        # check if within some tolerance
        if np.linalg.norm(vf - vc) <= t:
            Si[0].append(inPt)
            Si[1].append(fPt)
    return (H, Si)

'''
I assume that i recieve 2 lists, in which i have k points
'''
def make_P(list_kp1, list_kp2):
    k = len(list_kp1)
    # print ('list_kp1', list_kp1)

    # making P matrix
    P = np.zeros((2*k, 9))
    for i in range(0,2*k,2):
        x = list_kp1[int(i/2)][0]
        x_ = list_kp2[int(i/2)][0]
        y = list_kp1[int(i/2)][1]
        y_ = list_kp2[int(i/2)][1]

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
    return P

def stitch(imgA, imgB, H, s, ratio=0.75, reporjThrest=4.0):
    result = cv.warpPerspective(imgA, H, (imgA.shape[1] + imgB.shape[1], imgA.shape[0]))
    result[0:imgB.shape[0], 0:imgB.shape[1]] = imgB
    return result

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

##########################################################
import cv2 as cv
import numpy as np
import sys

##########################################################
#Reading files
##########################################################
path = './Images_Asgnmt3_1/I1/'
imagesNames = ['a.jpg', 'b.jpg', 'c.jpg']#, 'd.jpg', 'e.jpg', 'f.jpg']
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
    goodMatchings[(imgA,imgB)]= keyPointMatching(imageKeyPoints, imageDescriptors, imgA, imgB)

##########################################################
#Finding H for each of the pairs of images
##########################################################
n = 500 # iterations
r = 4 # no of point to calc homo
t = 4 # pixel threashold
T = int(0.8*len(goodMatchings)) # majority threashold

# currently for single
for i in range(1):
    imgA = imagesNames[i]
    imgB = imagesNames[i+1]
    list_kp = goodMatchings[(imgA, imgB)]
    # print ("length of good keyPoints", len(list_kp[0]))
    H = findHomoRanSac(n, r, list_kp, t, T)
    print (H)
    sys.exit()


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