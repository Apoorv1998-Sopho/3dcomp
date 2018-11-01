import numpy as np
import cv2 as cv
import sys
import math

# images is a dict of numpy arrays, containing images
def keyPoints(images):
    # for every image find keypoint discriptors
    sift = cv.xfeatures2d.SIFT_create()
    imageKeyPoints = {}
    imageDescriptors = {}
    for i in range(len(images)):
        img = images[i]
        # finding dicriptors
        img = cv.cvtColor(img, cv.COLOR_BGR2GRAY)
        keyPoints, descriptors = sift.detectAndCompute(img, None)
        imageDescriptors[i] = descriptors
        imageKeyPoints[i] = keyPoints
    # compare each image with every other
    return (imageKeyPoints, imageDescriptors)

def keyPointMatching(images, imageKeyPoints, imageDescriptors, imgA, imgB, lowsR):
    FLANN_INDEX_KDTREE = 0
    index_params = dict(algorithm=FLANN_INDEX_KDTREE, trees=5)
    search_params = dict(checks=50)   # or pass empty dictionary
    flann = cv.FlannBasedMatcher(index_params,search_params)
    matches = flann.knnMatch(imageDescriptors[imgA],
                             imageDescriptors[imgB], k=2)
                             # matches 2 nearest neigbours
    #using lows ratio test
    good = [[],[]]
    for i, (m, n) in enumerate(matches):
        if m.distance < lowsR * n.distance: # if closest match is ratio 
                                          # closer than the second closest one,
                                          # then the match is good
            good[0].append(imageKeyPoints[imgA][m.queryIdx].pt)
            good[1].append(imageKeyPoints[imgB][m.trainIdx].pt)
    return good

def drawPoint(canvas, RGBvalue, cord):
    x = cord[0]
    y = cord[1]
    canvas[y, x] = RGBvalue
    return

def createCanvas(img, factor=(3,3)):
    height, width, chnl = img.shape
    return np.zeros((height*factor[1], width*factor[0], chnl), dtype=np.uint16)

def strip(canvas2):
    true_points = np.argwhere(canvas2)
    top_left = true_points.min(axis=0)
    bottom_right = true_points.max(axis=0)
    out = canvas2[top_left[0]:bottom_right[0]+1,  # plus 1 because slice isn't
                 top_left[1]:bottom_right[1]+1]  # inclusive
    return out

'''
imgwithlines - image on which we draw the epilines
for the points in img2 lines - corresponding epilines
https://docs.opencv.org/trunk/da/de9/tutorial_py_epipolar_geometry.html
'''
def drawlines(imgwithlines,img2,lines,pts1,pts2):
    r,c, chl = imgwithlines.shape
    for r,pt1,pt2 in zip(lines,pts1,pts2):
        color = tuple(np.random.randint(0,255,3).tolist())
        x0,y0 = map(int, [0, -r[2]/r[1] ])
        x1,y1 = map(int, [c, -(r[2]+r[0]*c)/r[1] ])
        imgwithlines = cv.line(imgwithlines, (x0,y0), (x1,y1), color,1)
        imgwithlines = cv.circle(imgwithlines,tuple(pt1),5,color,-1)
        img2 = cv.circle(img2,tuple(pt2),5,color,-1)
    return imgwithlines,img2

def findEpilines(Points, F):
    temp = cv.computeCorrespondEpilines(Points, 1, F)
    return np.squeeze(temp, axis=1)

def findValidPoints(epiline, dim, width=2, padd=1):
    h, w = dim
    a, b, c = epiline
    validPoints = []
    # check every point to be within width
    for x in range(w):
        if (b != 0.):
            y_range = range(int((-a/b)*x-width-c), int((-a/b)*x+width-c) + 1)
            for y in y_range:
                # making sure index error doesn't occur in future
                if not(x < padd or y < padd or y >= h - (padd+2) or x >= w - (padd+2)):
                    validPoints.append((x, y))
        else: # ignore this case
            pass
    return validPoints

def findCustomDiscriptor(image, Points, channel='RGB'):
    discriptors = {}
    dsize = 18
    if channel == 'RGB':
        dsize = 27

    for pt in Points:
        x, y = pt
        if channel == 'RGB':
            temp = image[y-1:y+2,x-1:x+2]
        elif channel == 'LAB':
            temp = image[y-1:y+2,x-1:x+2, 1:] # ignoring Luminence
        temp = temp.flatten(order='F')
        # print('temp', temp)
        if temp.size == dsize:
            # assert(temp.size == 9)
            discriptors[pt] = np.array(temp)
        else:
            continue
        # print('Discriptor', image[y-1:y+2,x-1:x+2])
        # sys.exit()

    if len(discriptors.keys()) == 0:
        return None
    return discriptors
