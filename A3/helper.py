import cv2 as cv
import numpy as np
import sys
import matplotlib.pyplot as plt

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

def keyPointMatching(images, imageKeyPoints, imageDescriptors, imgA, imgB, dirr):
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
    return good

'''
brief:
    find the homography matrix, such that
    list_kp[0], transforms to list_kp[1]
params:
    n is the number of times to repeat ransac
    r is the number of points to approximate H(min 4)
    t is the number of pixels tolerance allowed
    list_kp is [list1, list2] where list1 and list2 contain
        the matched keypoints, on the same index, list1 is [(x1,y1),..]
    Tratio is the ratio of the for which we terminate early.
'''
def findHomoRanSac(n, r, list_kp, t, Tratio):
    list_kp1 = list_kp[0]
    list_kp2 = list_kp[1]
    T = int(Tratio * len(list_kp2))

    Sis = []
    Sisno = []
    for i in range(n):
        list_kp1r = []
        list_kp2r = []
        
        # selecting ramdomly r points
        for i in range(r):
            key = np.random.choice(len(list_kp2))
            list_kp1r.append(list_kp1[key])
            list_kp2r.append(list_kp2[key])
        # print (list_kp1r, list_kp2r)

        # find the homo, inlier set
        P = make_P(list_kp1r, list_kp2r)
        # print(P)
        H, Si = findH_Si(P, list_kp, t)
        Sis.append(Si)
        # print ('Si:',Si)
        Sisno.append(len(Si[0]))

        # if majority return with new H
        if len(Si[0]) >= T:
            P = make_P(Si[0], Si[1])
            # print('threashold crossed')
            # print('P output as:', P)
            H, Si = findH_Si(P, list_kp, t)
            # print ('si',Si)
            return (H / H[2,2], Si)

    # print('Sisno',Sisno)
    Sisnoi = np.argmax(np.array(Sisno)) # taking the first index 
                                        # with global max cardinality
    # print('i', Sisnoi)
    # print('maxii', Sisno[Sisnoi])
    Si = Sis[Sisnoi]
    P = make_P(Si[0], Si[1])
    H, Si = findH_Si(P, list_kp, t)
    # print ('si',Si)
    return (H / H[2,2], Si)

def findH_Si(P, list_kp, t):
    # do svd on P get perlimns H
    u, s, vh = np.linalg.svd(P, full_matrices=True)
    H = vh[-1].reshape(3,3) # taking the last singular vector
    Si = [[],[]]

    # multiply all the matches and find if within tol
    initialPts = list_kp[0]
    finalPts = list_kp[1]
    # print('no of keypts', len(initialPts))
    for i in range(len(initialPts)):
        inPt = initialPts[i]
        fPt = finalPts[i]
        vi = np.array([[inPt[0]],[inPt[1]], [1]])
        vf = np.matmul(H, vi)        
        vf /= vf[2,0] # making the last coordinate 1

        # check if within some tolerance
        vc = np.array([[fPt[0]],[fPt[1]], [1]])
        if np.linalg.norm(vf - vc) <= t:
            Si[0].append(inPt)
            Si[1].append(fPt)
    return (H, Si)
'''
I assume that i recieve 2 lists
'''
def make_P(list_kp1, list_kp2):
    k = len(list_kp1)
    # making P matrix
    P = np.zeros((2*k, 9))
    for i in range(0,2*k,2):
        x = list_kp1[int(i/2)][0]
        x_ = list_kp2[int(i/2)][0]
        y = list_kp1[int(i/2)][1]
        y_ = list_kp2[int(i/2)][1]
        P[i+0,:] = [x, y, 1, 0, 0, 0, -x*x_, -y*x_, -x_]
        P[i+1,:] = [0, 0, 0, x, y, 1, -x*y_, -y*y_, -y_]
    return P

'''
Makes a canvas for the panorama
'''
def createCanvas(img, factor):
    height, width, chnl = img.shape
    return np.zeros((height*factor[0], width*factor[1], chnl), dtype=np.uint16)

'''
brief:
    Draws an image on canvas after transformation with H
params:
    canvas- global canvas we want to draw on.
    img- the image that needs to be transformed
    H- Homography matrix
    offset- The offset list [x, y]
    fill- the number of pixels surrounding need to be filled too
    @param weightDic 
'''
def drawOnCanvas(canvas, img, H, offset, fill, weightDic=None):
    height, width, chnl = img.shape
    for i in range(height):
        for j in range(width):
            if weightDic != None:
                # finding the weight not exactly zero at edges
                if j > int(width/2):
                    weight = (width - j+1) / ((float(width)/2))
                else:
                    weight = (j+1)/((float(width)/2))

            vctr = np.array([[j,i,1]]).T #col vctr
            vctr2 = np.matmul(H, vctr)
            vctr2 /= vctr2[2,0] #last coordinate to 1
            pt = vctr2[:2,:] #getting rid of last coordinate
            pt += np.array(offset).T

            #drawing with interpolation
            x = int(pt[0,0])
            y = int(pt[1,0])
            # if weightDic not provided don't do blending
            if weightDic != None: # if we have been provided weightDic
                c = weight * np.full((fill, fill, 3), img[i,j])

            else: # weightDic == None
                c = np.full((fill, fill, 3), img[i,j])

            try:
                if weightDic != None:
                    canvas[y:y+fill, x:x+fill] += c.astype(np.uint16)# to reduce tearing
                else:
                    canvas[y:y+fill, x:x+fill] = c.astype(np.uint16)# to reduce tearing
                # adding weight as weightDic[col, row] if weightDic provided
                if weightDic != None:
                    for l in range(fill):
                        for k in range(fill):
                            try:
                                weightDic[(x+l,y+k)] += weight
                            except KeyError:
                                weightDic[(x+l,y+k)] = weight
            except ValueError:
                print('not able to print, x,y', x, y)
    return

def divideWeight(canvas, weightDic):
    for key in weightDic.keys():
        x, y = key
        weight = weightDic[(x, y)]
        canvas[y, x] = (canvas[y, x]/weight).astype(np.uint16)

# PART 2
#########################################################################################

def Quantize(dimages, depthNames, dlevels=5):
    # find the max depth
    for i in range(len(dimages)):
        if i == 0:
            max_depth = np.max(dimages[depthNames[i]])
        max_depth = max(max_depth, np.max(dimages[depthNames[i]]))
    # print ('max_depth',max_depth) = 255
    depth_quantum = int(max_depth/dlevels)
    print(depth_quantum)
    for i in range(len(dimages)):
        dimages[depthNames[i]] = (dimages[depthNames[i]]/dlevels).astype(np.uint8) #only dlevels
        dimages[depthNames[i]] *= depth_quantum
    return dimages