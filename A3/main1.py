'''
Author: Apoorv Agnihotri

Code samples from:
https://kushalvyas.github.io/stitching.html
'''
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
    return np.zeros((height*factor, width*factor, chnl), dtype=np.uint8)

'''
brief:
    Draws an image on canvas after transformation with H
params:
    canvas- global canvas we want to draw on.
    img- the image that needs to be transformed
    H- Homography matrix
    offset- The offset list [x, y]
    fill- the number of pixels surrounding need to be filled too
'''
def drawOnCanvas(canvas, img, H, offset, fill):
    height, width, chnl = img.shape
    for i in range(height):
        for j in range(width):
            vctr = np.array([[i,j,1]]).T #col vctr
            vctr2 = np.matmul(H, vctr)
            vctr2 /= vctr2[2,0] #last coordinate to 1
            pt = vctr2[:2,:] #getting rid of last coordinate
            pt += np.array(offset).T

            #drawing with interpolation
            y = int(pt[0,0])
            x = int(pt[1,0])
            canvas[y:y+fill, x:x+fill] = np.full((fill, fill, 3), img[i,j], dtype=np.uint8)
            # if i == 0 and j == 0:
            #     print (np.full((fill, fill, 3), img[i,j]))
            #     print (np.full((fill, fill, 3), img[i,j]).shape)
    # print('returned')
    return 

##########################################################
#Reading files
##########################################################
path = './Images_Asgnmt3_1/I1/'
imagesNames = ['a.jpg', 'b.jpg', 'c.jpg']#, 'd.jpg', 'e.jpg', 'f.jpg']
scale = (0.1, 0.1)
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
n = 1000 # iterations
r = 4 # no of point to calc homo
t = 4 # pixel threashold
Tratio = 0.95 # majority threashold

# currently for single
Hs = []
Ss = []
for i in range(imageNos-1):
    imgA = imagesNames[i]
    imgB = imagesNames[i+1]
    list_kp = goodMatchings[(imgA, imgB)]
    H, S = findHomoRanSac(n, r, list_kp, t, Tratio)
    Hs.append(H)
    Ss.append(S)

##########################################################
#Wrapping the images together using H
##########################################################
factor = imageNos+1
offset = [[400,400]]
canvas = createCanvas(images[imagesNames[0]], factor)
print('image.shape', images[imagesNames[0]].shape)
print('canvas.shape', canvas.shape)

# middle term is identity mat
Hss = {int(imageNos/2): np.eye(3)}

#forward terms filled
for i in range(int(imageNos/2), imageNos-1):
    Hss[i+1] = np.matmul(Hs[i], Hss[i])
#backward terms filled
for i in range(int(imageNos/2)-1, -1, -1):
    Hss[i] = np.matmul(np.linalg.inv(Hs[i]), Hss[i+1])

# print ('Hss0', Hss[2])
# print ('Hsinv', Hs[1])

# drawing
for i in range(0, imageNos):
    drawOnCanvas(canvas, images[imagesNames[i]], Hss[i], offset, abs(int(imageNos/2)-i)+1)

# print (canvas)
plt.imshow(canvas)
plt.show()
sys.exit()
