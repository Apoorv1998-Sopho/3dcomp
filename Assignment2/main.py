######################################################
############ Scale-space decomposition ###############
######################################################

import old_helper as hp
from scipy.ndimage import gaussian_filter
import cv2 as cv
import numpy as np
import math

# container is a dictionary in which we have tuples as keys, (octave, scale) and numpy arrays as vals
Container = {}
Dogs = {}
sigma = 1
gaussian_dim = 3
scales = 5
octaves = 4
k = 2**.5
result = "./result/"

# read img and convert to grey (resized)
img = hp.readImg_Grey_Resize(file='imgs/5_1.jpg', scale=.8)

# guassian blur
def blur(img, dim, newsigma):
    gauss = hp.gaussian_nor(dim, newsigma)
    return hp.covolv(img, gauss)

def absolDiff(img2, img1):
    rows, cols = img2.shape
    diff = np.zeros((rows, cols))    
    for i in range(rows):
        for k in range(cols):
            # making as type int because of underflow and realted problems
            diff[i,k]=abs(int(img2[i][k]) - int(img1[i][k]))
    return diff

# Q1
# generating scale space
print ("Starting blurring.")
for i in range(octaves):
    # For different octaves
    scale = 1./(2**i)
    newBaseImg = cv.resize(img, None, fx = scale, fy = scale, interpolation = cv.INTER_CUBIC)
    for j in range(scales):
        #find the related sigam
        newsigma = sigma*(k**((i*2)+j))
        Container[(i,j)]=blur(newBaseImg.astype(np.float32),3, newsigma)
print('Blurring done')
#imageshowing
for i in range(octaves):
    for j in range(scales):
        cv.namedWindow('gaussed', cv.WINDOW_NORMAL)
        cv.imwrite(result+"Blurred_Octave_"+str(i)+"_scale_"+str(j)+"_.jpg", Container[(i,j)].astype(np.uint8))
        cv.imshow('gaussed',Container[(i,j)].astype(np.uint8))
        cv.waitKey(0)
        cv.destroyAllWindows()

# generating dogs
print ("Starting difference of Gaussians")
for i in range(octaves):
    for j in range(scales-1):
        Dogs[(i,j)]= Container[i,j] - Container[i,j+1]
print('Dogs calculated')
#imageshowing
for i in range(octaves):
    for j in range(scales-1):
        cv.namedWindow('dogs', cv.WINDOW_NORMAL)
        cv.imwrite(result+"Dogs_Octave_"+str(i)+"_scale_"+str(j)+"_.jpg", Dogs[(i,j)])
        cv.imshow('dogs',Dogs[(i,j)])
        cv.waitKey(0)
        cv.destroyAllWindows()

######################################################
############### Key Point Detection ##################
######################################################

# returns a list of keypoints inside a dictionry with key (octave, scale-2)
def Kye(Dogs):
    Keypoints ={}
    global octaves
    global scales
    print ('Starting Keypoints')
    for i in range(octaves):
        # since dog is 1 less than scales and 
        # key points would further be 2 less than dogs
        for j in range(1, scales-2):
            up_scale = Dogs[(i,j+1)]
            mid_scale = Dogs[(i,j)]
            low_scale = Dogs[(i,j-1)]

            rows, cols = up_scale.shape
            for l in range(1, rows-1):
                for k in range(1, cols-1):
                    # checking if a keypt or not
                    flag = 0
                    keyPt = mid_scale[l][k]
                    mx = up_scale[l-1][k-1]
                    mi = up_scale[l-1][k-1]
                    # looking up and down
                    for m in range(l-1, l+2):
                        for n in range(k-1, k+2):
                            ptup = up_scale[m,n]
                            ptlow = low_scale[m,n]
                            mx = max(ptup, ptlow, mx)
                            mi = min(ptup, ptlow, mi)
                            if (keyPt < mx and keyPt > mi):
                                flag = 1
                                break
                        if flag == 1:
                                break

                    #looking in mid
                    for m in range(l-1, l+2):
                        for n in range(k-1, k+2):
                            # if the keypt condidate itself, ignore
                            if not(m == l and n == k):
                                ptmid = mid_scale[m,n]
                                mx = max(ptmid, mx)
                                mi = min(ptmid, mi)
                                if (keyPt < mx and keyPt > mi):
                                    flag = 1
                                    break
                        if flag == 1:
                                break

                    # if keypt is not good, escape
                    if flag == 0:
                        try:
                            Keypoints[i,j-1].append((l,k))
                        except KeyError:
                            Keypoints[i,j-1] = [(l,k)]

    print("Done keypoints")
    return Keypoints

Keypoints = Kye(Dogs)

#Show Initial KeyPoints 
for i in range(octaves):
    scale = 1./(2**i)
    draw = cv.resize(img, None, fx = scale, fy = scale, interpolation = cv.INTER_CUBIC)
    for j in range(scales-3):
        for key in Keypoints.keys():
            for ke in Keypoints[key]:
                cv.circle(draw, ke, 5, (0,0,0), 1)
    cv.namedWindow('Keypts', cv.WINDOW_NORMAL)
    cv.imwrite(result+"Keypts_Octave_"+str(i)+"_scale_"+str(j)+"_.jpg", draw)
    cv.imshow('Keypts',draw)
    cv.waitKey(0)
    cv.destroyAllWindows()


######################################################
############# Orientation Assignment  ################
######################################################

def mAssign(Container):
    Orientation={}
    for i in range (octaves):
        for j in range(2, scales-1):
            scale = hp.padd(Container[(i,j)], 1)
            rows, cols = scale.shape
            r = np.zeros((rows, cols))
            for l in range(1, rows-1):
                for k in range(1, cols-1):
                    dx = scale[l, k+1]- scale[l, k-1]
                    dy = scale[l+1, k] - scale[l-1, k]
                    r[l,k] = (dx**2 + dy**2)**0.5
            Orientation[(i,j-2)] = r[1:-1,1:-1]
    return Orientation
    
def thetaAssign(Container):
    global octaves
    global scales
    Theta={}
    for i in range (octaves):
        for j in range(2, scales-1):
            scale = hp.padd(Container[(i,j)], 1)
            rows, cols = scale.shape
            r = np.zeros((rows, cols))
            for l in range(1, rows-1):
                for k in range(1, cols-1):
                    dx = scale[l, k+1]- scale[l, k-1]
                    dy = scale[l+1, k] - scale[l-1, k]
                    t = math.atan2(dy, dx)*180/np.pi
                    if t < 0:
                        r[l,k] = t + 360
                    else:
                        r[l,k] = t
            Theta[(i,j-2)] = r[1:-1,1:-1]
    return Theta

print ("Starting Orientations")
mMatrix = mAssign(Container)
thetaMatrix = thetaAssign(Container)

def Orientation(mMatrix, thetaMatrix, Keypoints):
    global octaves
    global scales
    global sigma
    KeyPtOrientation = {}
    for i in range(octaves):
        for j in range (scales-3):
            m_oc = mMatrix[(i,j)]
            theta_oc = thetaMatrix[(i,j)]
            keypts_oc = Keypoints[(i,j)]
            newsigma = k**(2*i + j)*sigma
            rows, cols = theta_oc.shape
            # for every keypoint dicovered
            for ke in keypts_oc:
                # if the keypt is at boundary, no use.
                if not(ke[0]>=8 and ke[0]<cols - 9 and ke[1]>=8 and ke[1]<rows - 9):
                    continue
                
                mTheta = bts(ke, m_oc, theta_oc, newsigma)
                if not mTheta:
                    continue
                    
                # if mTheta is defined
                try:
                    KeyPtOrientation[(i,j)][ke]=mTheta
                except:
                    KeyPtOrientation[(i,j)]={ke:mTheta}
    return KeyPtOrientation


def bts(ke, m_oc, theta_oc, newsigma):
    m_slice = m_oc[ke[0]-8:ke[0]+8, ke[1]-8:ke[1]+8]
    rows , cols = m_slice.shape
    if rows < 16 or cols < 16:
        return None
    theta_slice = theta_oc[ke[0]-8:ke[0]+8, ke[1]-8:ke[1]+8]
    
    g = hp.gaussian_nor(16, newsigma)
    mg = np.dot(m_slice, g)
    # getting the histogram
    buckets={}
    for i in range(16):
        for j in range(16):
            try:
                buckets[int(m_slice[i,j]//10)].append((ke[0]+i-8,ke[1]+j-8))
            except KeyError:
                buckets[int(m_slice[i,j]//10)]= [(ke[0]+i-8,ke[1]+j-8)]
    
    '''
    getting the max sum of m values for all the
    elements within a bucket.
    If we have multiple buckets with same 
    max values, i take the convention of 
    selecting the first one
    '''
    max_m = 0
    bucket_i = None
    for i in range(36):
        sums = [0]*36
        try:
            m = buckets[i]
        except:
            m = []
        for pt in m:
            sums[i] += m_oc[pt[0], pt[1]]
        if max_m < sums[i]:
            max_m = sums[i]
            bucket_i = i
    return (max_m, bucket_i*10 + 5)

Orientations = Orientation(mMatrix, thetaMatrix, Keypoints)
print ("Done Orientations")
for o in range(octaves):
    for j in range(1, scales-2):
        ori = Container[(o,j)].astype(np.uint8)
        try:
            kkk = Orientations[(o,j-1)]
        except:
            kkk = {}
        for point in kkk.keys():
            x1,y1 = point
            m,th = kkk[point]
            x2 = x1 + (m%20)*math.cos(th*np.pi/180)
            y2 = y1 + (m%20)*math.sin(th*np.pi/180)
            cv.arrowedLine(ori, (int(x1),int(y1)), (int(x2),int(y2)), (250,250,0), 1)
        cv.imwrite(result+"Oriented_Octave_"+str(o)+"_scale_"+str(j)+"_.jpg", ori)
        cv.imshow("result",ori)
        cv.waitKey(0)
        cv.destroyAllWindows()

######################################################
#################### Descriptor  #####################
######################################################
def giveDiscript(mMatrix, thetaMatrix, Keypoints):
    global octaves
    global scales
    KeyPtOrientation = {}
    for i in range(octaves):
        for j in range(scales-3):
            m_oc = mMatrix[(i,j)]
            theta_oc = thetaMatrix[(i,j)]
            try:
                keypts_oc=Keypoints[(i,j)]
            except:
                keypts_oc=np.array([])
            newsigma = 8
            rows, cols = theta_oc.shape
            # for every keypoint dicovered
            for ke in keypts_oc:
                # if the keypt is at boundary, no use.
                if not(ke[0]>=8 and ke[0]<cols - 9 and ke[1]>=8 and ke[1]<rows - 9):
                    continue
                
                g = hp.gaussian_nor(16, newsigma)
                m_slice = np.dot(m_oc[ke[0]-8:ke[0]+8,ke[1]-8:ke[1]+8], g)
                for p in range(-2, 2):
                    for q in range(-2, 2):
            
                        mTheta = bts2(p, q, ke, theta_oc, m_slice)
                        if not mTheta:
                            mTheta=[0]*8

                        # if mTheta is defined
                        try:
                            KeyPtOrientation[(i,j)][ke].extend(mTheta)
                        except:
                            KeyPtOrientation[(i,j)]={ke:mTheta}

                try:
                    discript = KeyPtOrientation[(i,j)][ke]
                except:
                    discript = [1]*128
                discript=np.array(discript)
                # Normalizing and clipping Descriptor Values
                discript = discript/np.linalg.norm(discript)
                discript = np.clip(discript,0,0.2)
                discript = discript/np.linalg.norm(discript)
                KeyPtOrientation[(i,j)][ke]=discript
    return KeyPtOrientation


def bts2(p,q,ke, theta_oc, m_slice):
    a = p*4
    b = q*4
    rows , cols = m_slice.shape
    if rows < 4 or cols < 4:
        return None
    theta_slice = theta_oc[ke[0]+a:ke[0]+a+4, ke[1]+b:ke[1]+b+4]
    rows , cols = theta_slice.shape
    if rows < 4 or cols < 4:
        return None
    
    # getting the histogram
    msum=[0]*8
    for i in range(4):
        for j in range(4):
            msum[int(theta_slice[i,j]//45)]+= m_slice[i,j]
    return msum
print ("Starting Discriptors")
discriptr = giveDiscript(mMatrix, thetaMatrix, Keypoints)
print ("Done Discriptors | Lookat ./results/dicriptor")
np.save("./results/dicriptor", discriptr)
# the file contains the dicriptors in a dictionary with key as (octave, scale)
input()
