import sys
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
        print(P)
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
            return H / H[2,2]

    # print('Sisno',Sisno)
    Sisnoi = np.argmax(np.array(Sisno)) # taking the first index 
                                        # with global max cardinality
    # print('i', Sisnoi)
    # print('maxii', Sisno[Sisnoi])
    Si = Sis[Sisnoi]
    P = make_P(Si[0], Si[1])
    H, Si = findH_Si(P, list_kp, t)
    # print ('si',Si)
    return H / H[2,2]

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
I assume that i recieve 2 lists, in which i have k points
'''
def make_P(list_kp1, list_kp2):
    k = len(list_kp1)
    # print('k value, should be 4 usually', k)
    # print ('list_kp1', list_kp1)

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
import numpy as np
import cv2 as cv

n = 5
r = 4
t = 2
Tratio = 0.9	

keypts1=[(2,6),(124, -126), (324,54), (234,135), (234,634)]
keypts2=[(2,4),(1, -126), (34,54), (23,35), (29,34)]
list_kp = [keypts1, keypts2]
H, s = cv.findHomography(np.array(keypts2), np.array(keypts1), cv.RANSAC, 4)
print (H)
print (s)
print (findHomoRanSac(n, r, list_kp, t, Tratio))

'''
We want H = np.array([[1,2,1],
					  [0,1,-3],
					  [2,3,1]])
some vecotrs
2,6,1 -> 15, 3, 23
124, -126, 1 -> -127, -129, -129
12, -26, 1 ->  -39, -29, -53
12, -6, 1 -> 1, -9, 7
8, -6, 1 -> -4, -9, -1

'''