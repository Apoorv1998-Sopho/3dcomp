##########################################################
#Reading files
##########################################################
path = './RGBD dataset/000001524/'
imagesNames = ['a.jpg', 'b.jpg', 'c.jpg', 'd.jpg']#, 'e.jpg', 'f.jpg']
depthNames = ['d'+img for img in imagesNames]
scale = (0.2, 0.2)
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
    temp = cv.resize(temp, None, fx=scale[0], 
                     fy=scale[1], interpolation=cv.INTER_CUBIC)
    images[img] = temp
for img in depthNames:
    print(path + img)
    temp = cv.imread(path + img)
    temp = cv.resize(temp, None, fx=scale[0], 
    	  			 fy=scale[1], interpolation=cv.INTER_CUBIC)
    dimages[img] = temp
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
goodMatchings={}
for i in range(imageNos-1):
    imgA = imagesNames[i]
    imgB = imagesNames[i+1]
    goodMatchings[(imgA,imgB)]= keyPointMatching(imageKeyPoints, imageDescriptors, imgA, imgB)
print('done keymatches')

##########################################################
#Quantize the depth image
##########################################################
'''
Quantized is a dict
Quantized['da.jpg']=[imgdpt1, imgdpt2, imgdpt3...]
'''
def Quantize(depthNames):

Quantized = Quantize(depthNames)