# Intro
This folder deals with panorama stitching. I have implemeneted custom homography calculation function `findHomoRanSac` which take in a number of parameters to modify the RANSAC algorithm. I have also implemented a blending mechenism built into the function `drawOnCanvas` which does partial blending and the rest of the blending is done by `divideWeight`.

Run `main1.py` to get the resulting panoramas in `result/`.
Run `main2.py` to get the resulting warped RGBD reference images in `result/`.

### Requirements
```
python 3.7.0
opencv-python 3.4.2.16
opencv-contrib-python 3.4.2.16
numpy
imutils
```

# Part 1

One can set the value of `built_in` variable to `True` to use inbuilt function instead of the custom functions.
### Details
* I have used linear blending along the horizontal axis only, as the images that have been provided are horizontally stitched.
* I have used the threshold value in the RANSAC to be equal to 2 pixels.
* I have used `lowe's ratio` in finding good keypoint matches as it increase the realibility of the keypoint matches between images.



# Part 2

One can set the value of `usual_warping` variable to `True` to use usual warping without the quantizations w.r.t. depths. Both the cases use the custom functions as the correctness of the custom functions could be checked by setting `built_in` to `True` in the first part.
### Details
* I have divided the reference image in only `5` `dlevels` as the number of good keypoint matchings was drastically decreasing with `dlevels` set to `10`.
