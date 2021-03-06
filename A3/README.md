# Intro
This folder deals with panorama stitching. I have implemented custom homography calculation function `findHomoRanSac` which take in a number of parameters to modify the RANSAC algorithm. I have also implemented a blending mechenism built into the function `drawOnCanvas` which does partial blending and the rest of the blending is done by `divideWeight`.

Custom Functions are defined in `src/helper.py`
Look at `src/` for all the code.

**Note**: To have a look at the precompiled results have a look in `saved_result/`.

PART1 - Run `src/main1.py` to get the resulting panoramas in `result/`.
PART2 - Run `src/main2.py` to get the resulting warped RGBD reference images in `result/`.

### Requirements
```
python 3.7.0
opencv-python 3.4.2.16
opencv-contrib-python 3.4.2.16
numpy 1.15.2
```

# Part 1

One can set the value of `built_in` variable to `True` to use inbuilt function instead of the custom functions.
### Details
* I have used linear blending along the horizontal axis only, as the images that have been provided are horizontally stitched.
* I have used the threshold value in the RANSAC to be equal to 2 pixels.
* I have used `lowe's ratio = 0.85` in finding good keypoint matches as it increase the realibility of the keypoint matches between images.
* When we set the `built_in` flag to `True`, the in_built function `cv.findHomography()` is used to calculate `H`. The stitching (without blending) in this case is done by my custom function `drawOnCanvas()` (as it wasn't clear).

### Values
* The values of Homography Matrix are printed when we run the python script `main1.py` (both in inbuilt and custom versions).

----

# Part 2

One can set the value of `warp_usual` variable to `True` to use usual warping without the quantizations w.r.t. depths. Both the cases use the custom functions as the correctness of the custom functions could be checked by setting `built_in` to `True` in the first part.
### Details
* I have divided the reference image in only `5 dlevels` as the number of good keypoint matchings was drastically decreasing with `dlevels` set to `10`.
* Whenever there are less than `15` inliers when dealing with different depths, I use interpolation for determining `H` as the `H` that we would get in such a case would be unrealiable due to few keypoint matchings.
* I have used `lowe's ratio = 0.75` in finding good keypoint matches.

### Values
* The values of Homography Matrix are printed when we run the python script `main2.py` (both in inbuilt and custom versions).

---
