# Intro
Stereo image correspondences using Fundamental matrix.
Custom Functions are defined in `src/src.py`
Look at `src/` for all the code.
Run `src/demo.py` to get the resulting transformations in `result/`.

**Note**: To have a look at the precompiled results have a look in `saved_result/`.

### Requirements
```
python 3.7.0
opencv-python 3.4.2.16
opencv-contrib-python 3.4.2.16
numpy 1.15.2
matplotlib 3.0.0
```

### Details
* I allow setting a variable `width_epipolar` to make the line thicker on which we want to find the correspondences.
* I have used `lowe's ratio` in finding good keypoint matches as it increases the reliability of the key point matches between images.
* I take in a variable `method` that can be set to `SIFT`, `local` telling which descriptor to use. `local` here refers to the local 3x3 patch of RGB or LAB values.
* I have used **SIFT** key points to match the points on an epipolar line as they seemed to give the best results.

---

### Results

Reconstructed images,
![I1](https://i.imgur.com/beWHEzt.png) paramters used: discriptor:`'sift'` | width:`112` | lowsR:`0.70` | width_epipolar:`3`
![I2](https://i.imgur.com/KRfuPeY.png) paramters used: discriptor:`'sift'` | width:`112` | lowsR:`0.75` | width_epipolar:`3`
![I3](https://i.imgur.com/ufcSCLQ.png) paramters used: discriptor:`'sift'` | width:`200` | lowsR:`0.70` | width_epipolar:`3`

---
