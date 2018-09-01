# Intro
Assignment uses jupyter notebook with `python=3.7.0` and `opencv-python=3.4.2.17`.
`Assignment 1.ipynb` uses functions present `helper.py` that contains the 
implementations of all my custom functions.


### Usage:
The notebook `Assignment 1.ipynb` contains all the Questions with all the parts.

One can change the scale with which the images are scaled before convolving. This can
done using the paramter `scale` in the function `readImg_Grey_Resize()`

##### Part 1
`helper.py` contains the implementation of `prettyPrint` for the printing in required
format.

It's better to scale the images, else it will take a really long time because of
unoptimized implementation of `covolv()`
