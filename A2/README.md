# Intro
Assignment uses jupyter notebook with `python=3.7.0` and `opencv-python=3.4.2.17`. `main.py` uses functions present `old_helper.py` that contains the implementations of some custom functions.

### Usage:
The python file `main.py` contains the Questions with the corresponding parts. Run it and the image results would be stored in the folder `results` automatically.

### Assumptions:
* I have used number of octaves to be equal to `4`.
* I have used number of scales to be equal to `5`.
* I have used the initial sigma = `1`.
* I have used the factor `k` for calculating the different sigmas to be = $$\sqrt[2]{2}$$.

### Results:
For the results look at the folder named `results` to have a look at the files generated. The file names are such that the correspond to the octave and the scale they are from.

Also the `dicriptors` I got are saved as a numpy array file in `results`.
