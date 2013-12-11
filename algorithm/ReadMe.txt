2013

You should start with TrainScript.m
1 - There is a main script which is responsible for training and testing the CHM (TrainScript.m).
It only passes some parameters and computes a bunch of variables for preallocation.
2 - trainCHM is the main function that extracts features and trains the classifier, i.e., LDNN, 
and finally computes the results for the training images.
3 - testCHM is the main function that runs the trained model on new input images.
4 - You should put your training images in the "trainImages" folder.
5 - You should put your training labels in the "trainLabels" folder.
6 - You should put your testing images in the "testImages" folder.
7 - I have put one sample in the folders.

%================== Filters
I integrated the following codes in this package:
1 - HoG features -->http://www.mathworks.com/matlabcentral/fileexchange/33863-histograms-of-oriented-gradients
2 - SIFT features -->https://github.com/nazikus/dalcim
3 - Haar like features -->http://www.iri.upc.edu/people/mvillami/code.html
You can add/remove the features in the Filterbank.m function, e.g., Radon-like features
for membrane detection.


OS: I have only tested this code on 64-bit linux and precompiled binary versions
are included in the package.

Note: This code was written from scratch for the purpose of sharing.
The results are not exactly the same as the reported numbers in the paper
(fortunately they are slightly better).
The parameters are tuned for the membrane detection problem.
For horse segmentation you should use larger windows.

The code was written by Mojtaba Seyedhosseini and Mehdi Sajjadi.
Contact: mseyed@sci.utah.edu