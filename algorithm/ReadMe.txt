There are two phases to the algorithm: Training (creating the model) and
Testing (applying the model)
 - CHM_train does the training:
   it takes the training images and labels and outputs the model files.
 - CHM_test does the testing:
   it takes a images and the model files and generates the probability maps.

They can be run from within MATLAB or from the shell itself. See the usage in
each for more detailed information.

The quality of training labels are critical to good success of the testing
phase. The training data should be at least 500x500x20, but above 1000x1000x50
gets very memory and time prohibitive.

In the model folder, there are also temporary files in folders which can be
deleted. The only thing that one needs to keep is the many files like
MODEL_level#_stage#.mat and param.mat.

%================== Filters
The following codes are integrated in this package:
1 - HoG features -->http://www.mathworks.com/matlabcentral/fileexchange/33863-histograms-of-oriented-gradients
2 - SIFT features -->https://github.com/nazikus/dalcim
3 - Haar like features -->http://www.iri.upc.edu/people/mvillami/code.html
You can add/remove the features in the Filterbank.m function, e.g., Radon-like
features for membrane detection.



OS: Tested on 64-bit Linux and precompiled binary versions are included in the
package. Other systems should be supported but have not been tested.

Note: This code was written from scratch for the purpose of sharing.
The results are not exactly the same as the reported numbers in the paper
(fortunately they are slightly better).
The parameters are tuned for the membrane detection problem.
For horse segmentation you should use larger windows.

If you use this code please cite the following paper:
M. Seyedhosseini, M. Sajjadi, and T. Tasdizen. Image segmentation with cascaded hierarchical models and logistic disjunctive normal networks. In ICCV 2013.

Please check here for the latest version:
http://www.sci.utah.edu/~mseyed/Mojtaba_Seyedhosseini/CHM.html



The code was written by Mojtaba Seyedhosseini and Mehdi Sajjadi.
Contact: mseyed@sci.utah.edu
