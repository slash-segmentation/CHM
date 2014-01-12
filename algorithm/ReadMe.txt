2013

There are two phases to the algorithm: Training (creating the model) and Testing (applying the model)
 - CHM_train does the training and it the training images and labels and outputs the model files.
 - CHM_test does the testing, it takes a images and the model files and generates the probability maps.
 
In the model folder, there are also temporary files in folders which can be deleted. The only thing that
one needs to keep is the many files like MODEL_level#_stage#.mat and param.mat.

There are parameters that can be adjusted. When training, you can select a number of levels and stages
of processing. The default is 2 stages and 4 levels. Increasing these values really time required to
perform training, but may increase quuaility of the model. When testing, you can set a block size and
overlap size. The block size should be the same size as the training images during training for
efficiency. Using an overlap is important so that there are less edges seen by the algorithm, however
it does increase the processing time. When using blocks, you should use TIFF images which can be nearly
twice as fast as PNGs and require less memory. When the TIFF image is a multiple of blocksize-2*overlapsize
in width and height it needs much less memory (not even enough to store the whole image).

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

If you use this code please cite the following paper:
M. Seyedhosseini, M. Sajjadi, and T. Tasdizen. Image segmentation with cascaded hierarchical models and logistic disjunctive normal networks. In ICCV 2013.

Please check here for the latest version:
http://www.sci.utah.edu/~mseyed/Mojtaba_Seyedhosseini/CHM.html



The code was written by Mojtaba Seyedhosseini and Mehdi Sajjadi.
Contact: mseyed@sci.utah.edu
