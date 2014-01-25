##### Background Cropping #####
$ ./GetForegroundArea.sh example.tif
1 115 2891 3001

# The next 3 functions take those four values
$ ./FillBackgroundWithReflection.sh example.tif 1 115 2891 3001 example-refl.tif
(a bunch of MATLAB output, generates same-sized image with background replaced with relfection)

$ ./FillBackgroundWith0.sh example-refl.tif 1 115 2891 3001 example-0.tif
(a bunch of MATLAB output, generates same-sized image with background replaced with 0)

$ ./CropBackground.sh example.tif 1 115 2891 3001 example-cropped.tif
(a bunch of MATLAB output, generates smaller image without background)

# The next 2 are the same, the first two values are from the original command
# The next two numbers are the ORIGINAL height and width or an image with that height and width
$ ./AddBackground.sh example.tif 1 115 3001 3001 example-cropped.tif
(a bunch of MATLAB output, generates larger image with background of 0)

$ ./AddBackground.sh example.tif 1 115 example.tif example-cropped.tif
(a bunch of MATLAB output, generates larger image with background of 0)


##### Histogram Equalization #####
$ ./GetHistogram.sh example.tif hist.txt
$ ./HistogramEqualization.sh example.tif hist.txt example-histeq.tif

$ ./GetHistogram.sh example.tif | ./HistogramEqualization.sh example.tif - example-histeq.tif

# You can also generate the histogram from multiple files like the following (more examples in the usage)
$ ./GetHistogram.sh *.tif hist.txt
