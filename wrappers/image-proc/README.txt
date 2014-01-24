$ ./GetForegroundArea.sh 0114.tif
1 115 2891 3001

$ ./FillBackgroundWithReflection.sh 0114.tif 1 115 2891 3001 0114-refl.tif
(a bunch of MATLAB output, generates same-sized image with background replaced with relfection)

$ ./FillBackgroundWith0.sh 0114-refl.tif 1 115 2891 3001 0114-0.tif
(a bunch of MATLAB output, generates same-sized image with background replaced with 0)

$ ./CropBackground.sh 0114.tif 1 115 2891 3001 0114-cropped.tif
(a bunch of MATLAB output, generates smaller image without background)
