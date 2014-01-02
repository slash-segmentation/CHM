#! /bin/bash

#$ -V
#$ -cwd
#$ -j y
#$ -S /bin/bash
#$ -m eas

Dir_IM=/home/aperez/usr/local/bin
base=`basename ${img_in}`
base=${base%.*}

testTile=${testfolder}/${base}
mkdir ${testTile} 

${Dir_IM}/convert ${img_in} -crop ${TileX}x${TileY}+${Overlap}+${Overlap}@\! +repage +adjoin ${testTile}/${base}_tile_%04d.png

testF="'${testTile}'"
testO="'${testout}'"
matlab -nodisplay -singleCompThread -r 'TrainScript_test('${testF},${testO},${Nstage},${Nlevel}'); quit'

rm -rf ${testTile} #Remove input tiles

#####
# Stitching
#####

pwd
cd ../output_testImages

DX=${TileX} #Number of tiles in X
DY=${TileY} #Number of tiles in Y
OX=${Overlap} #Overlap in X in pixels
OY=${Overlap} #Overlap in Y in pixels

if (( ${OX} == 0 & ${OY} == 0 )); then #Perform simple stitching if no overlap was specified
    ${Dir_IM}/montage ${base}_tile*.png -mode concatenate -tile ${DX}x${DY} ${base}.png
    ${Dir_IM}/convert ${base}.png -equalize ${base}.png
    exit 0
fi

# First, the row is stitched together in a west-to-east (left-to-right) manner. Rows are then stitched together in a
# north-to-south (top-to-bottom) manner. Regions of overlap are handeld by taking the maximum pixel value in the 
# region
C=1
for ((j=1;j<=${DY};j+=1)); do #Loop over rows
    imgFirst=`ls ${base}_tile*.png | sed -n ''${C}','${C}'p'` #Name of first image in the row
    imgHeight=`${Dir_IM}/identify -format "%[fx:h]" ${imgFirst}` #Height of row
    for ((i=1;i<=${DX};i+=1)); do #Loop over each image within the row
        imgL=`ls ${base}_tile*.png | sed -n ''${C}','${C}'p'` #Name of first image
        imgLWidth=`${Dir_IM}/identify -format "%[fx:w]" ${imgL}`
        if (( i == 1 )); then #If first image in the row, no overlap is needed in the western direction
            imgR=`ls ${base}_tile*.png | sed -n ''$((C+1))','$((C+1))'p'` #Name of second image
	    imgRWidth=`${Dir_IM}/identify -format "%[fx:w]" ${imgR}`
	    ${Dir_IM}/convert ${imgL} -gravity west -crop -${OX}-0 ${base}_tempL.png #Crop non-overlap region of image1
	    ${Dir_IM}/convert ${imgL} -gravity east -crop ${OX}x${imgHeight}-0-0 ${base}_tempML.png #Crop eastern overlap region of image1
	    ${Dir_IM}/convert ${imgR} -gravity west -crop ${OX}x${imgHeight}-0-0 ${base}_tempMR.png #Crop western overlap region of image2
            ${Dir_IM}/convert ${base}_tempML.png ${base}_tempMR.png -compose lighten -composite ${base}_tempM.png #Take max of overlap regions
	    ${Dir_IM}/convert +append ${base}_tempL.png ${base}_tempM.png ${base}_out_temp.png #Append
       elif (( i > 1 & i <= $((DX-1)) )); then #All middle images in the row
            imgR=`ls ${base}_tile*.png | sed -n ''$((C+1))','$((C+1))'p'`
            imgRWidth=`${Dir_IM}/identify -format "%[fx:w]" ${imgR}`
	    ${Dir_IM}/convert ${imgL} -gravity center -crop $((imgLWidth-2*OX))x${imgHeight}-0-0 ${base}_tempL.png #Crop non-overlap region of image1
	    ${Dir_IM}/convert ${imgL} -gravity east -crop ${OX}x${imgHeight}-0-0 ${base}_tempML.png #Crop eastern overlap region of image1
	    ${Dir_IM}/convert ${imgR} -gravity west -crop ${OX}x${imgHeight}-0-0 ${base}_tempMR.png #Crop western overlap region of image2
	    ${Dir_IM}/convert ${base}_tempML.png ${base}_tempMR.png -compose lighten -composite ${base}_tempM.png #Take max of overlap regions
	    ${Dir_IM}/convert +append ${base}_out_temp.png ${base}_tempL.png ${base}_tempM.png ${base}_out_temp.png #Append
        else #Last image in the row, overlap has already been analyzed
            ${Dir_IM}/convert ${imgL} -gravity east -crop -${OX}-0 ${base}_tempL.png #Crop non-overlap region
	    ${Dir_IM}/convert +append ${base}_out_temp.png ${base}_tempL.png ${base}_out_temp.png #Append
        fi
        rm -rf ${base}_temp*.png 
    C=$((C+1))
    done
    if (( j == 1 )); then #If first row, rename to out.png
        imgWidthTotal=`${Dir_IM}/identify -format "%[fx:w]" ${base}_out_temp.png`
	mv ${base}_out_temp.png ${base}_out.png
    else
        imgHeightOut=`${Dir_IM}/identify -format "%[fx:h]" ${base}_out.png` #Get height of total output to this point
	${Dir_IM}/convert ${base}_out.png -gravity south -crop ${imgWidthTotal}x${OY}-0-0 ${base}_tempMU.png 
	${Dir_IM}/convert ${base}_out.png -gravity north -crop ${imgWidthTotal}x$((imgHeightOut-OY))-0-0 ${base}_tempU.png
	${Dir_IM}/convert ${base}_out_temp.png -gravity north -crop ${imgWidthTotal}x${OY}-0-0 ${base}_tempMD.png
	${Dir_IM}/convert ${base}_out_temp.png -gravity south -crop ${imgWidthTotal}x$((imgHeight-OY))-0-0 ${base}_tempD.png
	${Dir_IM}/convert ${base}_tempMU.png ${base}_tempMD.png -compose lighten -composite ${base}_tempM.png
	${Dir_IM}/convert -append ${base}_tempU.png ${base}_tempM.png ${base}_tempD.png ${base}_out.png
    fi
    rm -rf ${base}_temp*.png ${base}_out_temp*.png
done

rm -rf ${base}_tile*.png #Remove output tiles
