#! /bin/bash

##########

# CHM parameters
classdir=/data/aperez/CHM_classifiers/nucleus/ZT04/edge/Nstage2_Nlevel2
rootdir=/data/aperez
testdir=ZT04/input_iso_full/XY/histeq/PNG
outdir=ZT04/runCHM_nucleus/XY
Nstage=2
Nlevel=2
TileX=8
TileY=8
Overlap=200

# SGE parameters
maxvmem=10
email=a3perez@ucsd.edu

##########

DIR_test=${rootdir}/${testdir}
DIR_out=${rootdir}/${outdir}

# Make symbolic links to the CHM code
mkdir CHM
for file in `find /data/aperez/CHM_edge -maxdepth 1 -type f`; do
    ln -s ${file} CHM/`basename ${file}`
done
mkdir CHM/FilterStuff
for file in /data/aperez/CHM_edge/FilterStuff/*; do
    ln -s ${file} CHM/FilterStuff/`basename ${file}`
done

# Make symbolic links to classifier files
mkdir ${DIR_out}
for file in `find ${classdir} -maxdepth 1 -type f`; do
    ln -s ${file} ${DIR_out}/`basename ${file}`
done
for dir in `find ${classdir}/* -type d`; do
    mkdir ${DIR_out}/`basename ${dir}`
    for file in ${dir}/*; do
        ln -s ${file} ${DIR_out}/`basename ${dir}`/`basename ${file}`
    done
done

N_testImgs=`ls ${DIR_test} | wc -l`

echo "Test Directory:   ${DIR_test}"
echo "Output Directory: ${DIR_out}"
echo "Number of Images: ${N_testImgs}"

cd CHM

C0=1
for ((i=1;i<=${N_testImgs};i+=1)); do
    N_dir=`printf %05d $i`
    img_in=`ls ${DIR_test}/*.png | sed -n ''${i}','${i}'p'`
    qsub -v img_in=${img_in},testfolder=${DIR_test},testout=${DIR_out},Nstage=${Nstage},Nlevel=${Nlevel},TileX=${TileX},TileY=${TileY},Overlap=${Overlap} -l h_vmem=${maxvmem}G -M ${email} -N CHM${N_dir} CHM_qsub_testTile.sh
    C0=$((CF_round+1))
done

cd ..
