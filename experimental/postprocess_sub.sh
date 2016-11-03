#!/bin/bash
:<<doc
This script runs randomise using the sign flip for any single on the Language MVPA dataset. 
First section determines OS to properly allocate filepaths

First this checks if the randomise input already exists, then if not it reregisters/standardizes the raw images
Arguments:
1: directory of files (e.g. Maps/Encoding)
2: MASK name (in fmri/data/standard)
3: MODEL name
4: CHANCE level (in case of MVPA, 0 for encoding or RSA)

example usage:
sh postprocess_mfx.sh Maps/Encoding grayMatter cross_anim_L2P_ccsl 0

formula:
1/2 * ln (1+r)/(1-r) * sqrt(N(N2-3))

doc

RETURNHERE=${PWD}
CHANCE=0
while [[ $# -gt 1 ]]
do
key="$1"

case $key in
    -n|--model)
    MODEL="$2"
    shift # past argument
    ;;
    -p|--path)
    RESPATH="$2"
    shift # past argument
    ;;
    -m|--mask)
    MASK="$2"
    shift # past argument
    ;;
    -s|--sub)
    SUB="$2"
    shift # past argument
    ;;
	-c|--chance)
    CHANCE="$2"
    shift # past argument
    ;;
    *)
            # unknown option
    ;;
esac
shift # past argument or value
done



# check PLATFORM
PLATFORM='unknown'
if [[ "$OSTYPE" == "linux-gnu" ]]; then
   PLATFORM='linux'
   ROOTDIR=/mnt/d
   HOMEDIR=$ROOTDIR
elif [[ "$OSTYPE" == "darwin"* ]]; then
   PLATFORM='mac'
   ROOTDIR=/Volumes/fmri
   HOMEDIR=/Users/njchiang
fi

echo "PLATFORM  = ${PLATFORM}" 
echo "ROOTDIR  = ${ROOTDIR}"
echo "HOMEDIR  = ${HOMEDIR}"
echo "FILE PATH  = ${RESPATH}"
echo "MODEL     = ${MODEL}"
echo "MASK    = ${MASK}"
echo "CHANCE    = ${CHANCE}"
echo "SUB     = ${SUB}"
projectDir=${ROOTDIR}/fmri/LanguageMVPA
desDir=${HOMEDIR}/GitHub/LanguageMVPA/multivariate/bash
refImage=${projectDir}/MNI152_T1_3mm_brain.nii.gz
refMask=${projectDir}/fnirt/MNI152_T1_3mm_brain_mask.nii.gz
targetDir=${projectDir}/${RESPATH}

cd ${targetDir}
totalT=`fslval ${SUB}_${MASK}_${MODEL} dim4`
nR=$((${totalT} / 2))
N1=`fslval ${projectDir}/data/${SUB}/func/${SUB}_Run1.nii.gz dim4` # assume they roughly have same #TRs
N=$(echo "sqrt( ( ${N1}-3 ) * ( ${nR} - 1 ) )" | bc -l)
echo ${SUB}

fslroi ${SUB}_${MASK}_${MODEL} lang 0 $nR
fslroi ${SUB}_${MASK}_${MODEL} pic $nR -1
for f in lang pic
do
	fslmaths $f -add 1 num.nii.gz
	fslmaths $f -mul -1 -add 1 den.nii.gz
	fslmaths num -div den -log -mul .5 -Tmean -mul ${N} ${SUB}_${MASK}_${MODEL}_${f}_z
	fslmaths ${SUB}_${MASK}_${MODEL}_${f}_z -ztop -mul -1 -add 1 ${SUB}_${MASK}_${MODEL}_${f}_p
	fdr -i ${SUB}_${MASK}_${MODEL}_${f}_p --oneminusp -m ${projectDir}/data/${SUB}/masks/${SUB}_${MASK} \
		-q 0.05 -a ${SUB}_${MASK}_${MODEL}_${f}_fdrp

	regMat=${projectDir}/data/${SUB}/reg/${SUB}_example_func2standard.mat
	flirt -in ${SUB}_${MASK}_${MODEL}_${f}_fdrp -out ${SUB}_${MASK}_${MODEL}_${f}_fdrp_std -ref ${refImage} \
		-applyxfm -init ${regMat}
	rm ${f}.nii.gz num.nii.gz den.nii.gz
done
	
cd ${RETURNHERE}