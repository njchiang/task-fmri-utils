#!/bin/bash
:<<doc
This script runs randomise using the mixed-effects design for any cross-validated measure on the Language MVPA dataset. 
First section determines OS to properly allocate filepaths

First this checks if the randomise input already exists, then if not it reregisters/standardizes the raw images
Arguments:
1: directory of files (e.g. Maps/Encoding)
2: mask name (in fmri/data/standard)
3: model name
4: chance level (in case of MVPA, 0 for encoding or RSA)

example usage:
sh postprocess_mfx.sh Maps/Encoding grayMatter cross_anim_L2P_ccsl 0

doc
RETURNHERE=${PWD}

CHANCE=0
OVERRIDE=FALSE
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
	-c|--chance)
    CHANCE="$2"
    shift # past argument
    ;;
	-o|--override)
    OVERRIDE=TRUE
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
echo "OVERRIDE  = ${OVERRIDE}"
projectDir=${ROOTDIR}/fmri/LanguageMVPA
desDir=${HOMEDIR}/GitHub/LanguageMVPA/multivariate/bash
refImage=${projectDir}/MNI152_T1_3mm_brain.nii.gz
refMask=${projectDir}/fnirt/MNI152_T1_3mm_brain_mask.nii.gz
targetDir=${projectDir}/${RESPATH}

cd ${targetDir}

if [ ! -f ${MASK}_${MODEL}_Group.nii.gz ] || [ $OVERRIDE == "TRUE" ]
then
	echo "Moving files"
	#move raw outputs
	mkdir raw
	mkdir std
	mv *_${MASK}_${MODEL}.nii.gz raw
	cd raw
	pwd
	for indMap in `ls | grep _${MASK}_${MODEL}.nii.gz`
	do
		sub=`echo ${indMap} | cut -d '_' -f1`
		echo ${sub}
		fslmaths ${projectDir}/data/$sub/masks/${sub}_grayMatter.nii.gz -bin tmp.nii.gz

		if [ "$CHANCE" == "0" ]
		then
			cp ${indMap} rnd_${indMap}
		else
			fslmaths tmp.nii.gz -mul ${CHANCE} tmp.nii.gz
			fslmaths ${indMap} -mul 100 -sub tmp.nii.gz rnd_${indMap}
		fi


		regMat=${projectDir}/data/$sub/reg/${sub}_example_func2standard.mat

		flirt -in rnd_${indMap} -out ../std/std_${indMap} -ref ${refImage} \
			-applyxfm -init ${regMat}
		rm rnd_${indMap} tmp.nii.gz
	done

	fslmerge -t ../${MASK}_${MODEL}_Group.nii.gz \
	../std/std_LMVPA001_${MASK}_${MODEL}.nii.gz \
	../std/std_LMVPA002_${MASK}_${MODEL}.nii.gz \
	../std/std_LMVPA003_${MASK}_${MODEL}.nii.gz \
	../std/std_LMVPA005_${MASK}_${MODEL}.nii.gz \
	../std/std_LMVPA006_${MASK}_${MODEL}.nii.gz \
	../std/std_LMVPA007_${MASK}_${MODEL}.nii.gz \
	../std/std_LMVPA008_${MASK}_${MODEL}.nii.gz \
	../std/std_LMVPA009_${MASK}_${MODEL}.nii.gz \
	../std/std_LMVPA010_${MASK}_${MODEL}.nii.gz \
	../std/std_LMVPA011_${MASK}_${MODEL}.nii.gz \
	../std/std_LMVPA013_${MASK}_${MODEL}.nii.gz \
	../std/std_LMVPA014_${MASK}_${MODEL}.nii.gz \
	../std/std_LMVPA015_${MASK}_${MODEL}.nii.gz \
	../std/std_LMVPA016_${MASK}_${MODEL}.nii.gz \
	../std/std_LMVPA017_${MASK}_${MODEL}.nii.gz \
	../std/std_LMVPA018_${MASK}_${MODEL}.nii.gz \
	../std/std_LMVPA019_${MASK}_${MODEL}.nii.gz 

	cd ..
fi

randomise -i ${MASK}_${MODEL}_Group.nii.gz -o n1000_${MASK}_${MODEL} \
	-v 5 -d $desDir/mfx_design.mat -t $desDir/mfx_design.con \
	-T -x --uncorrp -n 1000 -m ${projectDir}/data/standard/3mm_grayMatter
	
fdr -i n1000_${MASK}_${MODEL}_tfce_p_tstat1 --oneminusp -m ${projectDir}/data/standard/3mm_grayMatter \
	-q 0.05 -a n1000_${MASK}_${MODEL}_tfce_fdrp_tstat1

fdr -i n1000_${MASK}_${MODEL}_vox_p_tstat1 --oneminusp -m ${projectDir}/data/standard/3mm_grayMatter \
	-q 0.05 -a n1000_${MASK}_${MODEL}_vox_fdrp_tstat1
		
fdr -i n1000_${MASK}_${MODEL}_tfce_p_tstat2 --oneminusp -m ${projectDir}/data/standard/3mm_grayMatter \
	-q 0.05 -a n1000_${MASK}_${MODEL}_tfce_fdrp_tstat2

fdr -i n1000_${MASK}_${MODEL}_vox_p_tstat2 --oneminusp -m ${projectDir}/data/standard/3mm_grayMatter \
	-q 0.05 -a n1000_${MASK}_${MODEL}_vox_fdrp_tstat2
	
cd ${RETURNHERE}