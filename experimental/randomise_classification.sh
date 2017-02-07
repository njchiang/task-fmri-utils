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
sh randomise_classification.sh -n mfx_design.fsf -s grayMatter -m grayMatter 
-c 0 -p Analogy -h analogy -r analysis/multivariate/searchlight

doc
RETURNHERE=${PWD}
DESIGN=mfx_design
MODEL=None
CHANCE=0
HEADER=None
MASK=None
PROJECT=None
SLMASK=grayMatter
RESPATH=analysis/multivariate/searchlight
while [[ $# -gt 1 ]]
do
key="$1"

case $key in
    -n|--model)
    MODEL="$2"
    shift # past argument
    ;;
	-d|--design)
    DESIGN="$2"
    shift # past argument
    ;;
    -r|--path)
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
	-p|--project)
	PROJECT="$2"
	shift # past argument
	;;
	-h|--header)
	HEADER="$2"
	shift # past argument
	;;
	-s|--slmask)
	SLMASK="$2"
	shift # past argument
	;;
	-o|--override)
	OVERRIDE="$2"
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
   ROOTDIR=/Volumes
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
projectDir=${ROOTDIR}/fmri/${PROJECT}
codeDir=${HOMEDIR}/GitHub/task-fmri-utils
desDir=${HOMEDIR}/CloudStation/Grad/Research/${PROJECT}/code/templates
refImage=${FSLDIR}/data/standard/MNI152_T1_2mm_brain.nii.gz
refMask=${FSLDIR}/data/standard/MNI152_T1_2mm_brain_mask.nii.gz
targetDir=${projectDir}/${RESPATH}

cd ${targetDir}

if [ ! -f ${SLMASK}_${MODEL}_Group.nii.gz ] || [ "$OVERRIDE" == "TRUE" ]
then
	echo "Moving files"
	#move raw outputs
	mkdir raw
	mkdir std
	mv *_${SLMASK}_${MODEL}.nii.gz raw
	cd raw
	pwd
	for indMap in `ls | grep _${SLMASK}_${MODEL}.nii.gz`
	do
		sub=`echo ${indMap} | cut -d '_' -f1`
		echo ${sub}
		fslmaths ${projectDir}/data/$sub/analysis/masks/${SLMASK}.nii.gz -bin tmp.nii.gz

		if [ "$CHANCE" == "0" ]
		then
			cp ${indMap} rnd_${indMap}
		else
			fslmaths tmp.nii.gz -mul ${CHANCE} tmp.nii.gz
			fslmaths ${indMap} -mul 100 -sub tmp.nii.gz rnd_${indMap}
		fi

		regMat=${projectDir}/data/$sub/analysis/reg/BOLD_template_to_standard.mat

		flirt -in rnd_${indMap} -out ../std/std_${indMap} -ref ${refImage} \
			-applyxfm -init ${regMat}
		rm rnd_${indMap} tmp.nii.gz
	done

	fslmerge -t ../${SLMASK}_${MODEL}_Group.nii.gz \
		../std/std_${HEADER}*_${SLMASK}_${MODEL}.nii.gz
#	../std/std_LMVPA001_${MASK}_${MODEL}.nii.gz \
#	../std/std_LMVPA002_${MASK}_${MODEL}.nii.gz \
#	../std/std_LMVPA003_${MASK}_${MODEL}.nii.gz \
#	../std/std_LMVPA005_${MASK}_${MODEL}.nii.gz \
#	../std/std_LMVPA006_${MASK}_${MODEL}.nii.gz \
#	../std/std_LMVPA007_${MASK}_${MODEL}.nii.gz \
#	../std/std_LMVPA008_${MASK}_${MODEL}.nii.gz \
#	../std/std_LMVPA009_${MASK}_${MODEL}.nii.gz \
#	../std/std_LMVPA010_${MASK}_${MODEL}.nii.gz \
#	../std/std_LMVPA011_${MASK}_${MODEL}.nii.gz \
#	../std/std_LMVPA013_${MASK}_${MODEL}.nii.gz \
#	../std/std_LMVPA014_${MASK}_${MODEL}.nii.gz \
#	../std/std_LMVPA015_${MASK}_${MODEL}.nii.gz \
#	../std/std_LMVPA016_${MASK}_${MODEL}.nii.gz \
#	../std/std_LMVPA017_${MASK}_${MODEL}.nii.gz \
#	../std/std_LMVPA018_${MASK}_${MODEL}.nii.gz \
#	../std/std_LMVPA019_${MASK}_${MODEL}.nii.gz 

	cd ..
fi

randomise -i ${SLMASK}_${MODEL}_Group.nii.gz -o n1000_${MASK}_${MODEL} \
	-v 5 -d $desDir/${DESIGN}.mat -t $desDir/${DESIGN}.con \
	-T -x --uncorrp -n 1000 -m ${projectDir}/data/standard/masks/${MASK}
	
fdr -i n1000_${MASK}_${MODEL}_tfce_p_tstat1 --oneminusp -m ${projectDir}/data/standard/masks/${MASK} \
	-q 0.05 -a n1000_${MASK}_${MODEL}_tfce_fdrp_tstat1

fdr -i n1000_${MASK}_${MODEL}_vox_p_tstat1 --oneminusp -m ${projectDir}/data/standard/masks/${MASK} \
	-q 0.05 -a n1000_${MASK}_${MODEL}_vox_fdrp_tstat1

	
cd ${RETURNHERE}
