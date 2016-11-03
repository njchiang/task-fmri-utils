#!/bin/bash - 
#===============================================================================
#
#          FILE: preprocess_BOLD_data.sh
# 
#         USAGE: ./preprocess_BOLD_data.sh 
# 
#   DESCRIPTION: 
#   preprocesses raw data with a bunch of switches to toggle analyses
#   assumes structure made with make_file_structure.sh
#   outputs processing into data/[SUB/preproc/intermediate 
#   copies final output into data/[SUB]/preproc
## 
#       OPTIONS: ---
#       	-f filename
#       	-s subject
#       	-p path
#       	-r removevols
#       	--mcflirt
#       	--slicetiming
#       	--optibet
#       	--bet
#       	--template
#       	--register

#  REQUIREMENTS: ---
#          BUGS: ---
#         NOTES: ---
#        AUTHOR: Jeff Chiang (jc), jeff.njchiang@gmail.com
#  ORGANIZATION: Monti Lab
#       CREATED: 10/22/2016 9:44:10 PM
#      REVISION:  ---
#===============================================================================

set -o nounset                              # Treat unset variables as an error

PROJECTDIR=${PWD}
REGISTRATION_TARGET="BOLD_Template"
MAKE_TEMPLATE=false
TEMPLATE_NAME="None"
FILENAME="None"
SUB="None"
METHOD="mean"
while [[ $# -ge 1 ]]
do
	key="$1"
	case $key in
		-p|--path)
		PROJECTDIR="$2"
		shift # past argument
		;;	
		-s|--sub)
		SUB="$2"
		shift # past argument
		;;
		-f|--filename)
		FILENAME="$2"
		shift
		;;
		-t|--target)
		REGISTRATION_TARGET=${2}
		shift
		;;
		--template)
		MAKE_TEMPLATE=true
		;;
		-o|--output)
		TEMPLATE_NAME=${2}
		shift
		;;
		-m|--method)
		METHOD=${2}
		shift
		;;
		*)
        echo " \
preprocess_data.sh
preprocesses raw data with a bunch of switches to toggle analyses
assumes structure made with make_file_structure.sh
outputs processing into data/[SUB/preproc/intermediate 
copies final output into data/[SUB]/preproc
inputs:
	-f filename
	-s subject
	-p path
	-r removevols
	--mcflirt
	--slicetiming
	--optibet
	--bet
	--template
	--register [LOCAL_PATH_TO_TARGET]
"
		;;
	esac
	shift # past argument or value
done

if [[ ${TEMPLATE_NAME} == "None" ]]
then
	TEMPLATE_NAME=${FILENAME}
fi

logfile=${PROJECTDIR}/data/${SUB}/notes/BOLD_template_${TEMPLATE_NAME}.log
echo "Root directory: ${PROJECTDIR}"
echo "Subject: ${SUB}"
echo "Filename: ${FILENAME}"
echo "Template name: ${TEMPLATE_NAME}"
echo "Register to: ${REGISTRATION_TARGET}"
echo "logging to: ${logfile}"
echo "Making BOLD template"
echo "Making BOLD template" >> $logfile
if [[ ${METHOD} == "middle" ]]
then
	echo "Using middle timepoint" >> $logfile
	ntr=`fslval ${PROJECTDIR}/data/${SUB}/analysis/preproc/${FILENAME} dim4`
	ntr=$(( ntr / 2 ))
	cmd="fslroi ${PROJECTDIR}/data/${SUB}/analysis/preproc/${FILENAME} \
		${PROJECTDIR}/data/${SUB}/analysis/reg/${TEMPLATE_NAME}_template_BOLD \
		${ntr} 1"
else
	echo "Using TS mean" >> $logfile
	cmd="fslmaths ${PROJECTDIR}/data/${SUB}/analysis/preproc/${FILENAME} -Tmean \
	${PROJECTDIR}/data/${SUB}/analysis/reg/${TEMPLATE_NAME}_template_BOLD"
fi
echo ${cmd} >> $logfile
${cmd}

echo "Registering to standard"
echo "Registering BOLD template to standard" >> $logfile
cmd="flirt -in ${PROJECTDIR}/data/${SUB}/analysis/reg/${TEMPLATE_NAME}_template_BOLD \
	-ref ${FSLDIR}/data/standard/MNI152_T1_2mm_brain.nii.gz -omat \
	${PROJECTDIR}/data/${SUB}/analysis/reg/${TEMPLATE_NAME}_template_to_standard.mat" 
echo ${cmd} >> ${logfile}
$cmd

echo "Inverting standard transformation" >> ${logfile}
cmd="convert_xfm -omat ${PROJECTDIR}/data/${SUB}/analysis/reg/${TEMPLATE_NAME}_standard_to_template.mat \
	-inverse ${PROJECTDIR}/data/${SUB}/analysis/reg/${TEMPLATE_NAME}_template_to_standard.mat"
echo ${cmd} >> ${logfile}
$cmd


if [[ $MAKE_TEMPLATE == "true" ]]
then
	echo "Copying file to bold template" 
	echo "copying file to bold template" >> ${logfile}
	cmd="cp ${PROJECTDIR}/data/${SUB}/analysis/reg/${TEMPLATE_NAME}_template_BOLD.nii.gz \
		${PROJECTDIR}/data/${SUB}/analysis/reg/BOLD_template.nii.gz"
	echo ${cmd} >> ${logfile}
	${cmd}
	echo "Copying unregistered file"
	cmd="cp ${PROJECTDIR}/data/${SUB}/analysis/preproc/${FILENAME}.nii.gz \
		${PROJECTDIR}/data/${SUB}/analysis/${TEMPLATE_NAME}.nii.gz"
	echo ${cmd} >> ${logfile}
	$cmd

else
	if [[ "${REGISTRATION_TARGET}" != "None" ]]
	then
		echo "Registration to template"
		if [ ! -f ${PROJECTDIR}/data/${SUB}/analysis/reg/${TEMPLATE_NAME}_to_template.mat ]
		then
			echo "Calculate registration to template" >> ${logfile}
			cmd="flirt -in ${PROJECTDIR}/data/${SUB}/analysis/reg/${TEMPLATE_NAME}_template_BOLD \
				-ref ${PROJECTDIR}/data/${SUB}/analysis/reg/${REGISTRATION_TARGET} -omat \
				${PROJECTDIR}/data/${SUB}/analysis/reg/${TEMPLATE_NAME}_to_template.mat"
			echo ${cmd} >> $logfile
			$cmd
		fi
		echo "Apply registration to template" >> ${logfile}
		cmd="flirt -in ${PROJECTDIR}/data/${SUB}/analysis/preproc/${FILENAME}.nii.gz \
			-out ${PROJECTDIR}/data/${SUB}/analysis/${TEMPLATE_NAME}.nii.gz \
			-ref ${PROJECTDIR}/data/${SUB}/analysis/reg/${REGISTRATION_TARGET} \
			-applyxfm -init \
			${PROJECTDIR}/data/${SUB}/analysis/reg/${TEMPLATE_NAME}_to_template.mat"
		echo ${cmd} >> $logfile
		$cmd
	fi
fi
