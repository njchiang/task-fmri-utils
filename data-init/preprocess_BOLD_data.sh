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
REMOVEVOLS=0
RUN_FLIRT=false
RUN_SLICETIMING=false
RUN_BET=false
RUN_OPTIBET=false
MCFLIRT_ARGS="-plots -report -stages 4 -sinc_final"
SLICETIMING_ARGS=''
MAKE_TEMPLATE=false
REGISTRATION_TARGET="None"
FILENAME="None"
SUB="None"
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
		-r|--removevols)
		REMOVEVOLS="$2"
		shift
		;;
		--mcflirt)
		RUN_FLIRT=true
		;;
		--mcflirt_args)
		MCFLIRT_ARGS=$2
		shift
		;;
		--slicetiming)
		RUN_SLICETIMING=true
		SLICETIMING_ARGS=$2
		shift
		;;
		--bet)
		RUN_BET=true
		;;
		--optibet)
		RUN_OPTIBET=true
		;;
		--template)
		MAKE_TEMPLATE=true
		;;
		--register)
		REGISTRATION_TARGET=${2}
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

logfile=${PROJECTDIR}/data/${SUB}/notes/preprocess_BOLD_${FILENAME}.log
echo "Root directory: ${PROJECTDIR}"
echo "Subject: ${SUB}"
echo "Filename: ${FILENAME}"
echo "Remove vols: ${REMOVEVOLS}"
echo "Motion correction: ${RUN_FLIRT}"
echo "Motion correction args: ${MCFLIRT_ARGS}"
echo "Slicetiming correction: ${RUN_SLICETIMING}"
echo "Slicetiming args: ${SLICETIMING_ARGS}"
echo "Make template: ${MAKE_TEMPLATE}"
echo "Bet: ${RUN_BET}"
echo "Optibet: ${RUN_OPTIBET}"
echo "Register to: ${REGISTRATION_TARGET}"
echo "logging to: ${logfile}"

if [[ "$FILENAME" == "None" ]]
then
	echo "Please specify a filename"
else

	if [[ "$SUB" == "None" ]]
	then
		echo "No subject specified"
	else

		FILENAME=`echo ${FILENAME} | cut -d '.' -f1` # assumes no extra .'s
		echo "Directory: $PROJECTDIR" >> $logfile
		echo "Subject: ${SUB}" >> $logfile
		echo "Filename: ${FILENAME}" >> $logfile

		# make it through sanity checks
		# check if files exist:
		#
		########### Remove TRs ###############
		if [[ "$REMOVEVOLS" -gt 0 ]] 
		then
			# fix FILENAME here
			cmd="fslroi ${PROJECTDIR}/data/${SUB}/raw/${FILENAME} \
				${PROJECTDIR}/data/${SUB}/analysis/tmp.nii.gz \
				${REMOVEVOLS} -1"
		else
			cmd="cp ${PROJECTDIR}/data/${SUB}/raw/${FILENAME}.nii.gz \
				${PROJECTDIR}/data/${SUB}/analysis/tmp.nii.gz"
		fi

		echo $cmd >> $logfile
		$cmd

		########### FLIRT ###############


		if [[ "$RUN_FLIRT" == "true" ]]
		then
			echo "Running FLIRT" >> $logfile
			echo "Running FLIRT"
			cmd="mcflirt -in ${PROJECTDIR}/data/${SUB}/analysis/tmp.nii.gz \
				-out ${PROJECTDIR}/data/${SUB}/analysis/tmp.nii.gz \
				${MCFLIRT_ARGS}"
			echo ${cmd} >> $logfile
			${cmd}
		fi

		if [[ "${RUN_SLICETIMING}" == "true" ]]
		then
			echo "Slicetiming" >> $logfile
			echo "Slicetiming"
			cmd="slicetimer -i ${PROJECTDIR}/data/${SUB}/analysis/tmp.nii.gz \
				-o ${PROJECTDIR}/data/${SUB}/analysis/tmp.nii.gz \
				$SLICETIMING_ARGS"
			echo ${cmd} >> $logfile
			$cmd
		fi

		if [[ "${MAKE_TEMPLATE}" == "true" ]]
		then
			echo "Making BOLD template"
			echo "Making BOLD template" >> $logfile
			cmd="fslmaths ${PROJECTDIR}/data/${SUB}/analysis/tmp.nii.gz -Tmean \
				${PROJECTDIR}/data/${SUB}/analysis/reg/${FILENAME}_template_BOLD"
			echo ${cmd} >> $logfile
			echo "Registering to standard"
			echo "Registering BOLD template to standard" >> $logfile
			cmd="flirt -in ${PROJECTDIR}/data/${SUB}/analysis/reg/${FILENAME}_template_BOLD \
				-ref ${FSLDIR}/data/standard/MNI152_T1_2mm_brain.nii.gz -omat \
				${PROJECTDIR}/data/${SUB}/analysis/reg/${FILENAME}_template_to_standard.mat" 
			echo ${cmd} >> ${logfile}
			$cmd
			echo "Inverting standard transformation" >> ${logfile}
			cmd="convert_xfm -omat ${FILENAME}_standard_to_template.mat \
				-inverse ${PROJECTDIR}/data/${SUB}/analysis/reg/${FILENAME}_template_to_standard.mat"
			echo ${cmd} >> ${logfile}
			$cmd
		fi

		if [[ "${REGISTRATION_TARGET}" != "None" ]]
		then
			echo "Registration"
			# should check for file first...
			cmd="flirt -in ${PROJECTDIR}/data/${SUB}/analysis/reg/${FILENAME}_template_BOLD \
				-ref ${REGISTRATION_TARGET} -omat \
				${PROJECTDIR}/data/${SUB}/analysis/reg/${FILENAME}_to_template.mat"
			echo ${cmd} >> $logfile
			$cmd
			cmd="flirt -in ${PROJECTDIR}/data/${SUB}/analysis/tmp.nii.gz \
				-out flirt -in ${PROJECTDIR}/data/${SUB}/analysis/tmp.nii.gz \
				-ref ${REGISTRATION_TARGET} -applyxfm -init \
				${PROJECTDIR}/data/${SUB}/analysis/reg/${FILENAME}_to_template.mat"
			echo ${cmd} >> $logfile
			$cmd
		fi
		echo "Renaming final file" >> $logfile
		echo "Renaming final file"
		cmd="mv flirt -in ${PROJECTDIR}/data/${SUB}/analysis/tmp.nii.gz \
			flirt -in ${PROJECTDIR}/data/${SUB}/analysis/${FILENAME}"
		echo ${cmd} >> $logfile
		$cmd

	fi
fi

