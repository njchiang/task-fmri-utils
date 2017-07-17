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
RUN_BET=false
RUN_FLIRT=false
RUN_SLICETIMING=false
MCFLIRT_ARGS="-plots -report -stages 4 -sinc_final"
SLICETIMING_ARGS=''
FILENAME="None"
OUTPUT="None"
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
		--bet)
		RUN_BET=true
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
		*)
        echo " \
preprocess_BOLD_data.sh
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
"
		;;
	esac
	shift # past argument or value
done
FILENAME=`echo ${FILENAME} | cut -d '.' -f1` # assumes no extra .'s
if [ $OUTPUT == "None" ]
then
	OUTPUT=${FILENAME}_preproc
fi

logfile=${PROJECTDIR}/data/${SUB}/notes/preprocess_BOLD_${FILENAME}.log
echo "Root directory: ${PROJECTDIR}"
echo "Subject: ${SUB}"
echo "Filename: ${FILENAME}"
echo "Output: ${OUTPUT}"
echo "Remove vols: ${REMOVEVOLS}"
echo "Motion correction: ${RUN_FLIRT}"
echo "Motion correction args: ${MCFLIRT_ARGS}"
echo "Slicetiming correction: ${RUN_SLICETIMING}"
echo "Slicetiming args: ${SLICETIMING_ARGS}"
echo "BET: ${RUN_BET}"
echo "logging to: ${logfile}"

if [[ "$FILENAME" == "None" ]]
then
	echo "Please specify a filename"
else

	if [[ "$SUB" == "None" ]]
	then
		echo "No subject specified"
	else
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
				${PROJECTDIR}/data/${SUB}/analysis/preproc/tmp.nii.gz \
				${REMOVEVOLS} -1"
		else
			cmd="cp ${PROJECTDIR}/data/${SUB}/raw/${FILENAME}.nii.gz \
				${PROJECTDIR}/data/${SUB}/analysis/preproc/tmp.nii.gz"
		fi

		echo $cmd >> $logfile
		$cmd

		########### FLIRT ###############
		if [[ "$RUN_FLIRT" == "true" ]]
		then
			echo "Running FLIRT" >> $logfile
			echo "Running FLIRT"
			cmd="mcflirt -in ${PROJECTDIR}/data/${SUB}/analysis/preproc/tmp \
				-out ${PROJECTDIR}/data/${SUB}/analysis/preproc/tmp \
				${MCFLIRT_ARGS}"
			echo ${cmd} >> $logfile
			${cmd}
			mv ${PROJECTDIR}/data/${SUB}/analysis/preproc/tmp.par \
			${PROJECTDIR}/data/${SUB}/analysis/preproc/${OUTPUT}.par

		fi

		########### SLICETIMING ##############
		if [[ "${RUN_SLICETIMING}" == "true" ]]
		then
			echo "Slicetiming" >> $logfile
			echo "Slicetiming"
			cmd="slicetimer -i ${PROJECTDIR}/data/${SUB}/analysis/preproc/tmp.nii.gz \
				-o ${PROJECTDIR}/data/${SUB}/analysis/preproc/tmp.nii.gz \
				$SLICETIMING_ARGS"
			echo ${cmd} >> $logfile
			$cmd
		fi

		############ RUN BET ##############
		if [[ ${RUN_BET} == "true" ]]
		then
			echo "Brain extraction" >> $logfile
			echo "Brain extraction"
			cmd="bet ${PROJECTDIR}/data/${SUB}/analysis/preproc/tmp \
				${PROJECTDIR}/data/${SUB}/analysis/preproc/tmp -m -F"
			echo ${cmd} >> $logfile
			$cmd
			mv ${PROJECTDIR}/data/${SUB}/analysis/preproc/tmp_mask.nii.gz \
				${PROJECTDIR}/data/${SUB}/analysis/preproc/${OUTPUT}_mask.nii.gz
		fi
		############ FINAL FILE ###############
		echo "Renaming final file" >> $logfile
		echo "Renaming final file"
		cmd="mv ${PROJECTDIR}/data/${SUB}/analysis/preproc/tmp.nii.gz \
			${PROJECTDIR}/data/${SUB}/analysis/preproc/${OUTPUT}.nii.gz"
		echo ${cmd} >> $logfile
		$cmd
	fi
fi


