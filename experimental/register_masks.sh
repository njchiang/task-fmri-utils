#!/bin/bash - 
#===============================================================================
#
#          FILE: register_masks.sh
# 
#         USAGE: ./register_masks.sh 
# 
#   DESCRIPTION: 
# 
#       OPTIONS: ---
#  REQUIREMENTS: ---
#          BUGS: ---
#         NOTES: ---
#        AUTHOR: Jeff Chiang (jc), jeff.njchiang@gmail.com
#  ORGANIZATION: Monti Lab
#       CREATED: 11/03/2016 13:47:36
#      REVISION:  ---
#===============================================================================

set -o nounset                              # Treat unset variables as an error

PROJECTDIR=${PWD}
REGISTRATION_TARGET="None"
MASKLIST=""
MASKDIR=standard/masks
FILE_HEADER=""
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
		-f|--header)
		FILE_HEADER="$2"
		shift
		;;
		-t|--target)
		REGISTRATION_TARGET=${2}
		shift
		;;
		-m|--mask)
		MASKLIST="${MASKLIST} ${2}"
		shift
		;;
		-d|--directory)
		MASKDIR=${s}
		shift
		;;
		*)
 echo " "
		;;
	esac
	shift # past argument or value
done
date=`date +"%Y%m%d%H%M"`
logfile=${PROJECTDIR}/data/${SUB}/notes/register_masks_${date}.log
echo "Root directory: ${PROJECTDIR}"
echo "Subject: ${SUB}"
echo "Masks: ${MASKLIST}"
echo "Mask directory: ${MASKDIR}"
echo "Mask header: ${FILE_HEADER}"
echo "Register to: ${REGISTRATION_TARGET}"
echo "logging to: ${logfile}"
echo "Registering masks" >> $logfile

if [ ${REGISTRATION_TARGET} == "None" ]
then
	echo "No registration target specified"
else
	for m in ${MASKLIST}
	do
		echo "Registering ${m}"
		echo "Registering ${m}" >> ${logfile}
		cmd="flirt -in ${PROJECTDIR}/data/${MASKDIR}/${m} \
			-out ${PROJECTDIR}/data/${SUB}/analysis/masks/${FILE_HEADER}_${m} \
			-ref ${PROJECTDIR}/data/${SUB}/analysis/reg/${REGISTRATION_TARGET}_template_BOLD \
			-applyxfm -init	${PROJECTDIR}/data/${SUB}/analysis/reg/${REGISTRATION_TARGET}_standard_to_template.mat"
		echo ${cmd} >> ${logfile}
		${cmd}
	done
fi
