#!/bin/bash - 
#===============================================================================
#
#          FILE: rename_runs.sh
# 
#         USAGE: ./rename_runs.sh 
# 
#   DESCRIPTION: searches the subject's behav folder for a filemapping.txt. If
#   found, it iterates through that and symlinks new filenames in the "raw"
#   directory
# 
#       OPTIONS: ---
#  REQUIREMENTS: ---
#          BUGS: ---
#         NOTES: ---
#        AUTHOR: Jeff Chiang (jc), jeff.njchiang@gmail.com
#  ORGANIZATION: Monti Lab
#       CREATED: 11/01/2016 16:19:09
#      REVISION:  ---
#===============================================================================

set -o nounset                              # Treat unset variables as an error

MODE="local" # or FUNC
OPATH=${PWD}
SUBLIST=''
FILENAME='filemapping.txt'
LINK=true
while [[ $# -ge 1 ]]
do
	key="$1"

	case $key in
		-p|--OPATH)
		OPATH="$2"
		shift # past argument
		;;	
		-s|--sub)
		SUBLIST="${SUBLIST} ${2}"
		shift # past argument
		;;
		-f|--file)
		FILENAME="${2}"
		shift # past argument
		;;
		--nolink)
		LINK=false
		;;
		*)
		echo "./format_data.sh -p PATH -s sub3 -s sub4 (to add more subjects)"
		;;
	esac
	shift # past argument or value
done

for sub in ${SUBLIST}
do
	echo "Processing ${sub}"
	if [ -d ${OPATH}/data/${sub} ]
	then
		while read line
		do
			orig_name=`echo ${line} | cut -d ',' -f1`
			target_name=`echo ${line} | cut -d ',' -f2`
			if [ LINK == "true" ]
			then
				ln -s ${OPATH}/data${sub}/raw/${orig_name}/${orig_name}.nii.gz \
					${OPATH}/data${sub}/raw/${target_name}
			else
				cp ${OPATH}/data${sub}/raw/${orig_name}/${orig_name}.nii.gz \
					${OPATH}/data${sub}/raw/${target_name}
			fi
		done < ${OPATH}/data/${sub}/behav/${FILENAME}.txt
	fi
done
