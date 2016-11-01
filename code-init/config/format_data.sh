#!/bin/bash
#===============================================================================
#
#          FILE: format_data.sh
# 
#         USAGE: ./format_data.sh --initialize -s sub1 -s sub2 -a analysis1 -p
#         FILE_PATH
#         OR
#         ./format_data.sh -s sub3 -s sub4 (to add more subjects)
# 
#   DESCRIPTION: 
#   this script is run in each analysis, sets up file structure and populates it with subjects
#   arguments:
#   basically is a wrapper for make_file_structure.sh and copies data in. should
#   be able to pull new subjects without the --initialize option
#   
#	OPTIONS: ---
#       --initialize | switch for first time initialization
#       -s subject
#       -a analyses 
#       -p path
#   REQUIREMENTS: ---
#          BUGS: ---
#         NOTES: ---
#        AUTHOR: Jeff Chiang (jc), jeff.njchiang@gmail.com
#  ORGANIZATION: Monti Lab
#       CREATED: 10/22/2016 10:18:15 PM
#      REVISION:  ---
#===============================================================================

set -o nounset                              # Treat unset variables as an error


INITIALIZE=false
OPATH=${PWD}
SUB=''
SUBLIST=''
ANALYSIS=''
SERVER='njchiang@funcserv1.psych.ucla.edu:'
SPATH='/space/raid5/data/monti/Analysis/'
while [[ $# -ge 1 ]]
do
	key="$1"

	case $key in
		-p|--OPATH)
		OPATH="$2"
		shift # past argument
		;;	
		-s|--sub)
		ADDSUBJECTS=true
		SUB="${SUB} -s $2"
		SUBLIST="${SUBLIST} ${2}"
		shift # past argument
		;;
		-a|--analysis)
		ANALYSIS="${ANALYSIS} -a ${2}"
		shift
		;;
		--initialize)
		INITIALIZE=true
		;;
		*)
		echo "./format_data.sh --initialize -s sub1 -s sub2 -a analysis1 -p FILE_PATH"
		echo "./format_data.sh -s sub3 -s sub4 (to add more subjects)"
		;;
	esac
	shift # past argument or value
done

case ${OSTYPE} in 
	darwin*)
	GITDIR='~/GitHub/task-fmri-tools'
	;;
	linux*)
	GITDIR='~/data/GitHub/task-fmri-tools'
esac

date=`date +"%Y%m%d%H%M"`
logfile=${OPATH}/data/${SUB}/logs/format_data_${date}.log
echo "Logging to: ${logfile}"

if [[ ${INITIALIZE} == "true" ]]
then
	cmd="sh ${GITDIR}/bash/make_file_structure.sh --initialize ${SUB} ${ANALYSIS} -p ${OPATH}"
else
	cmd="sh ${GITDIR}/bash/make_file_structure.sh ${SUB} --noanalysis -p ${OPATH}"
fi
echo ${cmd} >> ${logfile}
${cmd}
# scp command here
for s in ${SUBLIST}
do
	cmd="scp ${SERVER}:${SPATH}/analysis/${s}_*.nii.gz ${OPATH}/data/${s}/raw/"
	echo ${cmd} >> ${logfile}
	${cmd}
done
