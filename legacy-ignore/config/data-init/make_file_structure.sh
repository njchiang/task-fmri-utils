#!/bin/bash - 
#===============================================================================
#
#          FILE: make_file_structure.sh
# 
#         USAGE: ./make_file_structure.sh 
# 
#   DESCRIPTION:make_file_structure.sh
#   sets up file structure for a study
#   Inputs: 
#   OPTIONS: 
#       -p: root directory
#		-a: analysis directories (defaults to localizer, multivariate, univariate)
#		-s: subject directories
#		--noanalysis: does not make analysis directories (does not make subjects by
#		default)
#		--initialize: moves generic scripts over. BE CAREFUL WITH THIS.
#		Returns: none 
#  REQUIREMENTS: ---
#          BUGS: ---
#         NOTES: ---
#        AUTHOR: Jeff Chiang (jc), jeff.njchiang@gmail.com
#  ORGANIZATION: Monti Lab
#       CREATED: 10/22/2016 9:39:36 PM
#      REVISION:  ---
#===============================================================================

set -o nounset                              # Treat unset variables as an error
OPATH=${PWD}
SUB=''
ANALYSIS=''
ADDANALYSIS=true
ADDSUBJECTS=false
ADDCODE=false
case ${OSTYPE} in 
	darwin*)
	GITDIR="${HOME}/GitHub/task-fmri-utils"
	;;
	linux*)
	GITDIR="${HOME}/data/GitHub/task-fmri-utils"
esac

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
		SUB="${SUB} $2"
		shift # past argument
		;;
		-a|--analysis)
		ANALYSIS="${ANALYSIS} ${2}"
		shift
		;;
		--noanalysis)
		ADDANALYSIS=false
		;;
		--initialize)
		ADDCODE=true
		;;
		*)
			ADDANALYSIS=0
        echo " make_file_structure.sh "
		echo " sets up file structure for a study "
		echo " Inputs: "
		echo " -p: root directory"
		echo " -a: analysis directories (defaults to localizer, multivariate, univariate)"
		echo " -s: subject directories"
		echo " --noanalysis: does not make analysis directories (does not make subjects by"
		echo " default)"
		echo " --initialize: moves generic scripts over. BE CAREFUL WITH THIS."
		echo " Returns: none"
		;;
	esac
	shift # past argument or value
done

echo "Making file structure for project in: ${OPATH}"
echo "Write analysis directories: ${ADDANALYSIS}"
echo "Write subject directories: ${ADDSUBJECTS}"
echo "Copy generic code over: ${ADDCODE}"

if [[ "$ADDANALYSIS" == "true" ]]
then

	if [[ $ANALYSIS == '' ]]
	then
		ANALYSIS="localizer multivariate univariate encoding"
	fi

	for a in ${ANALYSIS}
		do
		echo "Making ${a}"
		mkdir -p ${OPATH}/analysis/${a}
	done
fi

for s in ${SUB}
do
	echo "Setting up ${s}"
	# result of setup_subject_nifti_2, contains raw and mcf files
	mkdir -p ${OPATH}/data/${s}/raw
	# raw output of behavioral
	mkdir -p ${OPATH}/data/${s}/behav/from_scanner
	# behavioral to regressors for FSL	
	mkdir -p ${OPATH}/data/${s}/behav/regressors
	# behavioral to labels for PyMVPA
	mkdir -p ${OPATH}/data/${s}/behav/labels
	# registration files (templates and xfm matrices)
	mkdir -p ${OPATH}/data/${s}/analysis/reg
	# preprocessed files (prior to registering)
	mkdir -p ${OPATH}/data/${s}/analysis/preproc
	# mask files (registered to either run or BOLD template)
	mkdir -p ${OPATH}/data/${s}/analysis/masks
	# log files from setup_subject
	mkdir -p ${OPATH}/data/${s}/notes
	# raw dicom files
	mkdir -p ${OPATH}/data/${s}/dicom
done

if [[ "$ADDCODE" == "true" ]]
then
	cp -r ${GITDIR}/code-init ${OPATH}/code-init
	# i think this should avoid overriding
	mv ${OPATH}/code-init ${OPATH}/code
fi


