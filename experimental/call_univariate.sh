#!/bin/bash - 
#===============================================================================
#
#          FILE: call_univariate.sh
# 
#         USAGE: ./call_univariate.sh 
# 
#   DESCRIPTION: 
#   call_univariate.sh
#   runs a sed replacement on the supplied subject, run and design file
#   should flexibly be able to take in sed arguments and generate it.
#   The fsf file should be specified ahead of time 
#       OPTIONS: ---
#       inputs: 
#       -d design fsf file
#       -s subject
#       -r run
#       -p path
#       -i input
#       -o output
#       outputs into analysis/univariate/subject 
#  REQUIREMENTS: ---
#          BUGS: ---
#         NOTES: ---
#        AUTHOR: Jeff Chiang (jc), jeff.njchiang@gmail.com
#  ORGANIZATION: Monti Lab
#       CREATED: 10/22/2016 10:06:27 PM
#      REVISION:  ---
#===============================================================================

set -o nounset                              # Treat unset variables as an error
PROJECTDIR=${PWD}
INPUT=''
OUTPUT=''
SUB="None"
RUN="None"
DESIGN="None"
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
		-r|--run)
		RUN="$2"
		shift
		;;
		-i|--input)
		INPUT="$2"
		shift
		;;
		-o|--output)
		OUTPUT="$2"
		shift
		;;
		-d|--design)
		DESIGN="$2"
		shift
		;;
		*)
        echo " \
call_univariate.sh
runs a sed replacement on the supplied subject, run and design file
should flexibly be able to take in sed arguments and generate it.
The fsf file should be specified ahead of time

inputs: 
-d design fsf file
-s subject
-r run
-p path
-i input
-o output
outputs into analysis/univariate/subject 

"
		;;
	esac
	shift # past argument or value
done
echo "Root directory: ${PROJECTDIR}"
echo "Subject: ${SUB} Run: ${RUN}"
echo "Input file: ${INPUT}"
echo "Output file: ${OUTPUT}"
echo "Design fsf: ${DESIGN}"

VOL=`fslinfo ${PROJECTDIR}/data/${SUB}/preproc/${INPUT} dim4`

  sed -e "s@###SUB###@${SUB}@g" -e "s@###RUN###@${RUN}@g" -e "s@###VOL###@${VOL}@g" \
	  -e "s@###INPUT###@${INPUT}@g" -e "s@###OUTPUT###@${OUTPUT}@g" \
	  ${PROJECTDIR}/code/des/${DESIGN}.fsf > ${PROJECTDIR}/data/${SUB}/des/${SUB}_${RUN}_${DESIGN}.fsf
  feat ${PROJECTDIR}/data/${SUB}/des/${SUB}_${RUN}_${DESIGN}.fsf
