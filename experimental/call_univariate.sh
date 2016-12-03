#!/bin/bash - 
#===============================================================================
#
#          FILE: univariate_setup.sh
# 
#         USAGE: ./univariate_setup.sh 
# 
#   DESCRIPTION: 
#   univariate_setup.sh
#   runs a sed replacement on the supplied subject, run and design file
#   should flexibly be able to take in sed arguments and generate it.
#   The fsf file should be specified ahead of time
#   This will reference the filemapping.txt file in order to specify the BOLD
#   scan unless otherwise specified
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
fmap="filemapping.txt"
TEMPLATE="None"
EXECUTE=False
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
		-i|--input)
		INPUT="$2"
		shift
		;;
		-o|--output)
		OUTPUT="$2"
		shift
		;;
		-t|--template)
		TEMPLATE="$2"
		shift
		;;
		-e|--execute)
		EXECUTE=True
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

function sedreplace {
vol=`fslinfo ${1}/data/${3}/raw/${5} dim4`
sed -e "s@###SUB###@${3}@g" -e "s@###RUN###@${4}@g" -e "s@###VOL###@${vol}}@g" \
	  -e "s@###INPUT###@${5}@g" -e "s@###OUTPUT###@${6}@g" \
	  ${1}/code/templates/${2E}.fsf > ${1}/data/${3}/notes/${3}_${4}_${2}.fsf

}


if [[ ${RUN} == "None" ]] && [ [${INPUT} == "None" ]]
then
	echo "Root directory: ${PROJECTDIR}"
	echo "Subject: ${SUB}"
	echo "Looping through runs in ${fmap}"
	echo "Design fsf: ${DESIGN}"
	echo "Output file: ${OUTPUT}"
	echo "Replacing: "
	while read line
	do
		output=`echo ${line} | cut -d ',' -f2`
		if [ [${output} == *"Run"* ]]
		then
			file=`echo ${line} | cut -d ',' -f1`
			rn=`echo ${output} | cut -d -f2
			printf -v j "%05d" $i
			sedreplace ${PROJECTDIR} ${TEMPLATE} ${SUB} ${run} ${file} ${OUTPUT}
		fi
	done < ${PROJECTDIR}/data/${SUB}/behav/${fmap}
else
echo "Input file: ${INPUT} | Run: ${RUN}"
fi


    feat ${PROJECTDIR}/data/${SUB}/des/${SUB}_${RUN}_${DESIGN}.fsf
