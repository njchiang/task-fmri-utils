# task-fmri-utils
generic utilities for task fmri analysis. assumes the user will be on the FUNC at UCLA.

Workflow after scanning a subject:
run setup_subject_nifti_2 on the FUNC
run make_file_structure.sh --initialize --noanalysis
modify format.sh to suit your analysis paths. modify git directory and FUNC directory

for each subject, run format_data.sh from that subject's code directory
data will be minimally preprocessed (motion corrected and registered to a BOLD template of your choosing)
