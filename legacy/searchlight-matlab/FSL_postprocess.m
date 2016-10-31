function []=FSL_postprocess(Maps_path, warp_path, ref_path, mask_path, maskName, models, userOptions)
userOptions=setIfUnset(userOptions, 'postprocess', 'linear');
logFile=fopen([userOptions.analysisName '_searchlightLog.txt'], 'w');
setenv('PATH', [getenv('PATH') ':usr/local/fsl']);
setenv('FSLDIR', '/usr/local/fsl')
setenv('FSLOUTPUTTYPE', 'NIFTI_GZ')

for modelI = 1:numel(models)
    modelName=models(modelI).name;
    fprintf(logFile, modelName)
    mergeList=[];
    for subjectI = 1:Nsubjects
        subject=userOptions.subjectNames{subjectI};
        fprintf(logFile, subject)
        
        
        thisFilename=[Maps_path '/' subject '_' maskName '_' modelName '_rMap'];
        
        [~, m]=unix(['/usr/local/fsl/bin/fslchfiletype NIFTI_GZ ' thisFilename]);
        fprintf(logFile,m);
        if strcmp(userOptions.interp,'nonlinear')
            [~, m]=unix(['/usr/local/fsl/bin/applywarp -r ' ref_path '/MNI152_T1_3mm_brain.nii.gz -w ' warp_path '/' subject '_warp_highres2standard.mat.nii.gz -i ' thisFilename ' -o std_' thisFilename]);
            fprintf(logFile,m);
            
        else
            [~, m]=unix(['/usr/local/fsl/bin/flirt -ref ' ref_path '/MNI152_T1_3mm_brain.nii.gz -init ' warp_path '/' subject '_highres2standard.mat.nii.gz -applyxfm -in ' thisFilename ' -out std_' thisFilename]);
            fprintf(logFile,m);
        end
        mergeList=[mergeList ' std_' thisFilename];
    end
    [s, m]=unix(['/usr/local/fsl/bin/fslmerge -t Group_' modelName ' ' mergeList]);
    fprintf(logFile,m);
    [s,m]=unix(['/usr/local/fsl/bin/randomise -i Group_' modelName ' -o  n1000_' modelName ' -m ' mask_path '/' maskName ' -n 1000 -1 -T'])
    
end
fclose(logFile);