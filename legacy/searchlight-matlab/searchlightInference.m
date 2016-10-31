function [] = searchlightInference(models, userOptions)

%% load the previously computed rMaps and concatenate across subjects
% prepare the rMaps:
for subjectI = 1:Nsubjects
    load([userOptions.rootPath,filesep,'Maps',filesep,'rs_subject',num2str(subjectI),'.mat'])
    rMaps{subjectI} = rs;
    fprintf(['loading the correlation maps for subject %d \n'],subjectI);
end
% concatenate across subjects
for modelI = 1:numel(models)
    modelName=models(modelI).name;
    for subI = 1:Nsubjects
        thisRs = rMaps{subI};
        thisModelSims(:,:,:,subI) = thisRs(:,:,:,modelI);
    end
    % obtain a pMaps from applying a 1-sided signrank test and also t-test to
    % the model similarities:
    for x=1:size(thisModelSims,1)
        for y=1:size(thisModelSims,2)
            for z=1:size(thisModelSims,3)
                if mask(x,y,z) == 1
                    [h p1(x,y,z)] = ttest(squeeze(thisModelSims(x,y,z,:)),0,0.05,'right');
                    [p2(x,y,z)] = signrank_onesided(squeeze(thisModelSims(x,y,z,:)));
                else
                    p1(x,y,z) = NaN;
                    p2(x,y,z) = NaN;
                end
            end
        end
        disp(x)
    end
    % apply FDR correction
    pThrsh_t = FDRthreshold(p1,0.05,mask);
    pThrsh_sr = FDRthreshold(p2,0.05,mask);
    p_bnf = 0.05/sum(mask(:));
    % mark the suprathreshold voxels in yellow
    supraThreshMarked_t = zeros(size(p1));
    supraThreshMarked_t(p1 <= pThrsh_t) = 1;
    supraThreshMarked_sr = zeros(size(p2));
    supraThreshMarked_sr(p2 <= pThrsh_sr) = 1;
    
    %don't need this
    %     % display the location where the effect was inserted (in green):
    %     brainVol = addRoiToVol(map2vol(anatVol),mask2roi(mask),[1 0 0],2);
    %     brainVol_effectLoc = addBinaryMapToVol(brainVol,Mask.*mask,[0 1 0]);
    %     showVol(brainVol_effectLoc,'simulated effect [green]',2);
    %     handleCurrentFigure([returnHere,filesep,'DEMO4',filesep,'results_DEMO4_simulatedEffectRegion'],userOptions);
    %
    
    %% figure these out for FSL
    % display the FDR-thresholded maps on a sample anatomy (signed rank test) :
    brainVol = addRoiToVol(map2vol(anatVol),mask2roi(mask),[1 0 0],2);
    brainVol_sr = addBinaryMapToVol(brainVol,supraThreshMarked_sr.*mask,[1 1 0]);
    showVol(brainVol_sr,'signrank, E(FDR) < .05',3)
    handleCurrentFigure([returnHere,filesep,'Figures',filesep, modelName '_Searchlight_signRank'],userOptions);
    
    % display the FDR-thresholded maps on a sample anatomy (t-test) :
    brainVol = addRoiToVol(map2vol(anatVol),mask2roi(mask),[1 0 0],2);
    brainVol_t = addBinaryMapToVol(brainVol,supraThreshMarked_t.*mask,[1 1 0]);
    showVol(brainVol_t,'t-test, E(FDR) < .05',4)
    handleCurrentFigure([returnHere,filesep,'Figures',filesep, modelName '_Searchlight_tTest'],userOptions);
    
    
    
    % this should read in 3mm standard
    writeOpts.name = [userOptions.analysisName '_rMap_' maskName '_' modelName '_groupStats.img'];
    writeOpts.description='group_level_stats';
    writeOpts.template=[userOptions.rootPath '/MNI152_T1_2mm_brain.hdr'];
    
    write_brainMap(brainVol_sr, userOptions, writeOpts)
    
    
end
cd(returnHere);