function []=run_fmri_searchlight()
% Recipe_fMRI_searchlight
%
% Cai Wingfield 11-2009, 2-2010, 3-2010, 8-2010
%__________________________________________________________________________
% Copyright (C) 2010 Medical Research Council

%%%%%%%%%%%%%%%%%%%%
%% Initialisation %%
%%%%%%%%%%%%%%%%%%%%
cd('/space/raid5/data/monti/Analysis/LanguageMVPA/RSA')
analysisType='RSA';
toolboxRoot = '/space/raid5/data/monti/Analysis/LanguageMVPA/RSA/code'; addpath(genpath(toolboxRoot)); % Catch sight of the toolbox code

% MAKE SURE THERE IS ONLY ONE ROI
userOptions = defineUserOptions_searchlight();
userOptions.analysisType=analysisType;

%%%%%%%%%%%%%%%%%%%%%%
%% Data preparation %%
%%%%%%%%%%%%%%%%%%%%%%

fullBrainVols = fMRIDataPreparation(betaCorrespondence_LMVPA(), userOptions);
binaryMasks_nS = fMRIMaskPreparation(userOptions);

%%%%%%%%%%%%%%%%%%%%%
%% RDM calculation %%
%%%%%%%%%%%%%%%%%%%%%
if strcmp(userOptions.analysisType, 'SVM')
    models=makeLabels_LMVPA();
else
    models = constructModelRDMs(modelRDMs_searchlight(), userOptions);
end
%%%%%%%%%%%%%%%%%
%% Searchlight %%
%%%%%%%%%%%%%%%%%

fMRISearchlight_jeff(fullBrainVols, binaryMasks_nS, models, betaCorrespondence_LMVPA(), userOptions);

searchlightInference(userOptions);
end