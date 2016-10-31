function [varargout] = SVM_across(fullBrainVols, binaryMasks_nS, models, betaCorrespondence, userOptions)
%
% fMRISearchlight is a function which takes some full brain volumes of data,
% some binary masks and some models and perfoms a searchlight in the data within
% each mask, matching to each of the models.  Saved are native-space r-maps for
% each model.
%
% [rMaps_sS, maskedSmoothedRMaps_sS, searchlightRDMs[, rMaps_nS, nMaps_nS] =]
%                                 fMRISearchlight(fullBrainVols,
%                                                 binaryMasks_nS,
%                                                 models,
%                                                 betaCorrespondence,
%                                                 userOptions)
%
%       fullBrainVols --- The unmasked beta (or t) images.
%               fullBrainVols.(subject) is a [nVoxel nCondition nSession]-sized
%               matrix. The order of the voxels is that given by reshape or (:).
%
%        binaryMasks_nS --- The native- (subject-) space masks.
%               binaryMasks_nS.(subject).(mask) is a [x y z]-sized binary matrix
%               (the same size as the native-space 3D beta images.
%
%        models --- A stack of model RDMs in a structure.
%               models is a [1 nModels] structure with fields:
%                       RDM
%                       name
%                       color
%
%        betaCorrespondence --- The array of beta filenames.
%               betas(condition, session).identifier is a string which referrs
%               to the filename (not including path) of the SPM beta image. (Or,
%               if not using SPM, just something, as it's used to determine the
%               number of conditions and sessions.)
%               Alternatively, this can be the string 'SPM', in which case the
%               SPM metadata will be used to infer this information, provided
%               that userOptions.conditionLabels is set, and the condition
%               labels are the same as those used in SPM.
%
%
%        userOptions --- The options struct.
%                userOptions.analysisName
%                        A string which is prepended to the saved files.
%                userOptions.rootPath
%                        A string describing the root path where files will be
%                        saved (inside created directories).
%                userOptions.subjectNames
%                        A cell array containing strings identifying the subject
%                        names. Defaults to the fieldnames in fullBrainVols.
%                userOptions.maskNames
%                        A cell array containing strings identifying the mask
%                        names. Defaults to the fieldnames of the first subject
%                        of binaryMasks_nS.
%                userOptions.voxelSize
%                        A tripple consisting of the [x y z] dimensions of each
%                        voxel in mm.
%                userOptions.structuralsPath
%                        A string which contains the absolute path to the
%                        location of the structural images and the normalisation
%                        warp definition file. It can contain the following
%                        wildcards which would be replaced as indicated:
%                                [[subjectName]]
%                                        To be replaced with the name of each
%                                        subject where appropriate.
%
% The following files are saved by this function:
%        userOptions.rootPath/Maps/
%                userOptions.analysisName_fMRISearchlight_Maps.mat
%                        Contains the searchlight statistical maps in struct so
%                        that rMaps_nS.(modelName).(subject).(maskName),
%                        rMaps_sS.(modelName).(subject).(maskName),
%                        maskedSmoothedRMaps_sS.(modelName).(subject).(maskName)
%                        and nMaps_nS.(modelName).(subject).(maskName) contain
%                        the appropriate data.
%        userOptions.rootPath/RDMs/
%                userOptions.analysisName_fMRISearchlight_RDMs.mat
%                        Contains the RDMs for each searchlight so that
%                        searchlightRDMs.(subject)(:, :, x, y, z) is the RDM.
%        userOptions.rootPath/Details/
%                userOptions.analysisName_fMRISearchlight_Details.mat
%                        Contains the userOptions for this execution of the
%                        function and a timestamp.
%
% Cai Wingfield 2-2010, 3-2010
%__________________________________________________________________________
% Copyright (C) 2010 Medical Research Council


returnHere = pwd; % We'll come back here later

%% Set defaults and check options struct
if ~isfield(userOptions, 'analysisName'), error('fMRISearchlight:NoAnalysisName', 'analysisName must be set. See help'); end%if
if ~isfield(userOptions, 'rootPath'), error('fMRISearchlight:NoRootPath', 'rootPath must be set. See help'); end%if
userOptions = setIfUnset(userOptions, 'subjectNames', fieldnames(fullBrainVols));
userOptions = setIfUnset(userOptions, 'maskNames', fieldnames(binaryMasks_nS.(userOptions.subjectNames{1})));
if ~isfield(userOptions, 'voxelSize'), error('fMRISearchlight:NoVoxelSize', 'voxelSize must be set. See help'); end%if

% The analysisName will be used to label the files which are eventually saved.
mapsFilename = [userOptions.analysisName,'_across_', userOptions.analysisType '_fMRISearchlight_Maps.mat'];
RDMsFilename = [userOptions.analysisName,'_across_', userOptions.analysisType '_fMRISearchlight_RDMs.mat'];
DetailsFilename = [userOptions.analysisName,'_across_', userOptions.analysisType '_fMRISearchlight_Details.mat'];

promptOptions.functionCaller = 'fMRISearchlight';
promptOptions.defaultResponse = 'S';
promptOptions.checkFiles(1).address = fullfile(userOptions.rootPath, 'Maps',userOptions.analysisType, mapsFilename);
promptOptions.checkFiles(2).address = fullfile(userOptions.rootPath, 'Details', DetailsFilename);

overwriteFlag = overwritePrompt(userOptions, promptOptions);

if overwriteFlag
    
    % Data
    nSubjects = numel(userOptions.subjectNames);
    nMasks = numel(userOptions.maskNames);
    
    searchlightOptions.monitor = false;
    searchlightOptions.fisher = true;
    
    warpFlags.interp = 1;
    warpFlags.wrap = [0 0 0];
    warpFlags.vox = userOptions.voxelSize; % [3 3 3.75]
    warpFlags.bb = [-78 -112 -50; 78 76 85];
    warpFlags.preserve = 0;
    
    fprintf('Shining RSA searchlights...\n');
    
    for subjectNumber = 1:nSubjects % and for each subject...
        
        tic;%1
        
        fprintf(['\t...in the brain of subject ' num2str(subjectNumber) ' of ' num2str(nSubjects)]);
        
        % Figure out which subject this is
        subject = userOptions.subjectNames{subjectNumber};
        
        if ischar(betaCorrespondence) && strcmpi(betaCorrespondence, 'SPM')
            betas = getDataFromSPM(userOptions);
        else
            betas = betaCorrespondence;
        end%if:SPM
        
        searchlightOptions.nSessions = size(betas, 1);
        searchlightOptions.nConditions = size(betas, 2);
        
        readFile = replaceWildcards(userOptions.betaPath, '[[subjectName]]', subject, '[[betaIdentifier]]', betas(1,1).identifier);
        subjectMetadataStruct = spm_vol(readFile);
        %		subjectMetadataStruct = spawnSPMStruct;
        
        for maskNumber = 1:nMasks % For each mask...
            
            % Get the mask
            maskName = userOptions.maskNames{maskNumber};
            mask = binaryMasks_nS.(subject).(maskName);
            
            % Full brain data volume to perform searchlight on
            singleSubjectVols = fullBrainVols.(subject);
            trainingVols=fullBrainVols;
            trainingVols=rmfield(trainingVols, subject);
            trainSubjs=fieldnames(trainingVols);
            trainVols=[];
            for trainSubNumber=1:length(trainSubjs)
                trainSub=trainSubjs{trainSubNumber};
                trainVols=[trainVols trainingVols.(trainSub)];
            end 
                    
            % Do the searchlight! ZOMG, this takes a while...
            if strcmp(userOptions.analysisType, 'SVM')
                [rs, ps, ns] = searchlightMapping_fMRI(singleSubjectVols, trainVols, models, mask, userOptions, searchlightOptions); % ps are from linear correlation p-values, and so aren't too useful here.
            else
                [rs, ps, ns, searchlightRDMs.(subject)] = searchlightMapping_fMRI(singleSubjectVols, models, mask, userOptions, searchlightOptions); % ps are from linear correlation p-values, and so aren't too useful here.
            end
            nMaps_nS.(subject).(maskName) = ns(:,:,:); % How many voxels contributed to the searchlight centred at each point. (Those with n==1 are excluded because the results aren't multivariate.)
           %% Save n-map version
                
                % Write the native-space n-map to a file
                nMapMetadataStruct_nS = subjectMetadataStruct;
                nMapMetadataStruct_nS.fname = fullfile(userOptions.rootPath, 'Maps', [userOptions.analysisName '_nMap_' maskName '_' subject '.img']);
                nMapMetadataStruct_nS.descrip =  'N-map';
                nMapMetadataStruct_nS.dim = size(nMaps_nS.(subject).(maskName));
                
                gotoDir(userOptions.rootPath, ['Maps/' userOptions.analysisType]);
                
                spm_write_vol(nMapMetadataStruct_nS, nMaps_nS.(subject).(maskName));
            for modelNumber = 1:numel(models)
                %JEFF EDIT
                % 				modelName = spacesToUnderscores(models(modelNumber).name);
                modelName=models(modelNumber).name;
                % Store results in indexed volumes
                rMaps_nS.(modelName).(subject).(maskName) = rs(:,:,:,modelNumber); % r-values for correlation with each model
                pMaps_nS.(modelName).(subject).(maskName) = ps(:,:,:,modelNumber); % r-values for correlation with each model
                
                %% Save native space version
                
                % Write the native-space r-map to a file
                rMapMetadataStruct_nS = subjectMetadataStruct;
                rMapMetadataStruct_nS.fname = fullfile(userOptions.rootPath, 'Maps', [userOptions.analysisName '_rMap_' maskName '_' modelName '_' subject '.img']);
                rMapMetadataStruct_nS.descrip =  'R-map';
                rMapMetadataStruct_nS.dim = size(rMaps_nS.(modelName).(subject).(maskName));
                
                gotoDir(userOptions.rootPath, ['Maps/' userOptions.analysisType]);
                
                spm_write_vol(rMapMetadataStruct_nS, rMaps_nS.(modelName).(subject).(maskName));
                
                 modelName=models(modelNumber).name;
                % Store results in indexed volumes
                rMaps_nS.(modelName).(subject).(maskName) = rs(:,:,:,modelNumber); % r-values for correlation with each model
                
                % Write the native-space p-map to a file
                pMapMetadataStruct_nS = subjectMetadataStruct;
                pMapMetadataStruct_nS.fname = fullfile(userOptions.rootPath, ['Maps/' userOptions.analysisType], [userOptions.analysisName '_pMap_' maskName '_' modelName '_' subject '.img']);
                pMapMetadataStruct_nS.descrip =  'P-map';
                pMapMetadataStruct_nS.dim = size(pMaps_nS.(modelName).(subject).(maskName));
                
                gotoDir(userOptions.rootPath, ['Maps/' userOptions.analysisType]);
                
                spm_write_vol(pMapMetadataStruct_nS, pMaps_nS.(modelName).(subject).(maskName));
                
                 modelName=models(modelNumber).name;
                % Store results in indexed volumes
                pMaps_nS.(modelName).(subject).(maskName) =ps(:,:,:,modelNumber); % p-values for correlation with each model
                
            end%for:models
            
            clear fullBrainVolumes rs ps ns;
            
            fprintf(':');
            
        end%for:maskNumber
        
        t = toc;%1
        fprintf([' [' num2str(ceil(t)) 's]\n']);
        
    end%for:subjectNumber
    
    %% Save relevant info
    
    timeStamp = datestr(now);
    
    fprintf(['Saving searchlight maps to ' fullfile(userOptions.rootPath, 'Maps', mapsFilename) '\n']);
    gotoDir(userOptions.rootPath, ['Maps/' userOptions.analysisType]);
    % 	if isfield(userOptions, 'structuralsPath')
    % 		save(mapsFilename, 'rMaps_nS', 'rMaps_sS', 'maskedSmoothedRMaps_sS', 'nMaps_nS');
    % 	else
    save(mapsFilename, 'rMaps_nS', 'nMaps_nS', 'pMaps_nS', '-v7.3');
    % 	end%if
    if strcmp(userOptions.analysisType, 'RSA')
        fprintf(['Saving RDMs to ' fullfile(userOptions.rootPath, 'RDMs', RDMsFilename) '\n']);
        gotoDir(userOptions.rootPath, 'RDMs');
        save(RDMsFilename, 'searchlightRDMs', '-v7.3');
    end
    fprintf(['Saving Details to ' fullfile(userOptions.rootPath, 'Details', DetailsFilename) '\n']);
    gotoDir(userOptions.rootPath, 'Details');
    save(DetailsFilename, 'timeStamp', 'userOptions');
    
else
    fprintf(['Loading previously saved maps from ' fullfile(userOptions.rootPath, 'Maps', mapsFilename) '...\n']);
    load(fullfile(userOptions.rootPath, 'Maps', userOptions.analysisType, mapsFilename));
    fprintf(['Loading previously saved RDMs from ' fullfile(userOptions.rootPath, 'RDMs', RDMsFilename) '...\n']);
    load(fullfile(userOptions.rootPath, 'RDMs',userOptions.analysisType, RDMsFilename));
end%if

if nargout == 3
    varargout{1} = rMaps_sS;
    varargout{2} = maskedSmoothedRMaps_sS;
    varargout{3} = searchlightRDMs;
elseif nargout == 5
    varargout{1} = rMaps_sS;
    varargout{2} = maskedSmoothedRMaps_sS;
    varargout{3} = searchlightRDMs;
    varargout{4} = rMaps_nS;
    varargout{5} = nMaps_nS;
elseif nargout > 0
    error('0, 3 or 5 arguments out, please.');
end%if:nargout

cd(returnHere); % And go back to where you started

end%function


%%%%%%%%%%%%%%%%%%%
%% Sub functions %%
%%%%%%%%%%%%%%%%%%%


function [smm_rs, smm_ps, n, searchlightRDMs] = searchlightMapping_fMRI(testBrainVolumes, trainVols, models, mask, userOptions, localOptions)

% ARGUMENTS
% testBrainVolumes	A voxel x condition x session matrix of activity
% 				patterns.
%
% models		A struct of model RDMs.
%
% mask     		A 3d or 4d mask to perform the searchlight in.
%
% userOptions and localOptions
%
% RETURN VALUES
% smm_rs        4D array of 3D maps (x by y by z by model index) of
%               correlations between the searchlight pattern similarity
%               matrix and each of the model similarity matrices.
%
% smm_ps        4D array of 3D maps (x by y by z by model index) of p
%               values computed for each corresponding entry of smm_rs.
%
% n             an array of the same dimensions as the volume, which
%               indicates for each position how many voxels contributed
%               data to the corresponding values of the infomaps.
%               this is the number of searchlight voxels, except at the
%               fringes, where the searchlight may illuminate voxels
%               outside the input-data mask or voxel with all-zero
%               time-courses (as can arise from head-motion correction).
%
% mappingMask_actual
%               3D mask indicating locations for which valid searchlight
%               statistics have been computed.
%
% Based on Niko Kriegeskorte's searchlightMapping_RDMs.m
%
% Additions by Cai Wingfield 2-2010:
% 	- Now skips points in the searchlight where there's only one voxel inside.
% 	- Now takes a userOptions struct for the input parameters.

localOptions = setIfUnset(localOptions, 'averageSessions', true);

%% Figure out whether to average over sessions or not
if localOptions.averageSessions
    for sessionNumber = 1:size(testBrainVolumes,3)
        thisSessionId = ['s' num2str(sessionNumber)];
        t_patsPerSession.(thisSessionId) = testBrainVolumes(:,:,sessionNumber)';
    end%for:sessionNumber
else
    justThisSession = 1;
    t_pats = testBrainVolumes(:,:,justThisSession)';
    
    fprintf(['\nYou have selected not to average over sessions.\n         Only session number ' num2str(justThisSession) ' will be used.\n']);
    
end%if

%% Get parameters
voxSize_mm = userOptions.voxelSize;
searchlightRad_mm = userOptions.searchlightRadius;
monitor = localOptions.monitor;
nConditions = size(testBrainVolumes, 2);

clear testBrainVolumes;

% Prepare models
if strcmp(userOptions.analysisType, 'SVM')
    modelRDMs_ltv=models';
else
    modelRDMs_ltv = permute(unwrapRDMs(vectorizeRDMs(models)), [3 2 1]);
end

% Prepare masks
mask(isnan(mask)) = 0; % Just in case!
if ndims(mask)==3
    inputDataMask=logical(mask);
    mappingMask_request=logical(mask);
else
    inputDataMask=logical(mask(:,:,:,1));
    mappingMask_request=logical(mask(:,:,:,2));
end

% Check to see if there's more data than mask...
if localOptions.averageSessions
    for sessionNumber = 1:numel(fieldnames(t_patsPerSession))
        thisSessionId = ['s' num2str(sessionNumber)];
        t_patsPerSession.(thisSessionId) = t_patsPerSession.(thisSessionId)(:, inputDataMask(:));
    end%for:sessionNumber
else
    if (size(t_pats,2)>sum(inputDataMask(:)))
        t_pats=t_pats(:,inputDataMask(:));
    end%if
end%if

% Other data
volSize_vox=size(inputDataMask);
nModelRDMs=size(modelRDMs_ltv,1);
rad_vox=searchlightRad_mm./voxSize_mm;
minMargin_vox=floor(rad_vox);


%% create spherical multivariate searchlight
[x,y,z]=meshgrid(-minMargin_vox(1):minMargin_vox(1),-minMargin_vox(2):minMargin_vox(2),-minMargin_vox(3):minMargin_vox(3));
sphere=((x*voxSize_mm(1)).^2+(y*voxSize_mm(2)).^2+(z*voxSize_mm(3)).^2)<=(searchlightRad_mm^2);  % volume with sphere voxels marked 1 and the outside 0
sphereSize_vox=[size(sphere),ones(1,3-ndims(sphere))]; % enforce 3D (matlab stupidly autosqueezes trailing singleton dimensions to 2D, try: ndims(ones(1,1,1)). )

if monitor, figure(50); clf; showVoxObj(sphere); end % show searchlight in 3D

% compute center-relative sphere SUBindices
[sphereSUBx,sphereSUBy,sphereSUBz]=ind2sub(sphereSize_vox,find(sphere)); % (SUB)indices pointing to sphere voxels
sphereSUBs=[sphereSUBx,sphereSUBy,sphereSUBz];
ctrSUB=sphereSize_vox/2+[.5 .5 .5]; % (c)en(t)e(r) position (sphere necessarily has odd number of voxels in each dimension)
ctrRelSphereSUBs=sphereSUBs-ones(size(sphereSUBs,1),1)*ctrSUB; % (c)en(t)e(r)-relative sphere-voxel (SUB)indices

nSearchlightVox=size(sphereSUBs,1);


%% define masks
validInputDataMask=inputDataMask;

if localOptions.averageSessions
    for sessionNumber = 1:numel(fieldnames(t_patsPerSession))
        thisSessionId = ['s' num2str(sessionNumber)];
        sumAbsY=sum(abs(t_patsPerSession.(thisSessionId)),1);
    end%for:sessionNumber
else
    sumAbsY=sum(abs(t_pats),1);
end%if

validYspace_logical= (sumAbsY~=0) & ~isnan(sumAbsY); clear sumAbsY;
validInputDataMask(inputDataMask)=validYspace_logical; % define valid-input-data brain mask

if localOptions.averageSessions
    for sessionNumber = 1:numel(fieldnames(t_patsPerSession))
        thisSessionId = ['s' num2str(sessionNumber)];
        t_patsPerSession.(thisSessionId) = t_patsPerSession.(thisSessionId)(:,validYspace_logical);
        nVox_validInputData=size(t_patsPerSession.(thisSessionId),2);
    end%for:sessionNumber
else
    t_pats=t_pats(:,validYspace_logical); % reduce t_pats to the valid-input-data brain mask
    nVox_validInputData=size(t_pats,2);
end%if

mappingMask_request_INDs=find(mappingMask_request);
nVox_mappingMask_request=length(mappingMask_request_INDs);

if monitor
    disp([num2str(round(nVox_mappingMask_request/prod(volSize_vox)*10000)/100),'% of the cuboid volume requested to be mapped.']);
    disp([num2str(round(nVox_validInputData/prod(volSize_vox)*10000)/100),'% of the cuboid volume to be used as input data.']);
    disp([num2str(nVox_validInputData),' of ',num2str(sum(inputDataMask(:))),' declared input-data voxels included in the analysis.']);
end

volIND2YspaceIND=nan(volSize_vox);
volIND2YspaceIND(validInputDataMask)=1:nVox_validInputData;

% n voxels contributing to infobased t at each location
n=nan(volSize_vox);

%% similarity-graph-map the volume with the searchlight
smm_bestModel=nan(volSize_vox);
smm_ps=nan([volSize_vox,nModelRDMs]);
smm_rs=nan([volSize_vox,nModelRDMs]);
if strcmp(userOptions.analysisType, 'RSA')
searchlightRDMs = nan([nConditions, nConditions, volSize_vox]);
end
if monitor
    h_progressMonitor=progressMonitor(1, nVox_mappingMask_request,  'Similarity-graph-mapping...');
end

%% THE BIG LOOP! %%

for cMappingVoxI=1:nVox_mappingMask_request
    
    if mod(cMappingVoxI,1000)==0
        if monitor
            progressMonitor(cMappingVoxI, nVox_mappingMask_request, 'Searchlight mapping Mahalanobis distance...', h_progressMonitor);
            %                 cMappingVoxI/nVox_mappingMask_request
        else
            fprintf('.');
        end%if
    end%if
%% apply sphere to index    
    [x y z]=ind2sub(volSize_vox,mappingMask_request_INDs(cMappingVoxI));
    
    % compute (sub)indices of (vox)els (c)urrently (ill)uminated by the spherical searchlight
    cIllVoxSUBs=repmat([x,y,z],[size(ctrRelSphereSUBs,1) 1])+ctrRelSphereSUBs;
    
    % exclude out-of-volume voxels
    outOfVolIs=(cIllVoxSUBs(:,1)<1 | cIllVoxSUBs(:,1)>volSize_vox(1)|...
        cIllVoxSUBs(:,2)<1 | cIllVoxSUBs(:,2)>volSize_vox(2)|...
        cIllVoxSUBs(:,3)<1 | cIllVoxSUBs(:,3)>volSize_vox(3));
    
    cIllVoxSUBs=cIllVoxSUBs(~outOfVolIs,:);
    
    % list of (IND)ices pointing to (vox)els (c)urrently (ill)uminated by the spherical searchlight
    cIllVox_volINDs=sub2ind(volSize_vox,cIllVoxSUBs(:,1),cIllVoxSUBs(:,2),cIllVoxSUBs(:,3));
    
    % restrict searchlight to voxels inside validDataBrainMask
    cIllValidVox_volINDs=cIllVox_volINDs(validInputDataMask(cIllVox_volINDs));
    cIllValidVox_YspaceINDs=volIND2YspaceIND(cIllValidVox_volINDs);
    
    % note how many voxels contributed to this locally multivariate stat
    n(x,y,z)=length(cIllValidVox_YspaceINDs);
    
    if n(x,y,z) < 2, continue; end%if % This stops the function crashing if it accidentally encounters an out-of-brain floating voxel (these can occur if, for example, skull stripping fails)
    %% analysis-Jeff
        if localOptions.averageSessions
            t_pats=zeros(size(t_patsPerSession.s1(:,cIllValidVox_YspaceINDs)));
            for session = 1:localOptions.nSessions
                sessionId = ['s' num2str(session)];
                t_pats = t_pats+t_patsPerSession.(sessionId)(:,cIllValidVox_YspaceINDs);
                t_pats=t_pats / localOptions.nSessions;
            end
        end
 %%%%%%%%%%%%%%%%%%%%%%%%%       
        
        
        trainData=zscore(trainVols');
        testData=zscore(t_pats);

opts=['-s 1 -t 0 -q']; %split data in half, linear SVM, etc.
topts=['-s 1 -t 0 -q'];
%% initialize
StimType=repmat(models(1).label, 1, 16);
Syntax=repmat(models(2).label,1,16);
Verb=repmat(models(3).label, 1, 16);
ActPass=repmat(models(4).label, 1, 16);
RelCan=repmat(models(5).label, 1, 16);
SVerb=repmat(models(6).label, 1, 16);
PVerb=repmat(models(7).label, 1, 16);

rs=zeros(1,9);

L_train=trainData(StimType==1,:);
L_test=testData(models(1).label==1,:);
P_train=trainData(StimType==2,:);
P_test=testData(models(1).label==2,:);
%% classify
% rs(1)=libsvmtrain(StimType',currPattern,opts);
% rs(2) = libsvmtrain(LMSyntax', currPattern, opts);
% rs(3) = libsvmtrain(LSyntax', L_patterns, opts);
% rs(4)=libsvmtrain(MSyntax', M_patterns, opts);
% rs(5)=libsvmtrain(LActPass(LActPass~=0)', L_patterns(LSyntax~=3,:),opts);
% rs(6)=libsvmtrain(MActPass(MActPass~=0)', M_patterns(MSyntax~=3,:),opts);

StimTypeStruct=libsvmtrain(StimType', trainData,topts);
[~, acc, ~] = libsvmpredict(models(1).label',testData, StimTypeStruct);
rs(1)=acc(1);

SyntaxStruct=libsvmtrain(Syntax', L_train,topts);
[~, acc, ~] = libsvmpredict(models(2).label',L_test, SyntaxStruct);
rs(2)=acc(1);

SvPStruct=libsvmtrain(Verb', trainData,topts);
[~, acc, ~] = libsvmpredict(models(3).label',testData, SvPStruct);
rs(3)=acc(1);

SvPStruct=libsvmtrain(ActPass', L_train,topts);
[~, acc, ~] = libsvmpredict(models(4).label',L_test, SvPStruct);
rs(4)=acc(1);

SvPStruct=libsvmtrain(RelCan', L_train,topts);
[~, acc, ~] = libsvmpredict(models(5).label',L_test, SvPStruct);
rs(5)=acc(1);

SvPStruct=libsvmtrain(SVerb', L_train,topts);
[~, acc, ~] = libsvmpredict(models(6).label',L_test, SvPStruct);
rs(6)=acc(1);

SvPStruct=libsvmtrain(PVerb', P_train,topts);
[~, acc, ~] = libsvmpredict(models(7).label',P_test, SvPStruct);
rs(7)=acc(1);


        
        
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%        
            ps=nan;
        

    %    smm_bestModel(x,y,z) = bestModelI;
    smm_ps(x,y,z,:) = ps;
    smm_rs(x,y,z,:) = rs;
    %%
end%for:cMappingVoxI

%% END OF THE BIG LOOP! %%

if monitor
    fprintf('\n');
    close(h_progressMonitor);
end

mappingMask_actual=mappingMask_request;
mappingMask_actual(isnan(sum(smm_rs,4)))=0;

%% visualize
if monitor
    aprox_p_uncorr=0.001;
    singleModel_p_crit=aprox_p_uncorr/nModelRDMs; % conservative assumption model proximities nonoverlapping
    smm_min_p=min(smm_ps,[],4);
    smm_significant=smm_min_p<singleModel_p_crit;
    
    vol=map2vol(mask);
    vol2=map2vol(mask);
    
    colors=[1 0 0
        0 1 0
        0 1 1
        1 1 0
        1 0 1];
    
    for modelRDMI=1:nModelRDMs
        vol=addBinaryMapToVol(vol, smm_significant&(smm_bestModel==modelRDMI), colors(modelRDMI,:));
        % 		vol2=addBinaryMapToVol(vol2, smm_bestModel==modelRDMI, colors(modelRDMI,:));
    end
    
    showVol(vol);
    

    
end%if

end%function
