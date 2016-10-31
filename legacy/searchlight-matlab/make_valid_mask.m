function [mappingMask_request_INDs, nVox_mappingMask_request, volIND2YspaceIND] = make_valid_mask(inputDataMask, localOptions)
    
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
end