import numpy as np
from scipy import signal as sl
from spectrum import mtm
# this all works, but i don't know what i did


def df_mtparam(x, nFFT=1024, Fs=2, WinLength=None, nOverlap=None, NW=3, Detrend=None, nTapers=None):
    # x: np array of nchannels by nsamples
    # parsing input to mimic matlab
    # helper function that gets things set up. works and tested with real numbers
    if WinLength is None:
        WinLength = nFFT
    if nOverlap is None:
        nOverlap = float(WinLength) / 2
    if nTapers is None:
        nTapers = 2 * NW - 1

    # Now do some compuatations that are common to all spectrogram functions
    winstep = WinLength - nOverlap
    nChannels = x.shape[1]
    nSamples = x.shape[0]

    print "nChannels: " + str(nChannels) + " nSamples: " + str(nSamples)

    # check for column vector input
    # if nSamples == 1
    #     x = x'
    #     nSamples = size(x, 1)
    #     nChannels = 1
    # end

    # calculate number of FFTChunks per channel
    nFFTChunks = round(((nSamples - WinLength) / winstep))
    # turn this into time, using the sample frequency
    t = winstep/Fs * np.arange(0,(nFFTChunks))

    # set up f and t arrays
    if not np.iscomplex(x).any(): # x purely real
        if nFFT % 2: # nfft odd
            select = np.arange(0, nFFT/2)
        else:
            select = np.arange(0, nFFT/2 + 1)
        nFreqBins = len(select)
    else:
        select = np.arange(0, nFFT)

    f = select*float(Fs)/nFFT

    return x, nFFT, Fs, WinLength, nOverlap, NW, Detrend, nTapers, nChannels, \
           nSamples, nFFTChunks, winstep, select, nFreqBins, f, t


def df_mtchd_JN(x, nFFT=1024, Fs=2, WinLength=None, nOverlap=None, NW=3, Detrend=None, nTapers=None):
    """ df_matchd_JN
     Multitaper Cross-Spectral Density, jacknived estimates and errors
    only meanP(:,1,2), Pupper(:,1,2), Plower(:,1,2) is the correct jack-knife.
    These values are the absolute values of the coherency, to get coherence, these values
    must be squared.
    y is the original cross spectrum without jack-knifing or normalizing to get coherency.
     function A=mtcsd(x,nFFT,Fs,WinLength,nOverlap,NW,Detrend,nTapers)
     x : input time series
     nFFT = number of points of FFT to calculate (default 1024)
     Fs = sampling frequency (default 2)
     WinLength = length of moving window (default is nFFT)
     nOverlap = overlap between successive windows (default is WinLength/2)
     NW = time bandwidth parameter (e.g. 3 or 4), default 3
     nTapers = number of data tapers kept, default 2*NW -1
    I've changed this program to output the magnitude of coherency, or coherence. output yo is yo(f)

     If x is a multicolumn matrix, each column will be treated as a time
     series and you'll get a matrix of cross-spectra out yo(f, Ch1, Ch2)
     NB they are cross-spectra not coherences. If you want coherences use
     mtcohere
    """
    x, nFFT, Fs, WinLength, nOverlap, NW, \
    Detrend, nTapers, nChannels, nSamples, \
    nFFTChunks, winstep, select, nFreqBins, f, t = df_mtparam(x, nFFT=nFFT, Fs=Fs,
                                                              WinLength=WinLength,
                                                              nOverlap=nOverlap,
                                                              NW=NW, Detrend=Detrend, nTapers=nTapers)

    WinLength = int(round(WinLength))
    winstep = int(round(winstep))
    nFFT = int(round(nFFT))

    # calculate Slepian sequences.Tapers is a matrix of size[WinLength, nTapers]

    stP = np.zeros([nFFT, nChannels, nChannels])
    varP = np.zeros([nFFT, nChannels, nChannels])
    Tapers, V = mtm.dpss(int(WinLength), NW, nTapers)
    Periodogram = np.zeros([nFFT, nTapers, nChannels], dtype=complex)  # intermediate FFTs
    Temp1 = np.zeros([nFFT, nTapers], dtype=complex)  # Temps are particular psd or csd values for a frequency and taper
    Temp2 = np.zeros([nFFT, nTapers], dtype=complex)
    Temp3 = np.zeros([nFFT, nTapers], dtype=complex)
    eJ = np.zeros([nFFT, 1], dtype=complex)
    JN = np.zeros([nFFTChunks, nFFT, nChannels, nChannels], dtype=complex)
    # jackknifed cross - spectral - densities or csd.Note: JN(.,., 1, 1) is the power - spectral - density of time
    # series 1 and JN(.,., 2, 2) is the psd of time series 2. Half - way through this codeJN(.,., 1, 2)
    # ceases to be the csd of 1 and 2 and becomes the abs coherency of 1 and 2.
    y = np.zeros([nFFT, nChannels, nChannels], dtype=complex) # output array for csd
    Py = np.zeros([nFFT, nChannels, nChannels]) # output array for psd's

    # New super duper vectorized alogirthm compute tapered periodogram with FFT
    # This involves lots of wrangling with multidimensional arrays.

    TaperingArray = np.tile(Tapers[:,:,np.newaxis], [1, 1, nChannels])
    for j in np.arange(0, nFFTChunks, dtype=int):
        Segment = x[(j) * winstep + np.arange(0,WinLength),:]
        if Detrend is not None:
            Segment = sl.detrend(Segment, Detrend)

        SegmentsArray = np.transpose(np.tile(Segment[:,:,np.newaxis], [1, 1, nTapers]), [0, 2, 1])
        TaperedSegments = TaperingArray* SegmentsArray

        Periodogram[:,:,:] = np.fft.fft(TaperedSegments, nFFT, axis=0)

        # % Now make cross - products of them to fill cross - spectrum matrix
        for Ch1 in np.arange(0,nChannels):
            for Ch2 in np.arange(Ch1, nChannels): # don't compute cross-spectra twice
                # Temp1 = Periodogram[:,:, Ch1]
                # Temp2 = Periodogram[:,:, Ch2]
                # Temp2 = np.conj(Temp2)
                # Temp3 = Temp1 .* Temp2
                Temp3 = Periodogram[:,:,Ch1] * np.conj(Periodogram[:,:,Ch2])

                # eJ and eJ2 are the sum over all the tapers.
                eJ = np.sum(Temp3, 1) / float(nTapers)
                JN[j,:, Ch1, Ch2] = eJ # Here it is just the cross - power for one particular chunk.
                y[:, Ch1, Ch2]= y[:, Ch1, Ch2] + eJ # y is the sum of the cross - power

    # % now fill other half of matrix with complex conjugate
    for Ch1 in np.arange(0,nChannels):
        for Ch2 in np.arange(Ch1+1,nChannels): # don't compute cross-spectra twice
            y[:, Ch2, Ch1] = y[:, Ch1, Ch2]
            Py[:, Ch1, Ch2] = np.arctanh(abs(y[:, Ch1, Ch2] / np.sqrt(abs(y[:, Ch1, Ch1])*abs(y[:, Ch2, Ch2]))))


    for j in np.arange(0, nFFTChunks):
        JN[j,:,:,:] = abs(y - JN[j,:,:,:]) #% This is wher it becomes % the JN % quantity % (the delete one)
        for Ch1 in np.arange(0, nChannels):
            for Ch2 in np.arange(Ch1+1, nChannels):
                # % Calculate the transformed coherence
                JN[j,:, Ch1, Ch2] = np.arctanh(np.real(JN[j,:, Ch1, Ch2])/ np.sqrt(abs(JN[j,:, Ch1, Ch1])
                                                                                   *abs(JN[j,:, Ch2, Ch2])))
    # % Obtain the pseudo values
                JN[j,:, Ch1, Ch2] = nFFTChunks * Py[:, Ch1, Ch2].T - (nFFTChunks-1)*JN[j,:, Ch1,Ch2]

    meanP = np.mean(JN, axis=0)
    for Ch1 in np.arange(0, nChannels):
        for Ch2 in np.arange(Ch1, nChannels):
            varP[:, Ch1, Ch2] = (1 / nFFTChunks) * np.var(JN[:,:, Ch1, Ch2], axis=0)


    # % upper and lower bounds will be 2 standard deviations away.
    stP = np.sqrt(varP)

    Pupper = meanP + 2 * stP
    Plower = meanP - 2 * stP
    meanP = np.tanh(meanP)
    Pupper = np.tanh(Pupper)
    Plower = np.tanh(Plower)

    select.tolist()
    # % set up f array
    meanP = meanP[select.astype(int),:,:]
    Pupper = Pupper[select.astype(int),:,:]
    Plower = Plower[select.astype(int),:,:]
    y = y[select.astype(int),:,:]

    fo = (select)*float(Fs)/nFFT

    # % we've now done the computation.  the rest of this code is stolen from specgram and just deals
    # with the output stage
    #
    # if nargout == 0
    #     % take abs, and plot results
    #     newplot
    #     for Ch1=1:nChannels,
    #     for Ch2 = 1:nChannels
    #     subplot(nChannels, nChannels, Ch1 + (Ch2 - 1) * nChannels)
    #     plot(f, 20 * log10(abs(y(:, Ch1, Ch2))+eps))
    #     grid
    #     on
    #     if (Ch1 == Ch2)
    #         ylabel('psd (dB)')
    #     else
    #         ylabel('csd (dB)')
    #     end
    #     xlabel('Frequency')
    # end
    # end
    # end
    return y, fo, meanP, Pupper, Plower, stP


def compute_coherence_mean(modelResponse, psth, sampleRate, freqCutoff=-1, windowSize=.5):
    """
    %   Input:
    %       modelResponse: time series model response, vector
    %       psth: time series actual PSTH, should be same length as response, vector
    %       sampleRate: sample rate of PSTH and model response (should be same)
    %       freqCutoff: only return info for frequencies less than this (optional), default -1
    %       windowSize: length in seconds of segments to take FFT of
    %                   and average across, default .5: 500ms, 2Hz and up
    %
    %   Output:
    %
    %       cStruct: structure containing info and coherence values
    %           .f: frequencies in Hz at which coherence was computed
    %           .c: mean coherence at each frequency
    %           .cUpper: upper bound of coherence at each frequency (from
    %             jacknife)
    %           .cLower: lower bound of coherence at each frequency (from
    %             jacknife)
    %           .info: normal mutual information of mean coherence (see eq. 4
    %             of Hsu et. al)
    %           .infoUpper: upper bound of normal mutual information
    %           .infoLower: lower bound of normal mutual information
    """

    # put psths in matrix for mtchd_JN
    if len(modelResponse) != len(psth):
        minLen = min(len(modelResponse), len(psth))
        modelResponse = modelResponse[1:minLen]
        psth = psth[1:minLen]

    x = np.squeeze(np.dstack([modelResponse, psth]))

    # % % compute  # of time bins per FFT segment
    minFreq = round(1 / windowSize)

    numTimeBin = round(sampleRate * windowSize)

    # % % get default parameter values
    vargs = {'x': x, 'nFFT': numTimeBin,'Fs': sampleRate}
    x, nFFT, Fs, WinLength, nOverlap, NW, Detrend, nTapers, nChannels, \
    nSamples, nFFTChunks, winstep, select, nFreqBins, f, t = df_mtparam(**vargs)

    # % % compute jacknifed coherence
    y, fpxy, cxyo, cxyo_u, cxyo_l, stP = df_mtchd_JN(x, nFFT, Fs, WinLength, nOverlap, NW, Detrend, nTapers)

    # % % normalize coherencies
    # cStruct = struct
    # cStruct.f = fpxy
    f = fpxy
    # cStruct.c = cxyo(:, 1, 2).^ 2
    c = cxyo[:,0,1]**2
    # cStruct.cUpper = cxyo_u(:, 1, 2).^ 2
    cUpper = cxyo_u[:,0,1]**2
    # clo = cxyo_l(:, 1, 2)
    clo = cxyo_l[:,0,1]
    # closgn = sign(real(clo))
    closgn = np.sign(np.real(clo))
    # cStruct.cLower = (clo. ^ 2). * closgn # cxyo_l can be negative, multiply by sign after squaring
    cLower = (clo** 2) * closgn
    # restrict frequencies analyzed to the requested cutoff and minimum frequency given the window size
    if freqCutoff != -1:
        findx = np.where(f < freqCutoff)
        eindx = np.max(findx)
        indx = np.arange(1, eindx)
        f = f[indx]
        c = c[indx]
        cUpper = cUpper[indx]
        cLower = cLower[indx]
        # cStruct.f = cStruct.f(indx)
        # cStruct.c = cStruct.c(indx)
        # cStruct.cUpper = cStruct.cUpper(indx)
        # cStruct.cLower = cStruct.cLower(indx)


    if minFreq > 0:
        findx = np.where(f >= minFreq)
        sindx = np.min(findx)
        f = f[np.arange(sindx,len(f))]
        c = c[np.arange(sindx,len(f))]
        cUpper = cUpper[np.arange(sindx,len(f))]
        cLower = cLower[np.arange(sindx,len(f))]

    # % % compute information by integrating log of 1 - coherence
    df = f[1] - f[0]
    minFreq = minFreq
    info = -df * np.sum(np.log2(1 - c))
    infoUpper = -df * np.sum(np.log2(1 - cUpper))
    infoLower = -df * np.sum(np.log2(1 - cLower))
    return f, c, cUpper, cLower, info, infoUpper, infoLower


def compute_coherence_full(modelResponse, psth, psthHalf1, psthHalf2, sampleRate, numTrials,
                           freqCutoff=-1, windowSize=.5):
    # % compute coherence between model response and mean PSTH
    fMM, cMM, cUpperMM, cLowerMM, infoMM, infoUpperMM, infoLowerMM = compute_coherence_mean(modelResponse, psth,
                                                                                            sampleRate,
                                                                                            freqCutoff, windowSize)
    # compute coherence between two halves of PSTH
    fH, cH, cUpperH, cLowerH, infoH, infoUpperH, infoLowerH = compute_coherence_mean(psthHalf1, psthHalf2,
                                                                                     sampleRate, freqCutoff, windowSize)
    # % compute normalized(single spike) expected coherences(Eq. 8 of Hsu et al), these are upper bounds that a perfect
    # model can achieve
    fMB = fH
    cMB = cH
    index = np.where(cH != 0)
    kdown = (-numTrials + numTrials * np.sqrt(1 / cH[index])) / 2
    cMB[index] = 1 / (kdown + 1) # not sure about this one
    # working here
    cUpperMB = cUpperH
    index = np.where(cUpperH != 0)
    kdown = (-numTrials + numTrials * np.sqrt(1 / cUpperH[index])) / 2
    cUpperMB[index] = 1 / (kdown + 1)

    cLowerMB = cLowerH
    index = np.where(cLowerH != 0)
    kdown = (-numTrials + numTrials * np.sqrt(1 / cLowerH[index])) / 2
    cLowerMB[index] = 1 / (kdown + 1)

    # % % compute coherences between a single trial and the model response, % % corresponds to Eq 11 in Hsu et al \
    # and respresents how good the model is
    fSS = fMM
    cSS = cMM
    index = np.where(cMM != 0)
    chval = cH[index]
    rhs = (1 + np.sqrt(1 / chval)) / (-numTrials + numTrials * np.sqrt(1 / chval) + 2)
    # % rhs of Eq 11 in Hsu et.al
    cSS[index] = cMM[index] * rhs

    cUpperSS = cUpperMM
    index = np.where(cUpperMM != 0)
    chval = cUpperH[index]
    rhs = (1 + np.sqrt(1 / chval)) / (-numTrials + numTrials * np.sqrt(1 / chval) + 2)
    # % rhs of Eq 11 in Hsu et.al
    cUpperSS[index] = cUpperMM[index] * rhs

    cLowerSS = cLowerMM
    index = np.where(cLowerMM != 0)
    chval = cLowerH[index]
    rhs = (1 + np.sqrt(1 / chval)) / (-numTrials + numTrials * np.sqrt(1 / chval) + 2)
    # % rhs of Eq 11 in Hsu et.al
    cLowerSS[index] = cLowerMM[index] * rhs

    # % % compute information values
    df = fSS[1] - fSS[0]
    infoSS = -df * np.sum(np.log2(1 - cSS))
    infoUpperSS = -df * np.sum(np.log2(1 - cUpperSS))
    infoLowerSS = -df * np.sum(np.log2(1 - cLowerSS))

    df = fMB[1] - fMB[0]
    infoMB = -df * np.sum(np.log2(1 - cMB))
    infoUpperMB = -df * np.sum(np.log2(1 - cUpperMB))
    infoLowerMB = -df * np.sum(np.log2(1 - cLowerMB))

    return fMB, cMB, cUpperMB, cLowerMB, infoMB, infoUpperMB, infoLowerMB, \
           fSS, cSS, cUpperSS, cLowerSS, infoSS, infoUpperSS, infoLowerSS