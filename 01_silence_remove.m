[x,fs] = audioread('female_zhongjin.wav');
x = mean(x,2);
x = x(1:0.25*fs,1);


    
    curPos = 1;
    frameLength = 0.00005; % Change this value to smaller if short silence regions
    windowLength = round(frameLength*fs);
    step = round(frameLength*fs);
    L = length(x);
    numOfFrames = floor((L-windowLength)/step) + 1;
    E = zeros(numOfFrames,1);
    for i=1:numOfFrames+1
        window = x(curPos:curPos+windowLength-1);
        E(i) = (1/(windowLength)) * sum(abs(window.^2));
        windowFrames(:,i) = window;
        if i == numOfFrames
            curPos = L-windowLength+1;
        else
            curPos = curPos + step;
        end
    end
        
    
    
    E(E == 0) = eps;
    figure();plot(E)
    
    Energy_max = max(E);
    Energy_min = min(E);
    
    T_1 = Energy_min*(1 + 2*log10(Energy_max/Energy_min));
    
    E_biggerT_1 = E(E>T_1);
    SL = sum(E_biggerT_1)/length(E_biggerT_1);
    T_2 = T_1 + 0.25*(SL-T_1);
    
    checkStart = 1;
    start_pos = 1;
    end_pos = size(windowFrames,1);
    for i=1:length(E)-1
        % Variable checkStart only changes if we look at the start or end of
        % the speech
        if checkStart == 1
            if E(i) > T_1 && E(i+1) > T_2
                startFrame = i; % Store which energy frame is bigger than T_1
                checkStart = 0;
                iter_start = i;
            end
        end
        if checkStart == 0
            if E(i)< T_2 && E(i+1) < T_1
                endFrame = i+1; % Store which energy frame is smaller than T_1
                keptFrames = windowFrames(:,startFrame:endFrame); % Store signal frames associated with the energy
                checkStart = 1;
                for l = 1:size(keptFrames,2) % Add frames to a new signal that only contains speech/music
                    newSig(start_pos:end_pos,1) = keptFrames(:,l);
                    start_pos = start_pos + size(windowFrames,1);
                    end_pos = start_pos + size(windowFrames,1)-1;
                end
                iter_end = i;
            end
        end
        
    end
    
    figure();subplot(2,1,1);plot(x); title('Original signal with silence')
    subplot(2,1,2);  plot(newSig); title('Original signal without silence')