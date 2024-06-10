% Read in audio file
[x,fs] = audioread('female.wav');

% Start splitting the signals into frames and compute the energy of each
% frame
curPos = 1;
frameLength = 0.00005; % Change this frame length depending on the signal you analyse
windowLength = round(frameLength*fs); % Length of frame in seconds
step = round(frameLength*fs); % Length of step size in seconds
L = length(x);
numOfFrames = floor((L-windowLength)/step) + 1; % Calculate how many frames there will be 

E = zeros(numOfFrames,1);
for i=1:numOfFrames % Look at every frame
    window = x(curPos:curPos+windowLength-1); % Take only the data from the frame from original signal x
    E(i) = (1/(windowLength)) * sum(abs(window.^2)); % Calculate energy
    windowFrames(:,i) = window; % Store the frame associated with the energy
    curPos = curPos + step; % Go to the next frame
end

% Replace at this stage all zeros from the Energy vector with a very small
% number
E(E == 0) = eps;
figure();plot(E)
% Calculate thresholds based on the paper sent
Energy_max = max(E);
Energy_min = min(E);
% Define thresholds
T_1 = Energy_min*(1 + 2*log10(Energy_max/Energy_min));

E_biggerT_1 = E(E>T_1); % Store only the energy that is bigger than T_1
SL = sum(E_biggerT_1)/length(E_biggerT_1);
T_2 = T_1 + 0.25*(SL-T_1);


% THis part decides what parts of the signal are speech/music and what is
% silence
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
        end
    end
    if checkStart == 0
        if E(i)< T_2 && E(i+1) < T_1
            endFrame = i+1; % Store which energy frame is smaller than T_1 
            keptFrames = windowFrames(:,startFrame:endFrame); % Store signal frames associated with the energy
            checkStart = 1;
            for l = 1:size(keptFrames,2) % Add frames to a new signal that only contains speech/music
                newSig(start_pos:end_pos) = keptFrames(:,l);
                start_pos = start_pos + size(windowFrames,1);
                end_pos = start_pos + size(windowFrames,1)-1;
            end
        end
    end
    
end
% Plot original signal with and without silence
figure();subplot(2,1,1);plot(x);xlim([0 800000]); title('Original signal with silence')
subplot(2,1,2); plot(newSig);xlim([0 800000]); title('Original signal without silence') 
% For the energy calculations, a shorter window frame helps detecting all
% silence parts. For the FFT calculations, usually a higher window length
% is better for the resolution. 
currentPos = 1;
nL=length(newSig);
wlen = 1024; % FFT length 
w = rectwin(wlen);
step = round(frameLength*fs); % Length of step size in seconds
new_numOfFrames = floor((nL-wlen)/step) + 1; % Calculate how many frames there will be 

newSig = newSig'; % This was the problem before when you had the same number in the averaged power spectrum. The vector orinetation was 1xnL instead of nLx1.
Power_spectrum = zeros(new_numOfFrames,1);
for i=1:new_numOfFrames % Look at every frame
    new_window = newSig(currentPos:currentPos+wlen-1).*w; % Take only the data from the frame from new signal
    y=fft(new_window,wlen);
    sq=abs(y);
    power=sq.^2;
    Power_spectrum_frame(:,i)=power;
    currentPos = currentPos + step; % Go to the next frame
end
Power2 = mean(Power_spectrum_frame,2);
% Generate a frequency vector
frequency_vector = 0:1:wlen-1;
frequency_vector = frequency_vector*fs/wlen;

% Plotting
figure();loglog(frequency_vector(1:end/2+1),Power2(1:end/2+1)) % Plot the averaged power spectrum over frequency
xlabel('Frequency [Hz] - log')
ylabel('Power spectrum')
title('Averaged Power Spectrum')



