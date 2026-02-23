%% EMG_Disease_Predict.m
% Use trained EMG model to predict the disease from a new EMG CSV file

clc; clear; close all;

%% 1. Load the trained model --------------------------------------------
modelPath = 'C:\Users\Machine\OneDrive\Desktop\EMG_Disease_CompactModel.mat';
modelPath = char(modelPath);  % ensure MATLAB treats it as a char array

if exist(modelPath, 'file') == 2
    load(modelPath, 'compactMdl', 'featureNames');
    disp('✅ Trained EMG disease classifier loaded successfully.');
else
    error('❌ Model file not found at the given path. Check your modelPath.');
end

%% 2. Load new EMG CSV data ---------------------------------------------
% 👉 Change this path to your new EMG test CSV file
csvFile = 'C:\Users\Machine\OneDrive\Desktop\emg_csv_only1\Brachial_Plexus_Injury.csv';

if ~isfile(csvFile)
    error('❌ CSV file not found at the given path.');
end

disp(['📂 Loaded CSV file: ', csvFile]);

% Read the EMG signal
data = readmatrix(csvFile);
if size(data,2) > 1 && mean(diff(data(:,1)))>0
    emg = data(:,2:end);   % skip time column
else
    emg = data;
end

fs = 1000;  % sampling rate (adjust if different)

%% 3. Preprocess the EMG signal -----------------------------------------
hp = 20; lp = 450;
[b,a] = butter(4,[hp lp]/(fs/2),'bandpass');
emg_filt = filtfilt(b,a,emg);
emg_rect = abs(emg_filt);
window_env = round(0.05*fs);
emg_env = sqrt(movmean(emg_rect.^2,window_env));

%% 4. Kalman filtering ---------------------------------------------------
Q = 1e-4; R = 1e-2;
kalman_est = zeros(size(emg_env));
for ch = 1:size(emg_env,2)
    x_est = emg_env(1,ch); P = 1;
    for k = 2:size(emg_env,1)
        x_pred = x_est;
        P = P + Q;
        K = P/(P + R);
        x_est = x_pred + K*(emg_env(k,ch) - x_pred);
        P = (1 - K)*P;
        kalman_est(k,ch) = x_est;
    end
end

%% 5. Extract same features as training ---------------------------------
winSec = 0.25; stepSec = 0.125;
win = round(winSec*fs);
step = round(stepSec*fs);
starts = 1:step:(size(emg,1)-win);
numWins = length(starts);

featureTbl = [];
for w = 1:numWins
    idx = starts(w):starts(w)+win-1;
    sig = emg_filt(idx,1); % assuming single channel
    rmsVal = rms(sig);
    meanAbs = mean(abs(sig));
    zc = sum(abs(diff(sign(sig))))/2;
    mav = mean(abs(sig));
    [pxx,f] = pwelch(sig,[],[],[],fs);
    specMed = f(find(cumsum(pxx)/sum(pxx) >= 0.5,1));
    kalMean = mean(kalman_est(idx,1));
    featureTbl = [featureTbl; [rmsVal meanAbs zc mav specMed kalMean]];
end
% Define feature names (same as used in training)
featureNames = {'RMS','MeanAbs','ZC','MAV','SpectralMedian','KalmanMean'};

featureTbl = array2table(featureTbl, 'VariableNames', featureNames);

%% 6. Predict using trained model ---------------------------------------
[predictedDisease, score] = predict(compactMdl, featureTbl);

% Get the most frequent predicted class across all windows
finalPrediction = mode(predictedDisease);

% --- Display results neatly ---
meanScores = mean(score,1);
[bestScore, idx] = max(meanScores);

% Convert class names to cell array (fixes the error)
classNames = cellstr(compactMdl.ClassNames);
meanScoresTbl = array2table(meanScores, 'VariableNames', classNames);

fprintf('\n🧠 Predicted Disease: %s\n', string(finalPrediction));
fprintf('Confidence: %.2f%%\n', bestScore*100);

disp('Class probabilities:');
disp(meanScoresTbl);
% --- Save results to CSV ---
outputFile = 'Prediction_Probabilities.csv';
writetable(meanScoresTbl, outputFile);
fprintf('📁 Saved class probabilities to %s\n', outputFile);

