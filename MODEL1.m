%% EMG_Feature_Extraction_All.m
% Automatically extracts EMG features for all disease CSV files in a folder
% and saves labeled feature .mat files for training

clc; clear; close all;

%% 1️⃣ USER SETTINGS -----------------------------------------------------
inputFolder = 'C:\Users\Machine\OneDrive\Desktop\emg_csv_only1';
outputFolder = fullfile(inputFolder, 'EMG_Results');
if ~exist(outputFolder, 'dir')
    mkdir(outputFolder);
end

fs = 1000;     % Sampling frequency (Hz)
hp = 20;       % High-pass cutoff
lp = 450;      % Low-pass cutoff
winSec = 0.25; % 250 ms window
stepSec = 0.125; % 50% overlap
Q = 1e-4;      % Kalman process noise
R = 1e-2;      % Kalman measurement noise

%% 2️⃣ GET ALL CSV FILES -------------------------------------------------
files = dir(fullfile(inputFolder, '*.csv'));
if isempty(files)
    error('No CSV files found in %s', inputFolder);
end

fprintf('Found %d EMG CSV files.\n', numel(files));

%% 3️⃣ PROCESS EACH FILE -------------------------------------------------
for k = 1:length(files)
    csvFile = fullfile(inputFolder, files(k).name);
    [~, name, ~] = fileparts(csvFile);
    
    % Extract disease name (remove unwanted parts if any)
    diseaseLabel = strrep(name, '_', '');
    
    fprintf('\n🔹 Processing file %d/%d: %s\n', k, length(files), name);
    
    % --- Load Data ---
    data = readmatrix(csvFile);
    if size(data,2) > 1 && mean(diff(data(:,1)))>0
        emg = data(:,2:end);
    else
        emg = data;
    end
    [nSamples, nCh] = size(emg);
    
    % --- Bandpass Filter ---
    [b,a] = butter(4, [hp lp]/(fs/2), 'bandpass');
    emg_filt = filtfilt(b,a,emg);
    
    % --- Kalman Filtering ---
    kalman_est = zeros(size(emg_filt));
    for ch = 1:nCh
        x_est = emg_filt(1,ch);
        P = 1;
        for i = 2:nSamples
            x_pred = x_est;
            P = P + Q;
            K = P / (P + R);
            x_est = x_pred + K * (emg_filt(i,ch) - x_pred);
            P = (1 - K) * P;
            kalman_est(i,ch) = x_est;
        end
    end
    
    % --- Feature Extraction ---
    win = round(winSec * fs);
    step = round(stepSec * fs);
    starts = 1:step:(nSamples - win);
    numWins = length(starts);
    
    feature_names = {'RMS','MeanAbs','ZC','MAV','SpectralMedian','KalmanMean'};
    X = zeros(numWins, nCh * numel(feature_names));
    
    for w = 1:numWins
        idx = starts(w):starts(w)+win-1;
        featRow = [];
        for ch = 1:nCh
            sig = emg_filt(idx,ch);
            rmsVal = rms(sig);
            meanAbs = mean(abs(sig));
            zc = sum(abs(diff(sign(sig)))) / 2;
            mav = mean(abs(sig));
            [pxx,f] = pwelch(sig,[],[],[],fs);
            specMed = f(find(cumsum(pxx)/sum(pxx) >= 0.5, 1));
            kalMean = mean(kalman_est(idx,ch));
            featRow = [featRow rmsVal meanAbs zc mav specMed kalMean];
        end
        X(w,:) = featRow;
    end
    
    % --- Labeling ---
    labels = repmat({diseaseLabel}, numWins, 1);
    
    % --- Save ---
    savePath = fullfile(outputFolder, ['features_' lower(diseaseLabel) '.mat']);
    save(savePath, 'X', 'labels', 'feature_names');
    fprintf('✅ Saved: %s\n', savePath);
end

fprintf('\n🎯 Feature extraction completed for all %d diseases!\n', length(files));
fprintf('Results saved in: %s\n', outputFolder);
