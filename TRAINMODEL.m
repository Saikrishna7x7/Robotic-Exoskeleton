%% EMG_Disease_Model_Train_Improved.m
% Full-featured training script for multi-disease EMG classification
% Sapthami version - with normalization + ensemble + 5-fold validation

clc; clear; close all;

%% 1. PATH SETTINGS
dataFolder = 'C:\Users\Machine\OneDrive\Desktop\emg_csv_only1\EMG_Results';

matFiles = dir(fullfile(dataFolder, 'features_*.mat'));
if isempty(matFiles)
    error('No feature files found in %s', dataFolder);
end

%% 2. INITIALIZE COMBINED ARRAYS
allFeatures = [];
allLabels = [];

fprintf('Loading and normalizing features from %d diseases...\n', numel(matFiles));

for k = 1:numel(matFiles)
    file = fullfile(dataFolder, matFiles(k).name);
    load(file, 'X'); % load feature matrix
    
    % --- Normalize features for this disease (column-wise) ---
    X = normalize(X);
    
    % Extract disease name from filename
    [~, name, ~] = fileparts(matFiles(k).name);
    label = erase(name, 'features_');
    
    nSamples = size(X,1);
    fprintf('%2d) %-25s  -> %4d samples\n', k, label, nSamples);
    
    allFeatures = [allFeatures; X];
    allLabels = [allLabels; repmat({label}, nSamples, 1)];
end

fprintf('\n✅ Total samples: %d | Classes: %d\n', size(allFeatures,1), numel(unique(allLabels)));

%% 3. CREATE TABLE
featureNames = { ...
    'RMS','MeanAbs','ZC','MAV','SpectralMedian','KalmanMean', ...
    'WaveformLength','Variance','Skewness','Kurtosis'};

% Handle dimension mismatch (truncate or pad)
nFeat = min(numel(featureNames), size(allFeatures,2));
T = array2table(allFeatures(:,1:nFeat), 'VariableNames', featureNames(1:nFeat));
T.Disease = categorical(allLabels);

%% 4. SPLIT DATA (80/20)
cv = cvpartition(T.Disease, 'HoldOut', 0.2);
trainData = T(training(cv), :);
testData  = T(test(cv), :);

%% 5. TRAIN ENSEMBLE CLASSIFIER (RANDOM FOREST)
fprintf('\nTraining Random Forest Ensemble model...\n');
template = templateTree('MaxNumSplits', 50);
Mdl = fitcensemble(trainData, 'Disease', ...
    'Method', 'Bag', ...
    'NumLearningCycles', 150, ...
    'Learners', template);

%% 6. TEST PERFORMANCE
pred = predict(Mdl, testData);
acc = mean(pred == testData.Disease);
fprintf('\n🎯 Test Accuracy: %.2f%%\n', acc*100);

% Confusion Matrix
figure;
cm = confusionchart(testData.Disease, pred);
cm.Title = 'EMG Disease Classification Confusion Matrix';
cm.RowSummary = 'row-normalized';
cm.ColumnSummary = 'column-normalized';

%% 7. 5-FOLD CROSS-VALIDATION
fprintf('\nPerforming 5-fold cross-validation...\n');
cvMdl = crossval(Mdl, 'KFold', 5);
cvLoss = kfoldLoss(cvMdl);
fprintf('📊 Cross-validated Accuracy: %.2f%%\n', (1 - cvLoss)*100);

%% 8. SAVE MODEL
compactMdl = compact(Mdl);
save('EMG_Disease_CompactModel.mat', 'compactMdl');
fprintf('\n✅ Saved trained model as EMG_Disease_CompactModel.mat\n');
