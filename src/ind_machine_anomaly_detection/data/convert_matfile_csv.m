clear all; clc;
load('MachineData.mat')

% Creating file for labels
labelTab = trainData(:, 'label');
writetable(labelTab, 'labels.csv')

% Creating .csv files for sensors readings
for i = 1:1:40
    rowMat = cell2mat(trainData{i, 1:3});
    writematrix(rowMat, string(i) + '.csv');
end