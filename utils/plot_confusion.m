function [ output_args ] = plot_confusion(net, tr, X, T)
%PLOT_CONFUSION Plot the confusion matrix for a trained neural network on
%the dataset used for training, validation, and testing.
%   net    Trained neural network.
%   tr     Training record.
%   X      Dataset, matrix of size m-by-n, where m is the space dimension,
%          n is the number of observations. 
%   T      Vector of size 1-by-n representing true classes of the
%          corresponding observations in X.

% Training Confusion Plot Variables

yTrn = net(X(:,tr.trainInd));

tTrn = T(:,tr.trainInd);

% Validation Confusion Plot Variables

yVal = net(X(:,tr.valInd));

tVal = T(:,tr.valInd);

% Test Confusion Plot Variables

yTst = net(X(:,tr.testInd));

tTst = T(:,tr.testInd);

% Overall Confusion Plot Variables

yAll = net(X);

tAll = T;

% Plot Confusion
figure;
plotconfusion(tTrn, yTrn, 'Training', tVal, yVal, 'Validation', tTst, yTst, 'Test', tAll, yAll, 'Overall')
end

