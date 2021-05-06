function [ acc, fn, fp, n_acc, n_fn, n_fp ] = testNN(net, X, T, positive_label)
%TESTNN Compute and display overall accuracy, false positives, and false
%negatives performance for a neural network on a dataset.
%   net        The trained neural network to test
%   X          The test samples, matrix of size n-by-m where n is the
%              dimension, m is the number of samples.
%   T          The test labels, vector of length m.
%   positive_label  Positive label value. Default is 0.
%   acc        Overall accuracy in percent.
%   fn         Fraction of false negatives. Number of false negatives
%              divides by number of test samples.
%   fp         Fraction of false positives. Number of false positives
%              divides by number of test samples.
%   n_acc      Number of samples that the NN predicted correctly.
%   n_fn       Number of false negatives.
%   n_fp       Number of false positives.

if ~exist('positive_label', 'var') || isempty(positive_label)
    positive_label = 0;
end

n = length(T);
Tout = net(X);
[~,cm] = confusion(T,Tout);
n_acc = sum(diag(cm));
% cm(i,j) is the number of samples whose target is the ith class that was classified as j
if positive_label == 0
    n_fp = cm(2, 1); % predict unsafe but actually safe
    n_fn = cm(1, 2); % predict safe but actually unsafe
else
    n_fp = cm(1, 2);
    n_fn = cm(2, 1);
end

acc = n_acc / n * 100;
fp = n_fp / n;
fn = n_fn / n;

fprintf('Overall accuracy: %.3f%% (%i/%i), FP: %.5f (%i/%i), FN: %.5f (%i/%i)\n', acc, n_acc, n, fp, n_fp, n, fn, n_fn, n);
% fprintf('False positives: %.5f (%i/%i)\n', fp, n_fp, n);
% fprintf('False negatives: %.5f (%i/%i)\n', fn, n_fn, n);
end