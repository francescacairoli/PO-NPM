function [ acc, fn, fp, n_acc, n_fn, n_fp ] = testNN_file(matFile, net, n, positive_label)
%TESTNN Compute and display overall accuracy, false positives, and false
%negatives performance for a neural network on a dataset.
%   matFile    .mat file containing test data set (X, T)
%   net        The trained neural network to test
%   n          Number of test points. For example, if matFile contains
%              10,000 points but you want to use only the first 5000 points
%              for testing, then specify n = 5000. n is optional.
%   positive_label Positive label value. Default is 0.
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

data = load(matFile);
if ~exist('n', 'var') || isempty(n)
    n = size(data.X, 2);
end
X = data.X(:, 1:n);
T = data.T(1:n);
[ acc, fn, fp, n_acc, n_fn, n_fp ] = testNN(net, X, T, positive_label);
end

