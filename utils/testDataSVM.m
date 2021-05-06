function [X_test,T_test,T_pred] = testDataSVM(path,SVMModel, sampleSize,conf_level)

if nargin <4
   conf_level = 0.99; 
end

load(path)
X_test = X;
T_test = T;

if isa(SVMModel,'cell')
    T_pred = predictDNNensbl(SVMModel,X_test);
elseif isa(SVMModel,'network')
    T_pred = round(SVMModel(X_test));
else
    X_test = X';
    T_test = T';
    T_pred = predict(SVMModel,X_test);
end

DSsize = length(T_test);

if nargin >= 3
   DSsize = min(DSsize,sampleSize);
   T_pred = T_pred(1:DSsize);
   T_test = T_test(1:DSsize);
end

% fprintf('Accuracy with %s: %f\n', path, sum(T_pred==T_test)/DSsize);
% fprintf('False positive rate with %s: %f\n',path, sum(T_pred==0&T_test==1)/DSsize);
% fprintf('False negative rate with %s: %f\n',path, sum(T_pred==1&T_test==0)/DSsize);


% statistics for accuracy
% [mean_acc,se_acc,ci_acc,printString] = BernoulliStatistics(T_test==T_pred,conf_level);
[mean_acc,se_acc,ci_acc,printString] = PreciseStatistics(T_test==T_pred,conf_level);
fprintf('Accuracy. Mean: %f; STD: %f; CI: [%f,%f]\n',mean_acc,se_acc,ci_acc(1),ci_acc(2));
disp(printString)
% statistics for FNs (old labelling)
[mean_fn,se_fn,ci_fn,printString] = PreciseStatistics(T_test==0&T_pred==1,conf_level);
fprintf('False negatives. Mean: %f; STD: %f; CI: [%f,%f]\n',mean_fn,se_fn,ci_fn(1),ci_fn(2));
disp(printString)
% statistics for FPs (old labelling)
[mean_fp,se_fp,ci_fp,printString] = PreciseStatistics(T_test==1&T_pred==0,conf_level);
fprintf('False positives. Mean: %f; STD: %f; CI: [%f,%f]\n',mean_fp,se_fp,ci_fp(1),ci_fp(2));
disp(printString)

end