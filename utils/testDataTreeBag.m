function [X_test,T_test,T_pred] = testDataTreeBag(path,TBModel,sampleSize)
load(path)
X_test = X';
T_test = T';
T_pred = predict(TBModel,X_test);
T_pred = str2double(cell2mat(T_pred));
DSsize = length(T_test);

if nargin >= 3
   DSsize = min(DSsize,sampleSize);
   T_pred = T_pred(1:DSsize);
   T_test = T_test(1:DSsize);
end

fprintf('Accuracy with %s: %f\n', path, sum(T_pred==T_test)/DSsize);
fprintf('False positive rate with %s: %f\n',path, sum(T_pred==0&T_test==1)/DSsize);
fprintf('False negative rate with %s: %f\n',path, sum(T_pred==1&T_test==0)/DSsize);

end