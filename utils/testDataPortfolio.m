function [X_test,T_test,T_pred] = testDataPortfolio(path,NNmodel,SVMModel,BTDmodel,sampleSize)


load(path)
X_test = X';
T_test = T';

T_pred = round(NNmodel(X_test'))';
T_pred = [T_pred predict(SVMModel,X_test)];
T_pred = [T_pred predict(BTDmodel,X_test)];
DSsize = length(T_test);

if nargin >= 3
   DSsize = min(DSsize,sampleSize);
   T_pred = T_pred(1:DSsize,:);
   T_test = T_test(1:DSsize);
end

T_pred_final = sum(T_pred,2)>1;

disp('Portfolio performance:');
fprintf('Accuracy with %s: %f\n', path, sum(T_pred_final==T_test)/DSsize);
fprintf('False positive rate with %s: %f\n',path, sum(T_pred_final==0&T_test==1)/DSsize);
fprintf('False negative rate with %s: %f\n',path, sum(T_pred_final==1&T_test==0)/DSsize);

end