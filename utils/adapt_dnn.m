function [ net, acc, fn, fp ] = adapt_dnn(net, test_ds, adapt_ds, lr)
%UNTITLED2 Summary of this function goes here
%   Detailed explanation goes here
% NOTE: please use adapt_net if you have false negatives already.
%       this script will first find false negatives before adapting the
%       neural network.

if ~exist('lr', 'var') || isempty(lr)
    lr = 0.0005;
end

n_adp = numel(adapt_ds);
acc = zeros(n_adp + 1, 1);
fp = zeros(n_adp + 1, 1);
fn = zeros(n_adp + 1, 1);

% Before adaptation
[acc(1), fn(1), fp(1)] = testNN(net, test_ds.X, test_ds.T);

%net.divideFcn = ''; % Equivalent to 'dividetrain' -- assign all targets to training set
net.trainParam.showWindow = 0;
net.adaptFcn = 'adaptwb';
%net.trainFcn = 'trainb';

% for i = 1:numel(net.biases)
%     if ~isempty(net.biases{i})
%         net.biases{i}.learnFcn = 'learngd';
%         net.biases{i}.learnParam.lr = 0.001;
%     end
% end

for i = 1:numel(net.layerWeights)
    if ~isempty(net.layerWeights{i})
        net.layerWeights{i}.learnFcn = 'learngd';
        net.layerWeights{i}.learnParam.lr = lr;
    end
end

for i = 1:numel(adapt_ds)
    X = adapt_ds{i}.X;
    T = adapt_ds{i}.T;
    % Note that the convention here is different from the paper's.
    % T == 0 (class index 1) means unsafe, i.e., positive sample.
    % T == 1 (class index 2) means safe, i.e., negative sample.
    % In contrast, in the paper, class 0 means unreachable, i.e., safe,
    % class 1 means unsafe-reachable, i.e., unsafe.
    % Please keep in mind!
    %nPos(i) = sum(T == 0); % Number of positive samples
    %nNeg(i) = sum(T == 1); % Number of negative samples
    Tout = net(X); % Simulate the network, another way is Tout = sim(net, X)
    [~,cm,ind,~] = confusion(T, Tout);
    % cm(i,j) is the number of samples whose target is the ith class that was classified as j
    %nFalsePos(i) = cm(2, 1); % predict unsafe (class index 1) but actually safe (class index i)
    %nFalseNeg(i) = cm(1, 2);
    %ind{i,j} contains the indices of samples with the ith target class, but jth output class
    fn_idx = ind{1, 2};  % false negative indices
    fp_idx = ind{2, 1};
    % sanity checks
%     if length(fn_idx) ~= nFalseNeg(i) || length(fp_idx) ~= nFalsePos(i)
%         error('Sanity check failed!\n');
%     end
    err_idx = fn_idx; % [fn_idx fp_idx];
    X_err = X(:, err_idx); % states where NN makes incorrect predictions
    T_err = T(err_idx);
    if ~isempty(X_err)
    % retrain NN using both false negatives and false positives
        [net, ~] = adapt(net, X_err, T_err);
    else
        fprintf('FNs not found in adaptation dataset #%d \n', i);
    end
    %[net, tr] = adapt(net, X, T);
    [acc(i+1), fn(i+1), fp(i+1)] = testNN(net, test_ds.X, test_ds.T);

%     acc(i+1) = acc1;
%     fn(i+1) = fn1;
%     fp(i+1) = fp1;
end

end

