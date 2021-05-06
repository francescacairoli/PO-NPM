function [ net ] = adapt_net(net, fn, lr, pos_label)
% ADAPT_NET: Adapt a neural network with false negative samples.

if ~exist('lr', 'var') || isempty(lr)
    lr = 0.0005;
end

if ~exist('pos_label', 'var') || isempty(pos_label)
    pos_label = 0; % Backward compatible
end

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

[net, ~] = adapt(net, fn, zeros(1, size(fn,2)) + pos_label);

end

