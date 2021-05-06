function [ net ] = create_dnn( nLayers, nNeurons, nInputs, nOutputs )
%CREATE_DNN Create a multilayer feedforward neural network.
%   nLayers    Number of layers including the output layer.
%   nNeurons   Number of neurons in each hidden layer.
%   nInputs    Number of input variables (size of an input vector).
%   nOutputs   Number of output variables (size of an output vector).
%   net        The multilayer neural network.
%
%   Example:   net = create_dnn(4, 10, 2, 1);
%              Create a DNN with 3 hidden layers of 10 neurons each, 2
%              inputs, and 1 output.

% Create a custom neuron network
nL = nLayers;
net = network;

net.numInputs = 1; % 1 input source
net.numLayers = nL;

net.inputs{1}.size = nInputs; % number of elements in an input vector.

% To create a bias connection to the ith layer you can set 
% net.biasConnect(i) to 1.
% Add a bias connection to every layer
net.biasConnect = ones(nL, 1);

% net.inputConnect(i,j) represents the presence of an input weight 
% connection going to the ith layer from the jth input.
% Specify that all inputs are connected to layer 1.
% No inputs are connected to the other layers.
net.inputConnect = [1; zeros(nL - 1, 1)];

% net.layerConnect(i,j) represents the presence of a layer-weight 
% connection going to the ith layer from the jth layer.
net.layerConnect = zeros(nL, nL);
for i = 2:nL
    net.layerConnect(i, i - 1) = 1;
end

% net.outputConnect(i) indicates whether to connect ith layer's output to
% the outside.
net.outputConnect = [zeros(1, nL - 1), 1];

for i = 1:nL - 1
    net.layers{i}.size = nNeurons;
    net.layers{i}.transferFcn = 'tansig'; % 'tansig'; 'poslin';
    net.layers{i}.initFcn = 'initnw';
end

net.inputs{1}.processFcns = {'mapminmax', 'removeconstantrows'};

net.layers{nL}.size = nOutputs;
net.layers{nL}.initFcn = 'initnw';
net.layers{nL}.transferFcn = 'logsig'; % 'logsig'; 'poslin';

% Set the divide function to dividerand (divide training data randomly).
net.divideFcn = 'dividerand';
net.adaptFcn = 'adaptwb';
net.performFcn = 'mse';
net.trainFcn = 'trainlm';
net.plotFcns = {'plotperform', 'plottrainstate', 'ploterrhist', ...
                'plotconfusion', 'plotroc'};

end

