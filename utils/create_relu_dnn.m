function [ net ] = create_relu_dnn( nLayers, nNeurons, nInputs, nOutputs )
%CREATE_RELU_DNN Create a multilayer feedforward neural network with ReLU
%as activation function in every layer.
%   nLayers    Number of layers including the output layer.
%   nNeurons   Number of neurons in each hidden layer.
%   nInputs    Number of input variables (size of an input vector).
%   nOutputs   Number of output variables (size of an output vector).
%   net        The ReLU multilayer neural network.
%
%   Example:   net = create_relu_dnn(4, 10, 2, 1);
%              Create a ReLU DNN with 3 hidden layers of 10 neurons each, 2
%              inputs, and 1 output.

net = create_dnn(nLayers + 1, nNeurons, nInputs, nOutputs);
for i = 1:nLayers
    net.layers{i}.transferFcn = 'poslin'; % 'tansig'; 'poslin';
end

%net.layers{net.numLayers}.transferFcn = 'softmax';

end