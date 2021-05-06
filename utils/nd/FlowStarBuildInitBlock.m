function [ init_block ] = FlowStarBuildInitBlock( init_mode, init_state )
%UNTITLED2 Summary of this function goes here
%   Detailed explanation goes here
nf = char(10);
init_block = ['init' nf '{' nf init_mode nf '{' nf];
names = fieldnames(init_state);
for i = 1:numel(names)
    var_name = names{i};
    var_value = num2str(getfield(init_state, var_name));
    init_block = [init_block, var_name, ' in [', var_value, ',', var_value, ']', nf];
end
init_block = [init_block '}' nf '}' nf];
end

