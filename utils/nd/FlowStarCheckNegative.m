function [ is_negative ] = FlowStarCheckNegative( model_part1, model_part2, init_mode, init_state, flowstar_path, model_tmp_file)
%UNTITLED Summary of this function goes here
%   Detailed explanation goes here

% is_negative = false;

if ~exist('model_tmp_file', 'var') || isempty(model_tmp_file)
    model_tmp_file = [pwd '/flow_star_tmp.model'];
end

init_block = FlowStarBuildInitBlock(init_mode, init_state);
model_file_contents = [model_part1 init_block model_part2];
fileID = fopen(model_tmp_file, 'w');
fprintf(fileID, '%s', model_file_contents);
fclose(fileID);

%fprintf('Temporary model file ''%s'' created.\n', model_tmp_file);

flowstar_cmd = sprintf('%s < %s', flowstar_path, model_tmp_file);
[status,cmd_out] = system(flowstar_cmd);

% disp(cmd_out);
k = strfind(cmd_out,'UNSAFE');

is_negative = (status == 0) && isempty(k);
end

