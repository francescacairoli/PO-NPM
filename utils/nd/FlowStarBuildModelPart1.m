function [ part1 ] = FlowStarBuildModelPart1( part1_template_file, T, max_jumps )
%UNTITLED3 Summary of this function goes here
%   Detailed explanation goes here

part1 = fileread(part1_template_file);
part1 = strrep(part1, '$TIME_BOUND$', num2str(T));
part1 = strrep(part1, '$MAX_JUMPS$', num2str(max_jumps));

end

