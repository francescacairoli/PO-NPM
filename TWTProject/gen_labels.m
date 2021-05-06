function T = gen_labels(X, dreal_path, paramNames, max_jumps, dreal_template)

% X has size n x m where n is the number of input dimensions, m
% is the number of points
%
% only X and dreal_path are mandatory, the others are optional

addpath('../utils');
addpath('../utils/nd');

if nargin < 5
    dreal_template_path = 'water-triple_template.drh';
    if nargin < 4
        max_jumps = 1;
        if nargin < 3
            paramNames = {'x1';'x2';'x3'};
        end
    end
end
dreal_template = fileread(dreal_template_path);

nPoints = size(X,2);
T = zeros(1,nPoints);
spurious = zeros(1,nPoints);
for i =1:nPoints
	i
    param_values = X(:, i);
    init_mode_num = 0;% for this model, the initial mode is fixed (in the template)
    time_bound = 0; % same thing for the time bound
    
    % get the task ID to use in tmp file naming, so that parallel processes
    % don't overwrite them
    t = getCurrentTask();
    taskID = 0;
    if ~isempty(t)
        taskID = t.ID;
    end
    unrollDepth = [1,max_jumps+1];
    dRealRes = dRealCheck(param_values, paramNames, init_mode_num, dreal_template,...
        dreal_path, time_bound, unrollDepth, taskID);
    
    if dRealRes ~= 0
        T(i) =  dRealRes~=1;
    else
        disp('spurious sample found or time-out, discarding...')
        spurious(i)=1;
    end
    
end


end
