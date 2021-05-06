function [state,options,optchanged] = myGAoutfun(options,state,flag)

persistent bestPopulation bestScores popuSize
optchanged = false;

% popuSize is the largest population of false negatives to keep before
% exiting
if isempty(popuSize)
    popuSize = 10000;
end

currentPop = state.Population;
currentScores = state.Score;

% select all the false negatives
% currentPop = currentPop(currentScores<=-0.5,:);
% currentScores = currentScores(currentScores<=-0.5,:);

currentPop = currentPop(currentScores<0.5,:);
currentScores = currentScores(currentScores<0.5,:);

%size(bestPopulation)
%size(currentPop)

% merge them with the previous iterations
bestPopulation = [bestPopulation;currentPop];
bestScores = [bestScores;currentScores];

% keep only the unique points
[bestPopulation,ia] = unique(bestPopulation,'rows');
bestScores = bestScores(ia);

assignin('base','falsePredictions',bestPopulation);
assignin('base','falsePredictionScores',bestScores);

if length(bestScores) >= popuSize
	state.StopFlag = 'y';
	fprintf('Found already more than %d false predictions. GA stopping...\n',popuSize);
    bestPopulation = [];
    bestScores = [];
end

end
