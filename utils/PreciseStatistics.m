function [mean_x,se_x,ci_x,printString] = PreciseStatistics(X,conf_level)


if conf_level >= 1
    error('You must specify a confidence level < 1');
end

if conf_level > 0.5
    conf_level = 1 - conf_level;
end

trials = length(X);
successes = sum(X);

mean_x = mean(X);
se_x = sqrt(mean_x*(1-mean_x)/trials);

% build precise intervals (Clopper?Pearson interval)
if successes==trials
    ci_x = [(conf_level/2)^(1/trials),1];
    
elseif successes==0
    ci_x = [0,1-(conf_level/2)^(1/trials)];
    
else
    pd_lb = makedist('Beta',successes,trials-successes+1);
    pd_ub = makedist('Beta',successes+1,trials-successes);
    
    ci_x = [icdf(pd_lb,conf_level/2),icdf(pd_ub,1-conf_level/2)];
    
    % other bounds from Langford "Tutorial on Practical Prediction Theory for
    % Classification"
    ci_x = [max(0,ci_x(1)), min(1,ci_x(2))];
    
end
% round to the 3rd digit
% printString = sprintf('%.2f \\ci{%.3f,%.3f}',mean_x*100,floor(ci_x(1)*100*1000)/1000,ceil(ci_x(2)*100*1000)/1000);
% round to the 2nd digit
printString = sprintf('%.2f \\ci{%.2f,%.2f}',mean_x*100,floor(ci_x(1)*100*100)/100,ceil(ci_x(2)*100*100)/100);
end
