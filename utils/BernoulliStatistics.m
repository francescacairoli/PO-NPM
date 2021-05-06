function [mean_x,se_x,ci_x,printString] = BernoulliStatistics(X,conf_level,wald)

if nargin < 3
    wald = false;
end

if conf_level >= 1
    error('You must specify a confidence level < 1');
end

if conf_level > 0.5
    conf_level = 1 - conf_level;
end

mean_x = mean(X);
pd = makedist('Normal',0,1);
z = icdf(pd,1-conf_level/2);
se_x = sqrt(mean_x*(1-mean_x)/length(X));

if wald
    ci_x = [max(0,mean_x - se_x*z), min(1,mean_x + se_x*z)];
    printString = sprintf('%f $\\pm$ %f',mean_x*100,ceil(se_x*z*100*1000)/1000);
else
    ns = sum(X==1); nf = sum(X==0); n = length(X);
    delta = sqrt(ns*nf/n + z^2/4);
    ci_x = (ns + z^2/2 + z*[-delta, delta])./(n + z^2);
    printString = sprintf('%f $\\pm$ %f - ',mean_x*100,ceil((diff(ci_x)/2)*100*1000)/1000);
    ci_x = [max(0,ci_x(1)), min(1,ci_x(2))];
end
%     printString = [printString, sprintf('[%.3f,%.3f]',floor(ci_x(1)*100*1000)/1000,ceil(ci_x(2)*100*1000)/1000)];

    printString = sprintf('%.2f \\ci{%.3f,%.3f}',mean_x*100,floor(ci_x(1)*100*1000)/1000,ceil(ci_x(2)*100*1000)/1000); 
end