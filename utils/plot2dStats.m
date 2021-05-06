function plot2dStats(gridx,gridy,means,cis,xlab,ylab,xtickloc,xticklab,ytickloc,yticklab)
xlims = [gridx(1),gridx(end-1)];
ylims = [gridy(1),gridy(end-1)];
%plot accuracy
minVal = min(min(min(cis)));
maxVal = max(max(max(cis)));
figure;
subplot(1,3,1);
surf(gridx(1:end-1),gridy(1:end-1),means,'LineStyle','none')
colorbar;caxis([minVal maxVal]);
xlim(xlims);ylim(ylims);view(2);title('Mean')
if nargin >=10
    xlabel(xlab);ylabel(ylab);xticks(xtickloc);xticklabels(xticklab);yticks(ytickloc);yticklabels(yticklab);
end
subplot(1,3,2);
surf(gridx(1:end-1),gridy(1:end-1),squeeze(cis(:,:,1)),'LineStyle','none')
colorbar;caxis([minVal maxVal]);
xlim(xlims);ylim(ylims);view(2);title('C.I. LB')
if nargin >=10
    xlabel(xlab);ylabel(ylab);xticks(xtickloc);xticklabels(xticklab);yticks(ytickloc);yticklabels(yticklab);
end
subplot(1,3,3);
surf(gridx(1:end-1),gridy(1:end-1),squeeze(cis(:,:,2)),'LineStyle','none')
colorbar;caxis([minVal maxVal]);
xlim(xlims);ylim(ylims);view(2);title('C.I. UB')
if nargin >=10
    xlabel(xlab);ylabel(ylab);xticks(xtickloc);xticklabels(xticklab);yticks(ytickloc);yticklabels(yticklab);
end