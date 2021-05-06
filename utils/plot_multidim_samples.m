function plot_multidim_samples(samples_file, ranges,markerSize)
if nargin < 3
    markerSize = 36;
end
load(samples_file);
nDims = size(X,1);
figure;
for i=1:nDims-1
    for j=i+1:nDims
        subplot(nDims-1,nDims-1,(nDims-1)*(i-1)+j-1)
        pos_entries = T==1;
        neg_entries = T==0;
        % positives
        hold on
        scatter(X(i,pos_entries),X(j,pos_entries),markerSize,'red')
        xlim(ranges(i,:));ylim(ranges(j,:));
        scatter(X(i,neg_entries),X(j,neg_entries),markerSize,'blue')
        hold off
    end
end
end