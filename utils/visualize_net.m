function [ h ] = visualize_net( net, x, y, threshold )
%UNTITLED6 Summary of this function goes here
%   Detailed explanation goes here

if ~exist('threshold', 'var') || isempty(threshold)
    threshold = 0.5;
end

X = [x(:) y(:)]';
Tout = net(X);
Tout = reshape(Tout, size(x));
safe = Tout >= threshold;
unsafe = Tout < threshold;
Tout(safe) = 1;
Tout(unsafe) = 0;

%surf(x, y, Tout, 'edgecolor','none');

cm = [1 1 0;
      1 1 1];

h = figure;
pcolor(y, x, Tout); colormap(cm); hold on;
shading interp;
y1 = y(:);
x1 = x(:);
rectangle('Position', [min(y1) min(x1) max(y1) - min(y1) max(x1) - min(x1)], 'LineWidth', 1);

end

