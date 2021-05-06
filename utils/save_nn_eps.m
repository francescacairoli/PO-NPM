function [ output_args ] = save_nn_eps(net, filename)

jframe = view(net);
w = jframe.size.width - 3;
h = jframe.size.height - 32;
%# create it in a MATLAB figure
hFig = figure('Position',[100 100 w h]);
jpanel = get(jframe,'ContentPane');
[~,h] = javacomponent(jpanel);
set(h, 'units','normalized', 'position',[0 0 1 1])

%# close java window
jframe.setVisible(false);
jframe.dispose();

%# print to file
set(hFig, 'PaperPositionMode', 'auto')
set(hFig, 'Color', 'w')
saveas(hFig, filename, 'epsc')

%# close figure
close(hFig)


end

