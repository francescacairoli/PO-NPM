function T_pred = predictDNNensbl(DNNs,X_test)

nsamples = size(X_test,2);
T_pred = zeros(1,nsamples);
for i=1:length(DNNs)
    net = DNNs{i}.net;
    T_pred = T_pred+round(net(X_test));
end
T_pred = round(T_pred./length(DNNs));