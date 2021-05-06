function [outcome,samplesNeeded] = SPRT(sampleFun,predFun,performanceFun,p_thresh,delta_indf,alpha,beta,maxSamples,label,verbose)

A = (1-beta)/alpha;
B = beta/(1-alpha);
p0 = p_thresh + delta_indf;
p1 = p_thresh - delta_indf;


n=0;
succ_samples = 0;
outcome = 0;

while n<maxSamples
    % one sample at a time
    if nargin(sampleFun)==1
        res = sampleFun(n+1);
        T = res(1,:);
        X = res(2:end,:);
    else
        [X, T] = sampleFun();
    end
    
    nsamples = length(T);
    Tpred = predFun(X);
    bernoulliSample = performanceFun(T,Tpred);
    succ_samples = succ_samples + bernoulliSample;
    p1m_p0m = ((p1.^succ_samples)*(1-p1).^(n+nsamples-succ_samples))/((p0.^succ_samples)*(1-p0).^(n+nsamples-succ_samples));
    
    if(p1m_p0m>=A)
        outcome = -1;%p<=p0
        n = n+nsamples;
        break;
    elseif (p1m_p0m<=B)
        outcome = 1;%p>=p1
        n = n+nsamples;
        break;
    end    
    n = n+nsamples;
end

samplesNeeded = n;

if nargin>=10 && verbose
    fprintf('%s: %d (%d); ',label,outcome,samplesNeeded);
end


end