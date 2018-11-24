function [pre] = nnrmmean_backward(layer, pre, now)

    xin = pre.x;
    [~,~,c,n] = size(xin);
    dzdy = squeeze(now.dzdx);
     % if GPU is used
    gpuMode = isa(xin, 'gpuArray');
    
    [~,~,c,n] = size(xin);
    if gpuMode
        pre.dzdx = gpuArray(zeros([1,1,c,n],'single'));
    else
        pre.dzdx = zeros([1,1,c,n],'single');
    end
    dmu = -1*sum(dzdy,2);
    pre.dzdx =  dzdy+1/n*repmat(dmu,1,n);