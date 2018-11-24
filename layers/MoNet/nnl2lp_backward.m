function [ pre ] = nnl2lp_backward( layer, pre, now )
%   pre.x   - [1,1,c,n]
%   now.x   - [1,1,NClass,n]
%   layer.weights{1} - [L,c,NClass]

    xin = squeeze(pre.x);
    [c,n] = size(xin);
    w = layer.weights{1};
    dzdy = squeeze(now.dzdx);
    [L,c,NClass] = size(w);
    gpuMode = isa(dzdy, 'gpuArray');
    if gpuMode
        pre.dzdx = gpuArray(zeros(1,1,c,n,'single'));
        pre.dzdw = {gpuArray(zeros(L,c,NClass,'single'))};
    else
        pre.dzdx = zeros(1,1,c,n,'single');
        per.dzdw = {zeros(L,c,NClass,'single')};        
    end

    for cl = 1:NClass
        xp = w(:,:,cl)*xin;
        ds = -(sum(xp.^2,1)+1e-10).^(-1);
        
        pre.dzdx(1,1,:,:) = pre.dzdx(1,1,:,:) + ...
            reshape(2*w(:,:,cl)'*(xp.*(ones(L,1)*(ds.*dzdy(cl,:)))),...
            [1,1,c,n]);
        pre.dzdw{1}(:,:,cl) = 2*(xp.*(ones(L,1)*(ds.*dzdy(cl,:))))*xin';
    end
end

