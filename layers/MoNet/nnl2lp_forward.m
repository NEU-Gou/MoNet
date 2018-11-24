function [ now ] = nnl2lp_forward( layer, pre, now )
%[ pre ] = nnl2lp_forward( layer, pre, now ) compute a linear projection of
%input - (||w * x||^2)^-1
%   pre.x   - [1,1,c,n]
%   now.x   - [1,1,NClass,n]
%   layer.w - [L,c,NClass]

    xin = squeeze(pre.x);
    [~,n] = size(xin);
    w = layer.weights{1};
    [L,~,Nclass] = size(w);
    
    gpuMode = isa(xin, 'gpuArray');
    if gpuMode
        now.x = gpuArray(zeros(1,1,Nclass,n,'single'));
    else
        now.x = zeros(1,1,Nclass,n,'single');
    end
    
    for cl = 1:Nclass
        now.x(1,1,cl,:) = -log(sum((w(:,:,cl)*xin).^2,1));
%         now.x(1,1,cl,:) = -sum((w(:,:,cl)*xin).^2,1);
    end
end

