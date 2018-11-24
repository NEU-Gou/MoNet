function [ pre ] = nnVpV_backward( layer, pre, now )
%nnVpV_backward compute the moment matrix V'*V
%   pre.x   - [h,w,c,n]
%   now.x   - [c,c,n]
    dzdy = now.dzdx;
    xin = pre.x;
    [h,w,c,n] = size(xin);
    
    gpuMode = isa(xin, 'gpuArray');
    if gpuMode
        pre.dzdx = gpuArray(zeros(h,w,c,n,'single'));
    else
        pre.dzdx = zeros(h,w,c,n,'single');
    end
    for i = 1:n
        dzdy_ = dzdy(:,:,i);
        a = reshape(xin(:,:,:,i), h*w, c);
        pre.dzdx(:,:,:,i) = 2*reshape(a*dzdy_', h,w,c);
    end
    pre.dzdx = pre.dzdx./(h*w);
end

