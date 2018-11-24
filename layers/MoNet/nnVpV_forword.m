function [ now ] = nnVpV_forword( layer, pre, now )
%nnVpV_forword compute the moment matrix V'*V
%   pre.x   - [h,w,c,n]
%   now.x   - [c,c,n]
%   layer.eps - regular term for moment matrix

    xin = pre.x;
    % if GPU is used
    gpuMode = isa(xin, 'gpuArray');
    
    [h,w,c,n] = size(xin);
    if gpuMode
        now.x = gpuArray(zeros([c,c,n],'single'));
    else
        now.x = zeros([c,c,n],'single');
    end
    for i = 1:n
        a = reshape(xin(:,:,:,i),[h*w,c]);
        now.x(:,:,i) = a'*a./(h*w);
        now.x(2:end,2:end,i) = now.x(2:end,2:end,i) + layer.eps*eye(c-1);
    end
end

