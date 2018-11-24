function [now, layer] = nnrmmean_forward(layer, pre, now, train)
% nnrmmean_forward remove mean for each feature, use mini-batch mean during
% training and running mean for testing
% layer.mean - running mean
% layer.momentum - weights for running mean
% pre.x     - [1,1,c,n]
% now.x     - [1,1,c,n]
    if nargin < 4
        train = 1;
    end
    xin = pre.x;
    % if GPU is used
    gpuMode = isa(xin, 'gpuArray');
    
    [~,~,c,n] = size(xin);
    if gpuMode
        now.x = gpuArray(zeros([1,1,c,n],'single'));
    else
        now.x = zeros([1,1,c,n],'single');
    end
    
    xin = squeeze(xin);
    if train || layer.mean==0 % training phase, using mini-batch mean and accumulating running mean
        layer.mean = (1-layer.momentum) * layer.mean + ...
                        layer.momentum * mean(xin,2);
        xin = bsxfun(@minus, xin, mean(xin,2));
    else % testing phase, using running mean
        xin = bsxfun(@minus, xin, layer.mean);
    end
        
    now.x = reshape(xin, 1,1,c,n);