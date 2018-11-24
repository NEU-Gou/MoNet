function [pre] = onevsall_sum_loss_backward(layer, pre, now)

[h,w,c,n] = size(pre.x);
C = layer.class;
Y = ones(c,n)*(-1)./(c-1);
Y(sub2ind([c,n],C,[1:n])) = 1;
Y = reshape(Y,1,1,c,n);
Y = gpuArray(Y);
pre.dzdx = Y;