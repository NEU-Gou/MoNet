function [now] = onevsall_sum_loss(layer, pre, now)
% transfer labels to +-1 and sum to get the loss
xin = pre.x;
[h,w,c,n] = size(xin);
xin = squeeze(xin); % xin should be [1,1,c,n]
C = layer.class;
Y = ones(c,n)*(-1)./(c-1); % balance the weights of pos/neg
Y(sub2ind([c,n],C,[1:n])) = 1;
now.x = sum(sum(Y.*xin));