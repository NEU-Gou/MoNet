function [ now ] = veronese_forward(layer, pre, now)
%veronese_forward compute the collection of monomials
%   layer.order = order;
%   layer.powers = powers;

powers = layer.powers;
order = layer.order;
[h,w,c,n] = size(pre.x);
if order == 0
    now.x = pre.x./sqrt(w*h);
    return
end

xin=permute(pre.x, [1,2,4,3]); % size is h, w, n, c
xin=reshape(xin, h*w*n, c)';

[now.x,power] = veronese(cat(1,ones(1,h*w*n),xin),order,powers);
now.x = reshape(now.x,size(now.x,1),h,w,n);
now.x = permute(now.x,[2,3,1,4]);

now.x = now.x./sqrt(w*h);
% now.aux{1} = power;
end