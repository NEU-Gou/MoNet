% clc
% clear 
%%
layer.order = 1;
layer.powers = exponent(layer.order,11);
pre.x = gpuArray(rand(5,5,8,4,'single'));
now.dzdx = gpuArray(ones(5,5,nchoosek(size(pre.x,3)+layer.order,layer.order),4,'single'));

fwd = @(x) veronese_forward(layer, x, now);
now = fwd(pre);
pre = veronese_backward(layer,pre,now);

%%
vl_testder_custom(fwd, pre, now)