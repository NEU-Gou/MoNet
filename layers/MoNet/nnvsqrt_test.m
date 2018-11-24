clc
clear 
%%
pre.x = rand(5,5,10,8);
now.dzdx = rand(5,5,10,8);
layer.eps = 1e-3;

fwd = @(x) nnvsqrt_forward(layer, x, now);
now = fwd(pre);
pre = nnvsqrt_backward(layer, pre, now);

%% 
vl_testder_custom(fwd, pre, now, 1e-3);