clc
clear 
%%
pre.x = rand(1,1,100,8);
now.dzdx = rand(1,1,5,8);
% now.dzdx = rand(100,8,5);
layer.weights = {randn(100,100,5)};

fwd = @(x) nnl2lp_forward(layer, x, now);
now = fwd(pre);
pre = nnl2lp_backward(layer, pre, now);

%% 
vl_testder_custom(fwd, pre, now, 1e-3);