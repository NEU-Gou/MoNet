clc
clear 
%%
pre.x = rand(5,5,10,8);
dzdy = rand(1,1,11*11,8);
opts.epsilon = 1e-3;
opts.alpha = 0.5;
fwd = @(x) nnfspool_forward_Gaussian(opts, x, struct('dzdx',dzdy));

now = fwd(pre);
pre.dzdx = nnfspool_backward_Gaussian(opts, pre, now);
%%
vl_testder_custom(fwd, pre, now);