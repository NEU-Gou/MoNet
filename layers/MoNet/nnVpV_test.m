clc
clear 
%%
pre.x = rand(5,5,3,10);
now.dzdx = rand(1,1,9,10)./10;
layer.eps = 0;
%%
% fwd = @(x) nnVpV_forword(layer, x, now);
fwd = @(x) yang_bilinear_forward(layer, x, now);
now = fwd(pre);
pre = yang_bilinear_backward(layer, pre, now);
% pre = nnVpV_backward(layer, pre, now);

%% numerical derivative
vl_testder_custom(fwd, pre,now,1e-5)