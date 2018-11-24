clc
clear
%%
pre.x = rand(5,5,3,10);
now.dzdx = ones(1,1,9,10);
for i = 1:10
    tmp = rand(25,3);
    now.dzdx(:,:,:,i) = reshape(tmp'*tmp, [1,1,9]);
end
layer = [];
%%
fwd = @(x) yang_bilinear_forward(layer,x, now);
now = fwd(pre);
pre = yang_bilinear_backward(layer,pre,now);

%% 
vl_testder_custom(fwd,pre,now,1e-3)