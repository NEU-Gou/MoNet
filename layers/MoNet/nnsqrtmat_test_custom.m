clc
clear
%%
x = zeros(11,11,10);
for i = 1:10
    tmp = rand(1000,10);
    tmp = cat(2,ones(1000,1), tmp);
    x(:,:,i) = (tmp'*tmp)/1000;
end
pre.x = x;
layer.eps = 0;1e-3;
now.dzdx = rand(1,1,11*11,10);
% now.dzdx = rand(11,11,10);

fwd = @(x) nnsqrtmat_forward(layer, x, now);
now = fwd(pre);
pre = nnsqrtmat_backward(layer, pre, now);

% fwd = @(x) nnlogmat_forward(layer, x, now);
% pre = nnlogmat_backward(layer, pre, fwd(pre));

%%
vl_testder_custom(fwd, pre, now,1e-2)