clc
clear 
%%
X = rand(1,1,100,10);
layer.mean = 0;
layer.momentum = 0.1;
dzdy = rand(1,1,100,10);
%%
fwd = @(X) nnrmmean_forward(layer, struct('x',X), struct('dzdx',dzdy));
dzdx = getfield(nnrmmean_backward(layer, struct('x',X), fwd(X)), 'dzdx');
%% numerical derivative
numerical_der_step = 1e-3;
[x,xp] = vl_testder(fwd,double(X),dzdy,dzdx,numerical_der_step);

% analyze absolute error
analyze_error(x-xp);
% analyze relative error
analyze_error((x-xp)./(x+1e-10));