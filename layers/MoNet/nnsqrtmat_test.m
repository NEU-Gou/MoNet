clc
clear 
%%
x = zeros(11,11,20);
for i = 1:20
    tmp = rand(1000,10);
    tmp = cat(2,ones(1000,1), tmp);
    x(:,:,i) = (tmp'*tmp)/1000;
end
layer.eps = 0;
dzdy = ones(1,1,11*11,20);
%%
fwd = @(x) nnsqrtmat_forward(layer, struct('x',x), struct('dzdx',dzdy));
% fwd = @(x) nnlogmat_forward(layer, struct('x',x), struct('dzdx',dzdy));

% backward
dzdx = getfield(nnsqrtmat_backward(layer, struct('x',x), fwd(x)),'dzdx');
% dzdx = getfield(nnlogmat_backward(layer, struct('x',x), fwd(x)),'dzdx');

%% numerical derivative
numerical_der_step = 1e-3;
[x,xp] = vl_testder(@(X) fwd(X),double(x),double(dzdy),double(dzdx),numerical_der_step);

% analyze absolute error
analyze_error(x-xp);
% analyze relative error
analyze_error((x-xp)./(x+1e-10));