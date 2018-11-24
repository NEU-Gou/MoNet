% test forward/backward 
clc
clear
%%
num_c = 5;
X = rand(10,10,8,10, 'single')+1;
label = randi(num_c, size(X,4),1);
for c = 1:num_c
    M = rand(size(X,3),size(X,3));
    M = M*M';
    [u,s,~] = svd(M);
    tmp = rand(size(X,3),size(X,3));
    layer.U{c} = tmp*tmp';
    tmp = rand(size(X,3),size(X,3));
    layer.US{c} = u*s^(-0.5);
    layer.S{c} = tmp*tmp';
    layer.M{c} = M;
    layer.Minv{c} = inv(M);
    layer.num_sample{c} = 100;
end
layer.class = label;

% forward
fwd = @(X) getfield(sos_loss_forward(layer, struct('x',X)),'x');

% backward
dzdy = ones(10,10,num_c,size(X,4),'single');
dzdx = getfield(sos_loss_backward(layer,struct('x',X),struct('dzdx',dzdy)),'dzdx');

% numeriacal derivative
numerical_der_step=1e-3;
[x,xp] = vl_testder(@(X) double(fwd(X)),double(X),double(dzdy),double(dzdx),numerical_der_step);

% analyze absolute error
analyze_error(x-xp);
% analyze relative error
analyze_error((x-xp)./(x+1e-10));