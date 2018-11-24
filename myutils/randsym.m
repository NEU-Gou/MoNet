function SR = randsym(dims,fun)
% compute a symmetric random matrix

if nargin == 1, fun = @(x)randn(x); end
  
SR = fun(dims);
for i = 1:size(SR,3)
  SR(:,:,i) = triu(SR(:,:,i))+triu(SR(:,:,i),1)';
end