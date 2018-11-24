function SR = sym_rand(dims,fun)
% compute a symmetric random matrix

if nargin == 1, fun = @(x)randn(x); end
  
SR = fun(dims);
SR = triu(SR)+triu(SR,1)';