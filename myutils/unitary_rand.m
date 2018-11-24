function Q = unitary_rand(N,fun)
  if nargin == 1
    A = randn(N);
  else
    A = fun(N);
  end
    
  [Q,~] = qr(A);
end