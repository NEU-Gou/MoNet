function [U,S,V] = tsvd(X)
% teig is truncated svd 
  [U,S,V] = svd(X);
  diagS = diag(S);
  tol = max(size(X))*eps(max(diagS));
  ind = diagS>tol;
  S = S(ind,ind);
  U = U(:,ind);
  V = V(:,ind);
end