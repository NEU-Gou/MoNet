function [U,S] = teig(X)
% teig is truncated eig 
  [U,S] = eig(X);
  diagS = diag(S);
  tol = eps(max(diagS));
  ind = diagS>tol+min(diagS);
  diagS(~ind)= min(diagS); S = diag(diagS);
%   ind = diagS>tol+min(diagS);
%   S = S(ind,ind);
%   U = U(:,ind);
end