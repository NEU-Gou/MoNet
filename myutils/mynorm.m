function [o,odev,odevmat] = mynorm(X,type)

switch type
  case 'fro2'
    o = norm(X,'fro').^2;
    odevmat = 2*X;
    odev = odevmat(:);
  otherwise
    error('Not supported!');
end