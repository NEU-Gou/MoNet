% [y,powers] = veronese(x,n,scale)
%     Computes the Veronese map of degree n, that is all
%     the monomials of a certain degree.
%     x is a K by N matrix, where K is dimension and N number of points
%     y is a K by Mn matrix, where Mn = nchoosek(n+K-1,n)
%     powes is a K by Mn matrix with the exponent of each monomial
%
%     Example veronese([x1;x2],2) gives
%     y = [x1^2;x1*x2;x2^2]
%     powers = [2 0; 1 1; 0 2]
%
% Copyright @ Rene Vidal, 2003

function [y,powers] = veronese(x,n,powers,scale)

[K,N] = size(x);
% powers = exponent(n,K);

if n==0
  y=1;
elseif n==1
  y=x;
else
  index = (abs(x) < 1e-39);
  x(index) = 1e-39;
  Mn = nchoosek(n+K-1,n);
  y = real(exp(powers*log(complex(x))));
end

if isreal(x)
    y = real(y);
end

if nargin==4
    if scale==1
        y = diag(sqrt(factorial(n)./prod(factorial(powers),2)))*y;
    end
end