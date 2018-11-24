function [L,Ldev] = diagFun(D,fun,c)
% compute function of a diagonal matrix add constant displacement c if necessary
% also compute its derivative if you're at it
  if ~exist('c','var'), c = 0; end
  [M,N] = size(D);
  if isa(D,'gpuArray')
    L = zeros(size(D),'single','gpuArray');
  else
    L = zeros(size(D));
  end
  
  m = min(M,N);
  if nargout == 2
    [f,fdev] = fun(tdiag(D,0)+c);
    Ldev(1:m,1:m,:) = tdiag(fdev);
  else
    f = fun(tdiag(D,0)+c);
  end
  L(1:m,1:m,:) = tdiag(f,1);
%   if nargin == 2
%     Ldev(1:m,1:m,:) = tdiag(fdev);
%   end
end