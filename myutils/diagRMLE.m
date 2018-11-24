function L = diagRMLE(D,c,alpha)
% compute log of a diagonal matrix add constant displacement c if necessary
  if ~exist('c','var'), c = 0; end
  [M,N] = size(D);
  if isa(D,'gpuArray')
    L = zeros(size(D),'single','gpuArray');
  else
    L = zeros(size(D));
  end
  
  m = min(M,N);
  L(1:m,1:m) = diag((diag(D)+c).^(alpha));
end