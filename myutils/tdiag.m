function Y = tdiag(X,isdiag)
% tensor diag is an equivalent to diag performed for a tensor
% FIXME both could be sped up i think but this is not a priority for now.

if isdiag
  [M,D] = size(X); 
  Y = zeros(M,M,D,'like',X);
  for i = 1: D
    Y(:,:,i) = diag(X(:,i));
  end

else
  [M,N,D] = size(X);
  Y = zeros(min(M,N),D,'like',X);
  for i = 1: D
    Y(:,i) = diag(X(:,:,i));
  end
end