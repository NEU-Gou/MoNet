function upper = reshape_forward(opts, lower, upper, masks)
% reshape_forward 
 
  [M,N,D,L] = size(lower.x);
  eval(['X=' opts.data_type '(gather(reshape(lower.x,M*N,D,L)));']);
  if exist('masks','var')
    mask_counts = cellfun(@(x)size(x,3),masks); n_masks = sum(mask_counts); mask = zeros(M*N,1,n_masks,'like',masks{1});
    parts = [1 cumsum(mask_counts(1:end-1))+1;cumsum(mask_counts)];
    ind = zeros(n_masks,1); L = length(ind);
    for i = 1: size(masks,2), ind(parts(1,i):parts(2,i)) = i; end;
    for i = 1: size(masks,2), mask(:,1,parts(1,i):parts(2,i)) = reshape(masks{i},[M*N 1 mask_counts(i)]); end;
    X = bsxfun(@times,X(:,:,ind),mask);
  end

  switch opts.shape
    case 'oprod'
      upper.x = zeros(D,D,L,opts.data_type);
      
      for i = 1: L
        upper.x(:,:,i) = X(:,:,i)'*X(:,:,i);
      end
      
    case 'iprod'      
      upper.x = zeros(M*N,M*N,L,opts.data_type);
      
      for i = 1: L
        upper.x(:,:,i) = X(:,:,i)*X(:,:,i)';
      end
      
    case 'simple'
      upper.x = X;
      
    otherwise
      error('Unsupported!');
  end
end