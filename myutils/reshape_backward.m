function lower = reshape_backward(opts, lower, upper, masks)
% reshape_backward

  [M,N,D,L] = size(lower.x);
  upper_dzdx = upper.dzdx;
  X = reshape(lower.x,[M*N,D,L]);
  if exist('masks','var')
    mask_counts = cellfun(@(x)size(x,3),masks); n_masks = sum(mask_counts); mask = zeros(M*N,1,n_masks,'like',masks{1});
    parts = [1 cumsum(mask_counts(1:end-1))+1;cumsum(mask_counts)];
    ind = zeros(n_masks,1);
    for i = 1: size(masks,2), ind(parts(1,i):parts(2,i)) = i; end;
    for i = 1: size(masks,2), mask(:,1,parts(1,i):parts(2,i)) = reshape(masks{i},[M*N 1 mask_counts(i)]); end;
  end
  
  dzdx = zeros(M*N,D,L,'like',lower.x);
  
  switch opts.shape
    case 'oprod'  
      if exist('masks','var')
        for a=1:L % iterate over images in batch
          for b=1:size(masks{a},3) % iterate over masks in image
            dzdx(:,:,a) = dzdx(:,:,a) + 2*bsxfun(@times,X(:,:,a),mask(:,1,parts(1,a)+b-1).^2)*symmetric(upper_dzdx(:,:,parts(1,a)+b-1));
          end
        end
      else
        for b=1:L % iterate over masks in image
          dzdx(:,:,b) = dzdx(:,:,b) + 2*X(:,:,b)*symmetric(upper_dzdx(:,:,b));
        end
      end
      
    case 'iprod'
      if exist('masks','var')
        for a=1:L % iterate over images in batch
          for b=1:size(masks{a},3) % iterate over masks in image
            dzdx(:,:,a) = dzdx(:,:,a) + 2*symmetric(upper_dzdx(:,:,parts(1,a)+b-1))*bsxfun(@times,X(:,:,a),mask(:,1,parts(1,a)+b-1).^2);
          end
        end
      else
        for b=1:L % iterate over masks in image
          dzdx(:,:,b) = dzdx(:,:,b) + 2*symmetric(upper_dzdx(:,:,b))*X(:,:,b);
        end
      end
      
    case 'simple'
      if exist('masks','var')
        for a=1:L % iterate over images in batch
          for b=1:size(masks{a},3) % iterate over masks in image
            dzdx(:,:,a) = dzdx(:,:,a) + bsxfun(@times,upper_dzdx(:,:,parts(1,a)+b-1),mask(:,1,b));
          end
        end
      else
        dzdx = gpuArray(single(upper_dzdx));
      end
    otherwise
      error('Unsupported!');
  end
  
  lower.dzdx = reshape(dzdx,[M,N,D,L]);
end