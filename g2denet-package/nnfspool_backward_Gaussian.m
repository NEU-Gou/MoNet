function BP = nnfspool_backward_Gaussian(opts, lower, upper, masks)

% nnfspool_forward_Gaussian is the function
% performing the backward pass of Gaussian embedding 

% Qilong Wang, Peihua Li, Lei Zhang. G^2DeNet: Global Gaussian 
% Distribution Embeding Network and Its Application to Visual
% Recognition. In CVPR, 2017.

% Copyright (C) 2017 Qilong Wang, Peihua Li, Lei Zhang.
% All rights reserved.

  [M,N,D,L] = size(lower.x);
  
  if ~exist('masks','var')
    for i = 1: L
      masks{i} = ones(M,N,1,1,'single','gpuArray')/(M*N);
    end
  end
  
  n = M * N;

  gpuMode = isa(lower.x, 'gpuArray') ;  
  if gpuMode 
       upper_dzdx = double(gather(upper.dzdx));
       lower_x = double(gather(lower.x));
  else
       upper_dzdx = double(upper.dzdx);
       lower_x = double(lower.x);
  end
  
  X = reshape(lower_x, [n, D, L]); 
 
  BP = zeros(M,N,D, L,'single','gpuArray');
  
  lower_dzdx = zeros(M, N, D, L, 'double');
  dzdx = zeros(n, D, 'double');
      
  D = D + 1;
      
  S = zeros(D, D, 'double');
  V = zeros(D, D, 'double');
      
  dLdC = zeros(D, D, 'double');
  dLdV = zeros(size(V), 'double');
  dLdS = zeros(size(S), 'double');
  K = zeros(size(S), 'double');
  temp_K = zeros(size(S), 'double');
     
  diag_S = zeros(D, 1, 'double');
      
  dLdY = zeros(D, D, 'double');
  for i=1:L % iterate over images in batch
          
        V = upper.aux{1}(:, :, i);
        S = upper.aux{2}(:, :, i);

        dLdC(:) = double(upper_dzdx(1, 1,  :,  i)); 
        diag_S = diag(S);
             
        ind =diag_S  > 1e-10; 
        Dmin = min(find(ind, 1, 'last'), n+1);

        dLdV(:, 1:Dmin) = 2 *  symmetric(dLdC) * V(:, 1:Dmin) * diag(diag_fun(diag_S(1:Dmin), opts.epsilon, 'power', opts.alpha));  
        dLdS(1:Dmin, 1:Dmin) =   diag(diag_fun_deri(diag_S(1:Dmin), opts.epsilon, 'power', opts.alpha)) * ( V(:, 1:Dmin)' * dLdC * V(:, 1:Dmin));
         
        K(1:Dmin, 1:Dmin) = diag_S(1:Dmin)  * ones(1, Dmin);
        K(1:Dmin, 1:Dmin)  = 1 ./ (K(1:Dmin, 1:Dmin)  - K(1:Dmin, 1:Dmin)');
        K(isinf(K)) = 0;
            
        dLdY =  symmetric( V(:, 1:Dmin) * (diag(diag( dLdS(1:Dmin, 1:Dmin) ))  +  K(1:Dmin, 1:Dmin)' .* (V(:, 1:Dmin)' * dLdV(:, 1:Dmin)) ) * V(:, 1:Dmin)' );       
        dzdx = (2 / n) .* bsxfun(@plus, X(:, :, i) * dLdY(1:D-1, 1:D-1), dLdY(D, 1:D-1));
        assert(~any(isnan(dzdx(:))));
          
       
        lower_dzdx(:,:,:,i) = reshape(dzdx,  [size(lower_dzdx,1) size(lower_dzdx, 2) size(lower_dzdx, 3)]); %warning('no normalization');   
  end
      
  if gpuMode
        lower.dzdx = gpuArray(single(lower_dzdx));
  else
        lower.dzdx = single(lower_dzdx);
  end
  
  BP = lower.dzdx;
end
