function upper = nnfspool_forward_Gaussian(opts, lower, upper, masks)

% nnfspool_forward_Gaussian is the function
% performing the forward pass of Gaussian embedding 

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
  
  n_masks = sum(cellfun(@(a) size(a,3), masks));
 
  gpuMode = isa(lower.x, 'gpuArray') ;  
  if gpuMode
      lower_x = double(gather(lower.x));
  else
      lower_x = double(lower.x);
  end
        
  n = M * N;
        
  X = reshape(lower_x, [n, D, L]);
  
  D = D + 1;
        
  upper_x = zeros(1, 1, D*D, L, 'double');      
    
  % alocate matrices
  S = zeros(D, D, L, 'double'); % eigenvalue matrix
  V = zeros(D, D, L, 'double'); 
  S_ = zeros(D, D, 'double');
  V_ = zeros(D, D, 'double');

  % compute the eigenvectors
  diag_S = zeros(D, 1, 'double');
  Y = ones(D, D, 'double');
  u = zeros(1, D-1, 'double');
  for i = 1: L
     % build Gaussian embedding
     Y(1:D-1, 1:D-1) = X(:, :, i)' * X(:, :, i) ./ n;
     u = mean(X(:, :, i));
     Y(D,1:D-1) = u;
     Y(1:D-1,D) = u;
     Y(D, D) = 1;
    
     [V_, S_] = eig(Y);
     
     % matrix sqre/log
     diag_S = diag(S_);
     [diag_S, idx] = sort(diag_S, 'descend');
     V_ = V_(:, idx);         
     ind =diag_S  > 1e-10; 
     Dmin = min(find(ind, 1, 'last'), n+1);
           
     V(:, 1:Dmin, i) = V_(:, 1:Dmin);
     S(1:Dmin, 1:Dmin, i) = diag(diag_S(1:Dmin));
      
     Y = V(:, 1:Dmin, i) * diag(diag_fun(diag_S(1:Dmin), opts.epsilon, 'power', opts.alpha)) * V(: , 1:Dmin, i)';
        
     upper_x(1,1, :, i) = Y(:);
  end

  if gpuMode
      upper.x = gpuArray(single(upper_x));
  else
      upper.x = single(upper_x);
  end
  
  upper.aux{1} = V;
  upper.aux{2} = S;
    
end

