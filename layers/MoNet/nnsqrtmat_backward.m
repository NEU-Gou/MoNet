function [ pre ] = nnsqrtmat_backward( layer, pre, now )
%nnsqrtmat_backward matrix wise square root
%   now.x       - [1,1,c*c,n]
%   now.aux{1}  - eigen vectors
%   now.aux{2}  - eigen values
%   pre.x       - [c,c,n]
    
    precision = class(now.aux{1});
    [c,~,n] = size(pre.x);
    xin = pre.x;
    dzdy = now.dzdx;
    
    gpuMode = isa(xin, 'gpuArray');
    if gpuMode
        dzdy = gather(dzdy);
        xin = gather(xin);
    end
    if strcmp(precision,'double')
        dzdy = double(dzdy);
        xin = double(xin);
    end
    if gpuMode
        pre.dzdx = gpuArray(zeros(c,c,n,'single'));
    else
        pre.dzdx = zeros(c,c,n,'single');
    end
    dLdC = zeros(c,c,precision);
    dLdV = zeros(c,c,precision);
    dLdS = zeros(c,c,precision);
    K = zeros(c,c,precision);
%     dLdY = zeros(c,c,precision);
    for i = 1:n
        V = now.aux{1}(:,:,i);
        S = now.aux{2}(:,:,i);
        
        dLdC(:) = dzdy(1,1,:,i);
        diag_S = diag(S);
        
        ind =diag_S  > 1e-10; 
%         Dmin = min(find(ind, 1, 'last'), n+1);
        Dmin = min(find(ind, 1, 'last'));
        
        dLdV(:, 1:Dmin) = 2 *  symmetric(dLdC) * V(:, 1:Dmin) * ...
            diag(sqrt(diag_S(1:Dmin)+ layer.eps));  

        dLdS(1:Dmin, 1:Dmin) =   0.5*diag((diag_S(1:Dmin) + layer.eps).^(-0.5)) * ...
            ( V(:, 1:Dmin)' * dLdC * V(:, 1:Dmin));
         
        K(1:Dmin, 1:Dmin) = diag_S(1:Dmin)  * ones(1, Dmin);
        K(1:Dmin, 1:Dmin)  = 1 ./ (K(1:Dmin, 1:Dmin)  - K(1:Dmin, 1:Dmin)');
        K(isinf(K)) = 0;
            
        dLdY =  symmetric( V(:, 1:Dmin) * (diag(diag( dLdS(1:Dmin, 1:Dmin) ))  +  ...
            K(1:Dmin, 1:Dmin)' .* (V(:, 1:Dmin)' * dLdV(:, 1:Dmin)) ) * V(:, 1:Dmin)' );       
        
        pre.dzdx(:,:,i) = dLdY;
    end
    if gpuMode
        pre.dzdx = gpuArray(single(pre.dzdx));
    end
end

