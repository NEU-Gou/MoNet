function [ pre ] = nnvsqrt_backward( layer, pre, now )
% layer.eps - regular term
% pre.x     - [h,w,c,n]
% pre.aux{1}    - U
% pre.aux{2}    - S
% pre.aux{3}    - V
% now.x     - [h,w,c,n]
    
%     up_limit = layer.eps^(-0.5); % upper limit to mimic the regularization
    
    precision = class(now.aux{1});
    [h,w,c,n] = size(pre.x);
    dzdy = now.dzdx;
    gpuMode = isa(dzdy, 'gpuArray');
    if gpuMode
        dzdy = gather(dzdy);
%         xin = gather(xin);
    end
    
    if strcmp(precision,'double')
        dzdy = double(dzdy);
%         xin = double(xin);
    end
    if gpuMode
        pre.dzdx = gpuArray(zeros(h,w,c,n,'single'));
%         dLdY = gpuArray(zeros(h*w,c,precision));
    else
        pre.dzdx = zeros(h,w,c,n,'single');
%         dLdY = zeros(h*w,c,preci/sion);
    end

        
    dLdY = zeros(h*w,c,precision);
    dLdS = zeros(h*w,c,precision);
    dLdV = zeros(c,c,precision);
    K = zeros(c,c,precision);
    
    if h*w > c
        A = cat(1,eye(c),zeros(h*w-c,c));
    else
        A = cat(2,eye(h*w),zeros(h*w,c-h*w));
    end
    for i = 1:n

    
        U = now.aux{1}(:,:,i);
        S = now.aux{2}(:,:,i);
        V = now.aux{3}(:,:,i);
        
        diag_S = diag(S);% + layer.eps^(1/2);
        ind = find(diag_S > 0, 1, 'last'); 
        
        dLdY(:) = dzdy(:,:,:,i);
        
%         diag_S_der = diag_S(1:ind).^(-0.5);
%         diag_S_der = min(diag_S_der,up_limit);
        diag_S_der = diag_S(1:ind).*(diag_S(1:ind).^2 + layer.eps).^(-0.75);
        
        % compute dLdS and dLdV
        if h*w > c
            dLdS(1:ind,1:ind) = 0.5*diag(diag_S_der)*A(:,1:ind)' * dLdY * V(:,1:ind);
            dLdV(:,1:ind) = dLdY'* A(:,1:ind) * diag((diag_S(1:ind).^2 + layer.eps))^(0.25);
        else
            dLdS(1:ind,1:ind) = 0.5 * diag(diag_S_der) * dLdY * V(:,1:ind) * A(:,1:ind)';
            dLdV(:,1:ind) = dLdY' * diag((diag_S(1:ind).^2 + layer.eps))^(0.25) * A(:,1:ind);
        end
        K(1:ind,1:ind) = diag_S(1:ind).^2  * ones(1,ind);
        K(1:ind,1:ind) = 1 ./ (K(1:ind,1:ind)  - K(1:ind,1:ind)');
        K(isinf(K)) = 0;
        
        if h*w > c
            dLdX = U(:,1:ind)*(A(1:ind,1:ind)*diag(diag(dLdS(1:ind,1:ind))) + ...
                2*A(1:ind,1:ind)*diag(diag_S(1:ind))*(symmetric(K(1:ind,1:ind)'.*...
                (V(:,1:ind)'*dLdV(:,1:ind)))))*V(:,1:ind)';
        else
            dLdX = U(:,1:ind)*(diag(diag(dLdS(1:ind,1:ind))) * A(1:ind,1:ind) + ...
                2*diag(diag_S(1:ind))*A(1:ind,1:ind)*(symmetric(K(1:ind,1:ind)'.*...
                (V(:,1:ind)'*dLdV(:,1:ind)))))*V(:,1:ind)';
        end
%         dLdS = A * 0.5*diag(diag_S.^(-0.5))*A' * dLdY * V;
%         dLdV = dLdY'* A * diag(diag_S.^(0.5));
%         
%         K = diag_S.^2  * ones(1,numel(diag_S));
%         K = 1 ./ (K  - K');
%         K(isinf(K)) = 0;
        
%         dLdX = U*(A*diag(diag(dLdS)) + 2*A*diag(diag_S)*(symmetric(K'.*(V'*dLdV))))*V';
        
        pre.dzdx(:,:,:,i) = permute(reshape(dLdX,h,w,1,c),[1,2,4,3]);
    end
    %----debug:Gou-----%
%     fprintf('MAXBP:%e\n',max(abs(reshape(pre.dzdx(:,:,2:end,:),[],1)))./sqrt(h*w));
end

