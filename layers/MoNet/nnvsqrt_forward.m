function now = nnvsqrt_forward(layer, pre, now)
% now = nnvsqrt_forward(layer, pre, now) 
% compute Vp = V*sqrt(sigma') , s.t. Vp'*Vp = sqrtm(V'*V)
% layer.eps     - regular term
% pre.x         - [h,w,c,n]
% now.aux{1}    - S
% now.aux{2}    - V
% now.x         - [h,w,c,n]

    precision = 'double'; % 'single'
    xin = pre.x;
    [h,w,c,n] = size(xin);
    gpuMode = isa(xin, 'gpuArray');
    
    if gpuMode
        xin = gather(xin);
    end
    if strcmp(precision, 'double')
        xin = double(xin);
    end
    
    S = zeros(h*w,c,n,precision); % singular value
    U = zeros(h*w,h*w,n,precision); % left singular vector
    V = zeros(c,c,n,precision); % right singular vector
    
    if gpuMode
%         S = gpuArray(zeros(h*w,c,n,precision)); % singular value
%         U = gpuArray(zeros(h*w,h*w,n,precision)); % left singular vector
%         V = gpuArray(zeros(c,c,n,precision)); % right singular vector
        now.x = gpuArray(zeros(h,w,c,n,'single'));
    else

        now.x = zeros(h,w,c,n,'single');
    end
    for i = 1:n
        tmpx = permute(xin(:,:,:,i),[1,2,4,3]);
%         if h*w < c
%             error('Input dimension is too large!');
%         else
            [U_,S_,V_] = svd(reshape(tmpx,[],c));
%         end
%         % sign stability
%         [~,idx] = max(min(abs(V_(:,:,1)),[],2)); % stable row
%         U_(:,1:c) = bsxfun(@times, sign(V_(idx,:)), U_(:,1:c));
%         V_ = bsxfun(@times, sign(V_(idx,:)), V_);
        
        
%         ind = find(cumsum(diag(S_))./sum(diag(S_)) > 1-1e-3, 1, 'first'); 
        tol = 1e-5;%max(h*w,c)*eps(S_(1,1)); %
        ind = find((diag(S_)) > tol,1,'last');
        %----- debug: Gou ------%
%         fprintf('MAX:%e; MIN:%e: CUT: %e\n',S_(1,1),S_(ind,ind),...
%             S_(min(ind+1,c),min(ind+1,c)));
        
        S(1:ind,1:ind,i) = S_(1:ind,1:ind);
        V(:,1:ind,i) = V_(:,1:ind);
        U(:,1:ind,i) = U_(:,1:ind);
        
        tmpy = zeros(size(S_));
        tmpy(1:ind,:) = (S_(1:ind,1:ind).^2 + layer.eps*eye(ind)).^(0.25)*V_(:,1:ind)'; % with regularize
%         tmpy = (S_.^2 + layer.eps*eye(size(S_))).^(0.25)*V_';
        now.x(:,:,:,i) = permute(reshape(tmpy,[h,w,1,c]),[1,2,4,3]);
    end
    now.aux{1} = U;
    now.aux{2} = S;
    now.aux{3} = V;
end
        
        