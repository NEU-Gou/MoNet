function [ now ] = nnlogmat_forward( layer, pre, now )
%nnsqrtmat_forward matrix wise log
%   layer.eps   - regular term
%   pre.x       - [c,c,n]
%   now.x       - [1,1,c*c,n]
    
    precision = 'double'; % 'single'
    
    xin = pre.x;
    [c,c,n] = size(xin);
    
    if strcmp(precision, 'double')
        xin = double(xin);
    end
    S = gpuArray(zeros(c,c,n,precision)); % eigen value
    V = gpuArray(zeros(c,c,n,precision)); % eigen vector
    
%     S_ = zeros(c,c,precision);
%     V_ = zeros(c,c,precision);
    
    now.x = gpuArray(zeros(1,1,c*c,n,'single'));
    for i = 1:n
        [V_, S_] = svd(xin(:,:,i));
        diag_S = diag(S_);
%         [diag_S, idx] = sort(diag_S, 'descend');
%         V_ = V_(:, idx);         
        ind = diag_S  > 1e-10; 
%         Dmin = min(find(ind, 1, 'last'), n+1);
        Dmin = min(find(ind,1,'last'));
        V(:, 1:Dmin, i) = V_(:, 1:Dmin);
        S(1:Dmin, 1:Dmin, i) = diag(diag_S(1:Dmin));
        % square root
        now.x(1,1,:,i) = reshape(V(:, 1:Dmin, i) * diag(log((diag_S(1:Dmin)+layer.eps)))...
                            * V(:, 1:Dmin, i)',1,[]);
    end
    now.aux{1} = V;
    now.aux{2} = S;
        
end

