function [ pre ] = veronese_backward(layer, pre, now)
%veronese_backward Back prop for veronese layer
%   pre.x   - [h,w,c,n]
%   now.x   - [h,w,c2,n]
%   layer.order - order
%   layer.powers - veronese power

    dzdy = now.dzdx;
    [h,w,cv,n] = size(dzdy);
    
    if layer.order == 1
        pre.dzdx = dzdy(:,:,2:end,:)./sqrt(w*h);
        return
    elseif layer.order == 0
        pre.dzdx = dzdy./sqrt(w*h);
        return
    end

    if isa(dzdy, 'gpuArray')
        gpuMode = 1;
    else 
        gpuMode = 0;
    end
    
    dzdy = permute(dzdy, [1,2,4,3]);
    dzdy = reshape(dzdy,[],cv)';
    xin = pre.x;
    [~,~,c,~] = size(xin);
    xin = permute(xin, [1,2,4,3]);
    xin = reshape(xin, [], c)';
%     xin = reshape(xin,1,size(xin,1),size(xin,2));
%     xin = cat(2,ones(size(xin,1),1),xin)';
    
    weights = layer.powers;
    weights(:,1) = []; % padded 1 at beginning
%     powers = weights-1;
%     powers(powers==-1)=0;
    
    if gpuMode
        pre.dzdx = gpuArray(zeros(h,w,c,n,'single'));
    else
        pre.dzdx = zeros(h,w,c,n,'single');
    end
    % per channel
    for i = 1:c         
        dYdXi_p = weights;
        dYdXi_p(:,i) = dYdXi_p(:,i) - 1;
        dYdXi_p(dYdXi_p<0)=0;
%         numerical 
        xin(abs(xin)<1e-39)=1e-39;
%         power
        dYdXi = real(exp(dYdXi_p*log(complex(xin))));
        dYdXi = bsxfun(@times, dYdXi, weights(:,i));
        
% %%%%%%%%% debug --- Gou %%%%%%%%
%         dYdXi = bsxfun(@times, squeeze(prod(bsxfun(@power, xin,dYdXi_p),2)), weights(:,i));
% %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%        
        tmp =sum(dzdy.*dYdXi,1);
        tmp = reshape(tmp,1,h,w,n);
        pre.dzdx(:,:,i,:) = permute(tmp,[2,3,1,4]);
    end
    pre.dzdx = pre.dzdx./sqrt(w*h);
end