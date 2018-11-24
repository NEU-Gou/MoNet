function [ pre, layer ] = sos_loss_backward( layer, pre, now )
%SOS_LOSS_BACKWARD backward function for SOS loss layer
%   layer: information and parameters
%        .U - singular vector
%        .S - singular values ^ (-0.5)
%        .M - empirical moment matrix
%        .num_sample - number of samples used to compute M
%        .l - class labels
%   pre : previous layer
    dzdy = squeeze(now.dzdx);
    xin = pre.x;
    C = layer.class;
    pre.dzdx = gpuArray.zeros(size(xin),'single');
    [h,w,c,n] = size(xin);
%     replabel = repmat(C,1,h*w)';
%     replabel = replabel(:);
%     binarylabel = ones(numel(replabel),1)*(-1);
    xin=permute(pre.x, [1,2,4,3]); % size is h, w, n, c
    xin=reshape(xin, h*w*n, c);
    for i = 1:numel(layer.M)
%         tmp_q = 2*xin*layer.Minv{i};
        tmp_q = 2*xin*layer.US{i}*layer.US{i}'; % key equation
        tmp_q = bsxfun(@times, tmp_q, 1./sum((xin*layer.US{i}).^2,2));
        tmp_dzdx = tmp_q;
%         tmp_l = binarylabel;
%         tmp_l(replabel==i) = 1;
%         tmp_dzdx = tmp_l.*tmp_q;
        tmp_dzdx = reshape(tmp_dzdx, h,w,n,c);
        tmp_dzdx = permute(tmp_dzdx, [1,2,4,3]);
%%%%%%%%%%%% multiply the ys
%         tmp_dzdx = reshape(tmp_dzdx, [],n);
%         tmp_dzdx = bsxfun(@times, tmp_dzdx, dzdy(i,:));
%         tmp_dzdx = reshape(tmp_dzdx, h,w,c,n);
%%%%%%%%%%%% followed by ip layer
        tmp_dzdx = bsxfun(@times, tmp_dzdx, dzdy(:,:,i,:));

        pre.dzdx = pre.dzdx + tmp_dzdx;
    end

    % update M
    for i = 1:numel(layer.M)
        if sum(C==i) == 0
            continue;
        end
        tmpx = pre.x(:,:,:,C==i);
        tmpx = permute(tmpx, [1,2,4,3]);
        tmpx = reshape(tmpx,[],c);
        tmp_num = size(tmpx,1);
        M_p = (tmpx' * tmpx)./tmp_num;
        layer.M{i} = (layer.num_sample{i}/(layer.num_sample{i} +tmp_num))*layer.M{i} + ...
                    (tmp_num/(layer.num_sample{i} + tmp_num))*M_p;
        % TODO: directly update u,s
        [u,s,~] = svd(layer.M{i} + eye(size(layer.M{i},1))*1e-5);
        layer.US{i} = u*s^(-0.5);
%         layer.U{i} = u;
%         layer.S{i} = s^(-0.5);
        layer.num_sample{i} = layer.num_sample{i} + tmp_num;
    end
end

