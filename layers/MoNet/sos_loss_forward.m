function [ now ] = sos_loss_forward(layer, pre, now)
%SOS_LOSS_FORWARD forward pass for sos loss layer
%   layer: information and parameters
%        .U - singular vector
%        .S - singular values ^ (-0.5)
%        .class - class labels
%   pre : input
%   Y: loss 
% C = layer.class;
xin = pre.x;
[h,w,c,n] = size(xin);
loss_perC =  gpuArray.zeros(h,w,numel(layer.M),n,'single');
% loss_perC = zeros(n,numel(layer.M));
% loss_perC = gpuArray(loss_perC);
% loss_perC = zeros(h,w,n,numel(layer.U));

% replabel = repmat(C,1,h*w)';
% replabel = replabel(:);
% binarylabel = ones(numel(replabel),1)*(-1);
xin=permute(pre.x, [1,2,4,3]); % size is h, w, n, c
xin=reshape(xin, h*w*n, c);

for i = 1:numel(layer.M)
    tmp_q = sum((xin*layer.US{i}).^2,2);
%     loss = tmp_q';
    loss = log(tmp_q');
%     tmp_l = binarylabel;
%     tmp_l(replabel==i) = 1;
%     loss = tmp_l .* tmp_q;
%     loss = reshape(loss,h,w,n);
%     loss_perC(:,:,:,i) = loss;

%     loss = reshape(loss,h*w,n);
%     loss_perC(:,i) = sum(loss,1)'; 

    loss_perC(:,:,i,:) = reshape(loss,h,w,1,n);
end
now.x = loss_perC;
% Y = sum(loss_perC,2);
% now.x = reshape(loss_perC',1,1,size(loss_perC,2),size(loss_perC,1));

