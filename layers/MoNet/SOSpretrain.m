function M = SOSpretrain(dataset, network, net, endLayer, order, mopts)

% endLayer=consts(dataset, 'endLayer', 'network', network);
[train_F, train_l, ~, ~] = get_activations_dataset_network_layer(dataset, network, endLayer, ...
    mopts.use448, net, mopts.batchSize);

thres = size(train_F,3)
for i = 1:max(train_l)
    tmpF = permute(train_F(:,:,:,train_l==i),[1,2,4,3]);
    [h,w,n,c] = size(tmpF);
    tmpF = reshape(tmpF,[],size(tmpF,4));
%     %% first round for background modeling
%     tmpM = (tmpF'*tmpF)./size(tmpF,1);
%     [u,s,~] = svd(tmpM);
%     Q = sum((tmpF*u*s^(-0.5)).^2,2);
%     tmpF = tmpF(Q > thres,:); 
    %% second round for foreward modeling
    M.M{i} = (tmpF'*tmpF)./size(tmpF,1);
    [u,s,~] = svd(M.M{i});
    M.U{i} = u;
    M.S{i} = s^(-0.5);
%     M.S{i} = tmpM.S^(-0.5);
    M.US{i} = M.U{i}*M.S{i};
    M.num_sample{i} = size(tmpF,1);
end
    