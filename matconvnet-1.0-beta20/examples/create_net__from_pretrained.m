function net = create_net_from_pretrained(pretrainedPath)
    net = []; % your network structure

    % load pre-trained network
    old_net = load(pretrainedPath);

    % add dropout layers in network (saved model has dropout removed)
    drop1 = struct('name', 'dropout6', 'type', 'dropout', 'rate' , 0.5) ;
    drop2 = struct('name', 'dropout7', 'type', 'dropout', 'rate' , 0.5) ;
    old_net.layers = [old_net.layers(1:33) drop1 old_net.layers(34:35) drop2 old_net.layers(36:end)] ;

    % ignore classification and last softmax layers (we will insert our own)
    net.layers = old_net.layers(1:end-2);

    % add our own conv layer and loss layer
    % I use the add_block() function available in matconvnet/examples/cnn_imagenet_init.m
    net = add_block(net, opts, '8', 1, 1, 4096, 200, 1, 0) ; % 200-way fully connected layer fc8
    net.layers(end) = [] ; % remove ReLU layer that gets added after conv layer
    net.layers{end+1} = struct('type', 'softmaxloss') ; % add loss layer
end