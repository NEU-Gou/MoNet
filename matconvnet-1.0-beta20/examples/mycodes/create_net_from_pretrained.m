function net = create_net_from_pretrained(opts,imdb)
    net = []; % your network structure

    % load pre-trained network
    old_net = load(opts.pretrainedPath);

    % add dropout layers in network (saved model has dropout removed)
    drop1 = struct('name', 'dropout6', 'type', 'dropout', 'rate' , 0.5) ;
    drop2 = struct('name', 'dropout7', 'type', 'dropout', 'rate' , 0.5) ;
    old_net.layers = [old_net.layers(1:33) drop1 old_net.layers(34:35) drop2 old_net.layers(36:end)] ;

    % ignore classification and last softmax layers (we will insert our own)
    net.layers = old_net.layers(1:end-2);

    % add our own conv layer and loss layer
    % I use the add_block() function available in matconvnet/examples/cnn_imagenet_init.m
    net = add_block(net, opts, '8', 1, 1, 4096, 67, 1, 0) ; % 200-way fully connected layer fc8
    net.layers(end) = [] ; % remove ReLU layer that gets added after conv layer
    net.layers{end+1} = struct('type', 'softmaxloss') ; % add loss layer
    
    % Set the class names in the network
    net.meta.classes.name = imdb.images.name ;
    %net.meta.classes.description = imdb.classes.description ;

    net.meta.normalization.imageSize = [224, 224, 3] ;
    net.meta.normalization.averageImage = [] ;
    net.meta.augmentation.rgbVariance = zeros(0,3) ;
    net.meta.trainOpts.batchSize = 64 ;
    net.meta.normalization.border = 256 - net.meta.normalization.imageSize(1:2) ;
    net.meta.augmentation.transformation = 'stretch' ;
    
    % Compute image statistics (mean, RGB covariances, etc.)
    imageStatsPath = fullfile(opts.expDir, 'imageStats.mat') ;
    if exist(imageStatsPath)
      load(imageStatsPath, 'averageImage', 'rgbMean', 'rgbCovariance') ;
    else
      [averageImage, rgbMean, rgbCovariance] = getImageStats(opts, net.meta, imdb) ;
      save(imageStatsPath, 'averageImage', 'rgbMean', 'rgbCovariance') ;
    end

    % Set the image average (use either an image or a color)
    %net.meta.normalization.averageImage = averageImage ;
    net.meta.normalization.averageImage = rgbMean ;

    % Set data augmentation statistics
    [v,d] = eig(rgbCovariance) ;
    net.meta.augmentation.rgbVariance = 0.1*sqrt(d)*v' ;
    clear v d ;
end


% --------------------------------------------------------------------
function net = add_block(net, opts, id, h, w, in, out, stride, pad)
% --------------------------------------------------------------------
    info = vl_simplenn_display(net) ;
    fc = (h == info.dataSize(1,end) && w == info.dataSize(2,end)) ;
    if fc
      name = 'fc' ;
    else
      name = 'conv' ;
    end
    convOpts = {'CudnnWorkspaceLimit', opts.cudnnWorkspaceLimit} ;
    net.layers{end+1} = struct('type', 'conv', 'name', sprintf('%s%s', name, id), ...
                               'weights', {{init_weight(opts, h, w, in, out, 'single'), ...
                                 ones(out, 1, 'single')*opts.initBias}}, ...
                               'stride', stride, ...
                               'pad', pad, ...
                               'learningRate', [1 2], ...
                               'weightDecay', [opts.weightDecay 0], ...
                               'opts', {convOpts}) ;
    if opts.batchNormalization
      net.layers{end+1} = struct('type', 'bnorm', 'name', sprintf('bn%s',id), ...
                                 'weights', {{ones(out, 1, 'single'), zeros(out, 1, 'single'), ...
                                   zeros(out, 2, 'single')}}, ...
                                 'learningRate', [2 1 0.3], ...
                                 'weightDecay', [0 0]) ;
    end
    net.layers{end+1} = struct('type', 'relu', 'name', sprintf('relu%s',id)) ;

end

% -------------------------------------------------------------------------
function weights = init_weight(opts, h, w, in, out, type)
% -------------------------------------------------------------------------
% See K. He, X. Zhang, S. Ren, and J. Sun. Delving deep into
% rectifiers: Surpassing human-level performance on imagenet
% classification. CoRR, (arXiv:1502.01852v1), 2015.

    switch lower(opts.weightInitMethod)
      case 'gaussian'
        sc = 0.01/opts.scale ;
        weights = randn(h, w, in, out, type)*sc;
      case 'xavier'
        sc = sqrt(3/(h*w*in)) ;
        weights = (rand(h, w, in, out, type)*2 - 1)*sc ;
      case 'xavierimproved'
        sc = sqrt(2/(h*w*out)) ;
        weights = randn(h, w, in, out, type)*sc ;
      otherwise
        error('Unknown weight initialization method''%s''', opts.weightInitMethod) ;
    end
end

% -------------------------------------------------------------------------
function [averageImage, rgbMean, rgbCovariance] = getImageStats(opts, meta, imdb)
% -------------------------------------------------------------------------
    train = find(imdb.images.set == 1) ;
    train = train(1: 101: end);
    bs = 256 ;
    opts.networkType = 'simplenn' ;
    fn = getBatchFn(opts, meta) ;
    avg = {}; rgbm1 = {}; rgbm2 = {};

    for t=1:bs:numel(train)
      batch_time = tic ;
      batch = train(t:min(t+bs-1, numel(train))) ;
      fprintf('collecting image stats: batch starting with image %d ...', batch(1)) ;
      temp = fn(imdb, batch) ;
      z = reshape(permute(temp,[3 1 2 4]),3,[]) ;
      n = size(z,2) ;
      avg{end+1} = mean(temp, 4) ;
      rgbm1{end+1} = sum(z,2)/n ;
      rgbm2{end+1} = z*z'/n ;
      batch_time = toc(batch_time) ;
      fprintf(' %.2f s (%.1f images/s)\n', batch_time, numel(batch)/ batch_time) ;
    end
    averageImage = mean(cat(4,avg{:}),4) ;
    rgbm1 = mean(cat(2,rgbm1{:}),2) ;
    rgbm2 = mean(cat(3,rgbm2{:}),3) ;
    rgbMean = rgbm1 ;
    rgbCovariance = rgbm2 - rgbm1*rgbm1' ;
end