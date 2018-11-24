function net = initializeNetworkSharedWeights(imdb, encoderOpts, opts)
% Initialize the network structure
% Avaliable options for opt.network:
%     'g2deNet'   - proposed by Wang et.al in CVPR 2017
%     'MoNet'     - proposed network with bilinear pooling
%     'MoNet-TS'  - proposed network with compact pooling (TS)
%     'MoNet-2U'  - remove sub-matrix square-root and homogeneouse mapping layer
%     'MoNet-2U-TS'- same as above with compact pooling
%     'MoNet-2'   - remove homogeneous mapping layer
%     'MoNet-2-TS'- same as above with compact pooling
%     'MoNet-U'   - remove sub-matrix square-root layer
%     'MoNet-U-TS'- same as above with compact pooling
% Modified by Mengran Gou @ 2018/07
% ----------------------------Original info--------------------------------
% This code is used for initializing Global Gaussian Distribution 
% Embeding Network.  

% Qilong Wang, Peihua Li, Lei Zhang. G^2DeNet: Global Gaussian 
% Distribution Embeding Network and Its Application to Visual
% Recognition. In CVPR, 2017.

% This file is part of the G2DeNet and is modified from code 
% of BCNN (https://bitbucket.org/tsungyu/bcnn).

% Copyright (C) 2017 Qilong Wang, Peihua Li, Lei Zhang.
% All rights reserved.
% -------------------------------------------------------------------------

scal = 1 ;
init_bias = 0.1;
numClass = length(imdb.classes.name);


assert(strcmp(encoderOpts.modela, encoderOpts.modelb), 'neta and netb are required to be the same');
assert(~isempty(encoderOpts.modela), 'network is not specified');

% Load the model
net = load(encoderOpts.modela);
net.meta.normalization.keepAspect = opts.keepAspect;

% truncate the network
maxLayer = max(encoderOpts.layera, encoderOpts.layerb);
net.layers = net.layers(1:maxLayer);

% get the feature dimension for both layers
netInfo = vl_simplenn_display(net);
mapSize1 = netInfo.dataSize(3, encoderOpts.layera+1);
mapSize2 = netInfo.dataSize(3, encoderOpts.layerb+1);


for l=numel(net.layers):-1:1
    if strcmp(net.layers{l}.type, 'conv')
        net.layers{l}.opts = {'CudnnWorkspaceLimit', opts.cudnnWorkspaceLimit};
    end
end

% add batch normalization
if opts.batchNormalization
    for l=numel(net.layers):-1:1
        if isfield(net.layers{l}, 'weights')
            ndim = size(net.layers{l}.weights{1}, 4);
            
            layer = struct('type', 'bnorm', 'name', sprintf('bn%s',l), ...
                'weights', {{ones(ndim, 1, 'single'), zeros(ndim, 1, 'single'), [zeros(ndim, 1, 'single'), ones(ndim, 1, 'single')]}}, ...
                'learningRate', [2 1 0.05], ...
                'weightDecay', [0 0]) ;
            
            
            net.layers = horzcat(net.layers(1:l), layer, net.layers(l+1:end)) ;
            
        end
    end
    net = simpleRemoveLayersOfType(net,'lrn');
end

% network setting
net = vl_simplenn_tidy(net) ;

switch opts.network
    case 'g2deNet'
        net.layers{end+1} = struct('type', 'fsqrt', 'name', 'fea_sqrt');  % feature normalization

        % add global Gaussian embedding layer
        if(encoderOpts.layera==encoderOpts.layerb)
            net.layers{end+1} = struct('type', 'GlbGaussian', 'name', 'GGE');
        else
            error('Not support now~');
        end

        % stack normalization
        net.layers{end+1} = struct('type', 'sqrt', 'name', 'sqrt_norm');
        net.layers{end+1} = struct('type', 'l2norm', 'name', 'l2_norm');
        outDim = (mapSize1+1)*(mapSize2+1);
    case 'MoNet-2U' % equal to BCNN
        net = addBilinear(net);
        net = addSqrt(net);
        net = addL2norm(net);
        outDim = mapSize1*mapSize2;
    case 'MoNet-2U-TS' % equal to compact-BCNN
        net = addCompactTS(net, opts.projDim, mapSize1, 0);
        net = addSqrt(net);
        net = addL2norm(net);
        outDim = opts.projDim;
    case 'MoNet-2' % equal to iBCNN
        net = addVsqrt(net,0);
        net = addBilinear(net);
        net = addSqrt(net);
        net = addL2norm(net);
        outDim = mapSize1*mapSize2;
    case 'MoNet-2-TS' 
        net = addVsqrt(net,0);
        net = addCompactTS(net, opts.projDim, mapSize1, 0);
        net = addSqrt(net);
        net = addL2norm(net);
        outDim = opts.projDim;
    case 'MoNet-U'
	net = addfSqrt(net);
        net = addVero(net, mapSize1, 1);
        net = addBilinear(net);
        net = addSqrt(net);
        net = addL2norm(net);
        outDim = (mapSize1+1)*(mapSize2+1);
    case 'MoNet-U-TS'
        net = addfSqrt(net);
        net = addVero(net, mapSize1, 1);
        verodim = nchoosek(mapSize1+1,1);
        net = addCompactTS(net, opts.projDim, verodim, 0);
        net = addSqrt(net);
        net = addL2norm(net);
        outDim = opts.projDim;
    case 'MoNet'
        net = addfSqrt(net);
        net = addVero(net, mapSize1, 1);
        net = addVsqrt(net,0);
        net = addBilinear(net);
        net = addSqrt(net);
        net = addL2norm(net);
        outDim = (mapSize1+1)*(mapSize2+1);
    case 'MoNet-TS'
        net = addfSqrt(net);
        net = addVero(net, mapSize1, 1);
        net = addVsqrt(net,0);
        verodim = nchoosek(mapSize1+1,1);
        net = addCompactTS(net, opts.projDim, verodim, 0);
        net = addSqrt(net);
        net = addL2norm(net);
        outDim = opts.projDim;
%     case '1stMoment_vsqrt_TS_l2lp'
%         net = addfSqrt(net);
%         net = addVero(net, mapSize1, 1);
%         net = addVsqrt(net,0);
%         verodim = nchoosek(mapSize1+1,1);
%         net = addCompactTS(net, opts.projDim, verodim, 0);
%         net = addSqrt(net);
%         net = addL2norm(net);
%         net = addVero(net, opts.projDim, 1);
%         outDim = opts.projDim + 1;
%     case '1stMoment_vsqrt_inhomoTS'
%         net = addfSqrt(net);
%         net = addVero(net, mapSize1, 1);
%         net = addVsqrt(net,0);
%         verodim = nchoosek(mapSize1+1,1);
%         net = addVero(net, verodim, 1);
%         net = addCompactTS(net, opts.projDim, verodim+1, 0);
%         net = addSqrt(net);
%         net = addL2norm(net);
%         outDim = opts.projDim;        
%     case '2ndMoment_vsqrt'
%         pcadim = 38;
%         net=addfSqrt(net);
%         net = addPCA(net, mapSize1, pcadim, opts, imdb);    
% %         net = addConvT(net, 2, pcadim, pcadim, 2, 0, 0);
%         net = addVero(net, pcadim, 2);
%         net = addVsqrt(net,1e-3);
%         verodim = nchoosek(pcadim+2,2);
%         net = addBilinear(net);
%         net=addSqrt(net);
%         net=addL2norm(net);
%         outDim=verodim^2;
%     case '2ndMoment_vsqrt_TS'
%         pcadim = 60;
%         net=addfSqrt(net);
%         net = addPCA(net, mapSize1, pcadim, opts, imdb);
%         net = addConvT(net, 2, pcadim, pcadim, 2, 0, 0);
%         net = addVero(net, pcadim, 2);
%         net = addVsqrt(net,1e-4);
%         verodim = nchoosek(pcadim+2,2);
%         net=addCompactTS(net, opts.projDim, verodim, 0);
%         net=addSqrt(net);
%         net=addL2norm(net);
%         outDim=opts.projDim;
end

% network setting
net = vl_simplenn_tidy(net) ;

% build a linear classifier netc
initialW = 0.001/scal * randn(1,1,outDim,numClass,'single');
initialBias = init_bias.*ones(1, numClass, 'single');
netc.layers = {};
if ~isempty(strfind(opts.network, '_l2lp'))
    initialW = 0.001/scal * randn(100, outDim, numClass,'single');
     netc.layers{end+1} = struct('type', 'custom', 'name', 'classifier_l2', ...
        'forward',@nnl2lp_forward,...
        'backward',@nnl2lp_backward, ...
        'weights', {{initialW}}, ...
        'learningRate', [100 100], ...
        'weightDecay', [0 0]) ;
else
    netc.layers{end+1} = struct('type', 'conv', 'name', 'classifier', ...
        'weights', {{initialW, initialBias}}, ...
        'stride', 1, ...
        'pad', 0, ...
        'learningRate', [1000 1000], ...
        'weightDecay', [0 0]) ;
end
netc.layers{end+1} = struct('type', 'softmaxloss', 'name', 'loss') ;
netc = vl_simplenn_tidy(netc) ;


% pretrain the linear classifier with logistic regression
if(opts.bcnnLRinit && ~opts.fromScratch)
    
    % get bcnn feature for train and val sets
    train = find(imdb.images.set==1);%find(imdb.images.set==1|imdb.images.set==2);
    if ~exist(opts.nonftGlbGauDir, 'dir')
        netInit = net;
        
        if ~isempty(opts.train.gpus)
            netInit = vl_simplenn_move(netInit, 'gpu') ;
        end
        
        batchSize = 16;
        
        bopts = netInit.meta.normalization ;
        bopts.numThreads = opts.numFetchThreads ;
        bopts.transformation = 'none' ;
        bopts.rgbVariance = [] ;
        bopts.scale = opts.imgScale;
        
        
        getBatchFn = getBatchSimpleNNWrapper(bopts);
        
        mkdir(opts.nonftGlbGauDir)
        
        % compute and cache the G2DeNet features
        for t=1:batchSize:numel(train)
            fprintf('Initialization: extracting bcnn feature of batch %d/%d\n', ceil(t/batchSize), ceil(numel(train)/batchSize));
            batch = train(t:min(numel(train), t+batchSize-1));
            [im, labels] = getBatchFn(imdb, batch) ;
            if opts.train.prefetch
                nextBatch = train(t+batchSize:min(t+2*batchSize-1, numel(train))) ;
                getBatcFn(imdb, nextBatch) ;
            end
            im = im{1};
            if ~isempty(opts.train.gpus)
                im = gpuArray(im) ;
            end
            
            net.layers{end}.class = labels ;
            
            res = [] ;
            res = vl_GlbGaussian_nn(netInit, im, [], res, ...
                'accumulate', false, ...
                'mode', 'test', ...
                'conserveMemory', true, ...
                'sync', true, ...
                'cudnn', opts.cudnn) ;
            codeb = squeeze(gather(res(end).x));
            for i=1:numel(batch)
                code = codeb(:,i);
                savefast(fullfile(opts.nonftGlbGauDir, ['GlbGau_nonft_', num2str(batch(i), '%05d')]), 'code');
            end
        end
    end
    
    clear code res netInit
    
    % get the pretrain linear classifier
    if exist(fullfile(opts.expDir, 'initial_fc.mat'), 'file')
        load(fullfile(opts.expDir, 'initial_fc.mat'), 'netc') ;
    else
        
        GlbGaudb = imdb;
        tempStr = sprintf('%05d\t', train);
        tempStr = textscan(tempStr, '%s', 'delimiter', '\t');
        GlbGaudb.images.name = strcat('GlbGau_nonft_', tempStr{1}');
        GlbGaudb.images.id = GlbGaudb.images.id(train);
        GlbGaudb.images.label = GlbGaudb.images.label(train);
        GlbGaudb.images.set = GlbGaudb.images.set(train);
        GlbGaudb.imageDir = opts.nonftGlbGauDir;
        
        %train logistic regression
        [netc, info] = cnn_train(netc, GlbGaudb, @getBatch_GlbGaudb_fromdisk, opts.inittrain, ...
            'conserveMemory', true);
        save(fullfile(opts.expDir, 'initial_fc.mat'), 'netc', '-v7.3') ;
    end
end

% set all parameters to random number if train the model from scratch
if(opts.fromScratch)
    for i=1:numel(net.layers)
        if ~strcmp(net.layers{i}.type, 'conv'), continue ; end
        net.layers{i}.learningRate = [1 2];
        net.layers{i}.weightDecay = [1 0];
        net.layers{i}.weights = {0.01/scal * randn(size(net.layers{i}.weights{1}), 'single'), init_bias*ones(size(net.layers{i}.weights{2}), 'single')};
    end
end

% stack netc on network
for i=1:numel(netc.layers)
    net.layers{end+1} = netc.layers{i};
end
%clear netc

% Rename classes
net.meta.classes.name = imdb.classes.name;
net.meta.classes.description = imdb.classes.name;

% add border for translation data jittering
if(~strcmp(opts.dataAugmentation{1}, 'f2') && ~strcmp(opts.dataAugmentation{1}, 'none'))
    net.meta.normalization.border = 256 - net.meta.normalization.imageSize(1:2) ;
end

end
 

function [im,labels] = getBatch_GlbGaudb_fromdisk(imdb, batch)
% -------------------------------------------------------------------------

im = cell(1, numel(batch));
for i=1:numel(batch)
    load(fullfile(imdb.imageDir, imdb.images.name{batch(i)}));
    im{i} = code;
end
im = cat(2, im{:});
im = reshape(im, 1, 1, size(im,1), size(im, 2));
labels = imdb.images.label(batch) ;
end


function layers = simpleFindLayersOfType(net, type)
% -------------------------------------------------------------------------
layers = find(cellfun(@(x)strcmp(x.type, type), net.layers)) ;
end

% -------------------------------------------------------------------------
function net = simpleRemoveLayersOfType(net, type)
% -------------------------------------------------------------------------
layers = simpleFindLayersOfType(net, type) ;
net.layers(layers) = [] ;
end

%% network layer helper functions
function net=addfSqrt(net)
    net.layers{end+1} = struct('type', 'fsqrt', 'name', 'fea_sqrt');  % feature normalization
end

function net=addSqrt(net)
    net.layers{end+1} = struct('type', 'sqrt', 'name', 'sqrt_norm');
end

function net=addL2norm(net)
    net.layers{end+1} = struct('type', 'l2norm', 'name', 'l2_norm');
end

function net=addBilinear(net)
    net.layers{end+1}=struct('type', 'custom',...
        'forward', @yang_bilinear_forward, ...
        'backward', @yang_bilinear_backward, ...
        'name', 'bilinear');
end

function net = addVero(net, dim, order)
    powers = exponent(order, dim+1);
    net.layers{end+1} = struct('type', 'custom', ...
        'powers', powers, 'order', order,'name', 'veronese',...
        'forward', @veronese_forward, 'backward', @veronese_backward);
end

function net = addVsqrt(net,eps)
    net.layers{end+1} = struct('type','custom', ...
        'eps',eps, 'name', 'vect_sqrt', ...
        'forward', @nnvsqrt_forward, 'backward', @nnvsqrt_backward);
end
    

function net=addCompactTS(net, projDim, nc, learnW)
    TS_obj=yang_compact_bilinear_TS_nopool_2stream(projDim, [nc nc], 1);

    wf = sprintf('projectW_compact_bilinear_TS_%d_%d.mat',nc, projDim);
    wf = fullfile('data',wf);
    if exist(wf, 'file') ==2
        fprintf('loading an old compact2 weight\n');
        load(wf);
%         % modify the initialized parameters to the loaded params 
%         TS_obj.h_={hs(1,:), hs(2,:)};
%         TS_obj.weights_={ss(1,:), ss(2,:)};
%         TS_obj.setSparseM(TS_obj.weights_, 1);
    else
        save(wf,'TS_obj');
    end
    
    net.layers{end+1}=struct('type', 'custom',...
        'layerObj', TS_obj, ...
        'forward', @TS_obj.forward_simplenn, ...
        'backward', @TS_obj.backward_simplenn, ...
        'name', 'compact_TS', ...
        'outDim', projDim, ...
        'weights', {{cat(1, TS_obj.weights_{:})}}, ...
        'learningRate', [1]*learnW);
end

function net=addCompactRM(net, projDim, nc, learnW, dataset)
    wf = sprintf('projectW_compact_bilinear_RM_%d_%d.mat',nc, projDim);
    wf = fullfile('data',wf);
    
    if exist(wf, 'file') ==2
        fprintf('loading an old compact bilinear weight\n');
        load(wf);
    else
        factor=1.0/sqrt(projDim);
        fprintf('generating a new compact bilinear weight\n');
        init_w={{factor*(randi(2,nc, projDim)*2-3),...
                 factor*(randi(2,nc, projDim)*2-3)}};
        savefast(wf, 'init_w');
    end
    
    net.layers{end+1}=struct('type', 'custom',...
        'forward', @yang_compact_bilinear_RM_forward, ...
        'backward', @yang_compact_bilinear_RM_backward, ...
        'name', 'compact_RM', ...
        'weights', init_w,...
        'outDim', projDim, ...
        'learnW', learnW, ...
        'learningRate', [1 1]*learnW);
end

function net=addPCA(net, lastNchannel, projDim, opts, imdb)
    pcaOut=projDim;%    floor(sqrt(projDim));
%     assert(pcaOut^2==projDim);
    % pca is a filter of size: 1*1*lastNchannel*sqrtProjDim
    % with a bias of length sqrtProjDim. All in single. 
        
    pcaWeightFile = sprintf('pcaWeight_%s_VGG16_%d_%d.mat', opts.dataset, ...
        opts.imgScale*224,pcaOut);
    pcaWeightFile = fullfile('data',pcaWeightFile);
    if exist(pcaWeightFile, 'file') == 2
        fprintf('Load PCA weight from saved file: %s\n', pcaWeightFile);
        load(pcaWeightFile);
    else
        % get activations from the last conv layer % checkpoint
        [trainFV, trainY, valFV, valY]=...
            get_activations_dataset_network_layer(...
            opts.dataset, 'VGG16', 'pcaConv', opts.imgScale==2, net, ...
            opts.batchSize, imdb);
       
        samples=permute(trainFV, [1,2,4,3]); % from hwcn to hwnc
        samples=reshape(samples, [], size(samples, 4));
        ave=mean(samples, 1); % 1*dim vector
        coeff=pca(samples); % dim*dim matrix, each column is a principle direction
        bias=-ave*coeff; % a row vector, should be the initial value for bias
        
        coeff=single(coeff); 
        bias=single(bias);
        
        savefast(pcaWeightFile, 'coeff', 'bias');
        fprintf('save PCA weight to new file: %s\n', pcaWeightFile);
    end
    initPCAparam={{reshape(coeff(:, 1:pcaOut), 1, 1, lastNchannel, pcaOut),...
                   bias(1:pcaOut)}};
    
    net.layers{end+1} = struct('type', 'conv', 'name', 'pca_reduction', ...
       'weights', initPCAparam, ...
       'stride', 1, ...
       'pad', 0, ...
       'learningRate', [1 2]);
end
function f = bilinear_u(k, numGroups, numClasses)
%BILINEAR_U  Create bilinear interpolation filters
%   BILINEAR_U(K, NUMGROUPS, NUMCLASSES) compute a square bilinear filter
%   of size k for deconv layer of depth numClasses and number of groups
%   numGroups

factor = floor((k+1)/2) ;
if rem(k,2)==1
    center = factor ;
else
    center = factor + 0.5 ;
end
C = 1:k ;
if numGroups ~= numClasses
    f = zeros(k,k,numGroups,numClasses) ;
else
    f = zeros(k,k,1,numClasses) ;
end

for i =1:numClasses
    if numGroups ~= numClasses
        index = i ;
    else
        index = 1 ;
    end
    f(:,:,index,i) = (ones(1,k) - abs(C-center)./factor)'*(ones(1,k) - abs(C-center)./(factor));
end
end

function net = addConvT(net, kernel_size, input_dim, output_dim, up, crop, learnW)
    f = bilinear_u(kernel_size, output_dim, input_dim);
    net.layers{end+1} = struct('type','convt','name','deconv',...
                        'weights',{{single(f),[]}}, 'upsample',up,...
                        'numGroups', output_dim,...
                        'crop',crop, 'learningRate',[1,1]*learnW);
end




