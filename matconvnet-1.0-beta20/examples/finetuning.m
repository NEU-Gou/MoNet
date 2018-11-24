clear

addpath('mycodes');
run(fullfile(fileparts(mfilename('fullpath')), '../matlab/vl_setupnn.m')) ;

opts.pretrainedPath = '../pretrained/imagenet-vgg-verydeep-16.mat';
opts.dataset = 'scene67';
opts.modelType = 'vgg-16' ;
opts.networkType = 'simplenn' ;
opts.scale = 1 ;
opts.initBias = 0.1 ;
opts.weightDecay = 1 ;
opts.batchNormalization = false ;
opts.weightInitMethod = 'gaussian' ;
opts.cudnnWorkspaceLimit = 1024*1024*8192 ; % 1GB
sfx = opts.modelType ;
if opts.batchNormalization, sfx = [sfx '-bnorm'] ; end
sfx = [sfx '-' opts.networkType] ;
opts.expDir = fullfile(vl_rootnn, 'data', [opts.dataset '-' sfx]) ;

opts.numFetchThreads = 4 ;
opts.lite = false ;
opts.imdbPath = fullfile(opts.expDir, 'imdb.mat');
opts.train = struct() ;
if ~isfield(opts.train, 'gpus'), opts.train.gpus = []; end;
opts.train.learningRate = 0.001*[ones(1, 3) .1*ones(1, 3) .01*ones(1, 2)];
% opts.train.learningRate = 0.001*[ones(1, 10)];
opts.numEpochs = numel(opts.train.learningRate) ;
if exist(opts.imdbPath)
    load(opts.imdbPath);
else
    imdb = setupDateSet(opts.dataset);
    mkdir(opts.expDir) ;
    save(opts.imdbPath,'imdb');
end

net = create_net_from_pretrained(opts,imdb);

[net, info] = cnn_train(net, imdb, getBatchFn(opts, net.meta), ...
                      'expDir', opts.expDir, ...
                      net.meta.trainOpts, ...
                      opts.train,'val', find(imdb.images.set == 3)) ;
                  
net = cnn_imagenet_deploy(net) ;
modelPath = fullfile(opts.expDir, 'net-deployed.mat');

switch opts.networkType
  case 'simplenn'
    save(modelPath, '-struct', 'net') ;
  case 'dagnn'
    net_ = net.saveobj() ;
    save(modelPath, '-struct', 'net_') ;
    clear net_ ;
end