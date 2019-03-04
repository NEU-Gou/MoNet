% script_run_exp.m
% This is main code to train and test MoNet and its variations
% For more details, please refer to the following paper:
% Mengran Gou, Fei Xiong, Octavia Camps and Mario Sznaier, MoNet: Moment
% Embedding Network. CVPR 2018
% 
% Part of the code is modified from BCNN 
% (https://bitbucket.org/tsungyu/bcnn)
% and G^2DeNet
% (http://peihuali.org/publications/G2DeNet/G2DeNet-FGVC-v1.0.zip)
%
% Write by Mengran Gou @ 07/2018

diary('command_log.txt');
diary on

if(~exist('data', 'dir'))
    mkdir('data');
end

datafolder = './data/';

monet.name = 'monet' ;
monet.opts = {...
    'type', 'monet', ...
    'modela', [datafolder, 'models/imagenet-matconvnet-vgg-verydeep-16.mat'], ... %imagenet-vgg-verydeep-16.mat
    'layera', 30,...
    'modelb', [datafolder, 'models/imagenet-matconvnet-vgg-verydeep-16.mat'], ...
    'layerb', 30,...
    'shareWeight', true,...
    };

addpath 'myutils/';
addpath 'main/';
addpath 'base/';
addpath(genpath('layers/'));

setupNameList = {'monet'};
encoderList = {{monet}};
datasetList = {{'cub', 1}};  %{{'cub', 1}}, {{'cars', 1}}; {'aircraft-variant', 1}
networkList = {'MoNet'};

for ii = 1 : numel(datasetList)
    dataset = datasetList{ii} ;
    if iscell(dataset)
        numSplits = dataset{2} ;
        dataset = dataset{1} ;
    else
        numSplits = 1 ;
    end
    for jj = 1 : numSplits
        for ee = 1: numel(encoderList)
            for nn = 1:numel(networkList)
                
                [opts, imdb] = model_setup('dataset', dataset, ...
                    'datafolder', datafolder,...
                    'encoders', encoderList{ee}, ...
                    'network', networkList{nn}, ...
                    'prefix', networkList{nn}, ...  % output folder name
                    'projDim', 1e4, .... % Compact pooling projection dim
                    'batchSize', 16, ... %% 64_old
                    'imgScale', 2, ...       % specify the scale of input images
                    'bcnnLRinit', true, ...   % do logistic regression to initilize softmax layer
                    'dataAugmentation', {'f2','none','none'},...      % do data augmentation [train, val, test]. Only support flipping ('f2') for train set on current release.
                    'useGpu', [], ...          %specify the GPU to use. empty for using CPU
                    'learningRate', 0.001, ... %%0.001
                    'numEpochs', 100, ...
                    'momentum', 0.9, ... %0.9
                    'keepAspect', true, ...
                    'printDatasetInfo', true, ...
                    'fromScratch', false, ...
                    'rgbJitter', false, ...
                    'useVal', false,...
		    'gradClip', true,...
                    'numSubBatches', 2);
                if strcmp(dataset,'aircraft-variant')==1
                    imdb.images.set(imdb.images.set==2) = 1;
                    imdb.images.set(imdb.images.set==3) = 2;
                else
                    imdb.images.set(imdb.images.set==3) = 2;
                end
                imdb_g2denet_train_dag(imdb, opts);
            end
        end
    end
end





