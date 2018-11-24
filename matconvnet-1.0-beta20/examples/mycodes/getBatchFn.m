% -------------------------------------------------------------------------
function fn = getBatchFn(opts, meta)
% -------------------------------------------------------------------------
useGpu = numel(opts.train.gpus) > 0 ;

bopts.numThreads = opts.numFetchThreads ;
bopts.imageSize = meta.normalization.imageSize ;
bopts.border = meta.normalization.border ;
bopts.averageImage = meta.normalization.averageImage ;
bopts.rgbVariance = meta.augmentation.rgbVariance ;
bopts.transformation = meta.augmentation.transformation ;

switch lower(opts.networkType)
  case 'simplenn'
    fn = @(x,y) getSimpleNNBatch(bopts,x,y) ;
  case 'dagnn'
    fn = @(x,y) getDagNNBatch(bopts,useGpu,x,y) ;
end

% -------------------------------------------------------------------------
function [im,labels] = getSimpleNNBatch(opts, imdb, batch)
% -------------------------------------------------------------------------
images = strcat([imdb.imageDir filesep], imdb.images.name(batch)) ;
isVal = ~isempty(batch) && imdb.images.set(batch(1)) ~= 1 ;

if ~isVal
  % training
%   im = cnn_imagenet_get_batch(images, opts, ...
%                               'prefetch', nargout == 0) ;
  im = cnn_imagenet_get_batch(images, opts, ...
                              'prefetch', nargout == 0, 'numAugments',2) ;
else
  % validation: disable data augmentation
  im = cnn_imagenet_get_batch(images, opts, ...
                              'prefetch', nargout == 0, ...
                              'transformation', 'none') ;
end

if nargout > 0
  labels = imdb.images.class(batch) ;
end

if ~isVal
    labels = repmat(labels,[2,1]);
    labels = labels(:)';
end

% -------------------------------------------------------------------------
function inputs = getDagNNBatch(opts, useGpu, imdb, batch)
% -------------------------------------------------------------------------
images = strcat([imdb.imageDir filesep], imdb.images.name(batch)) ;
isVal = ~isempty(batch) && imdb.images.set(batch(1)) ~= 1 ;

if ~isVal
  % training
  im = cnn_imagenet_get_batch(images, opts, ...
                              'prefetch', nargout == 0) ;
else
  % validation: disable data augmentation
  im = cnn_imagenet_get_batch(images, opts, ...
                              'prefetch', nargout == 0, ...
                              'transformation', 'none') ;
end

if nargout > 0
  if useGpu
    im = gpuArray(im) ;
  end
  labels = imdb.images.label(batch) ;
  inputs = {'input', im, 'label', labels} ;
end