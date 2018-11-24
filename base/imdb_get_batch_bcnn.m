function imo = imdb_get_batch_bcnn(images, varargin)
% imdb_get_batch_bcnn  Load, preprocess, and pack images for BCNN evaluation
% For asymmetric model, the function preprocesses the images twice for two networks
% separately.

% OUTPUT
% imo: a cell array where each element is a cell array of images.
%       For symmetric bcnn model, numel(imo) will be 1 and imo{1} will be a
%       cell array of images
%       For asymmetric bcnn, numel(imo) will be 2. imo{1} is a cell array containing the preprocessed images for network A 
%       Similarly, imo{2} is a cell array containing the preprocessed images for network B

%
% Copyright (C) 2015 Tsung-Yu Lin, Aruni RoyChowdhury, Subhransu Maji.
% All rights reserved.
%
% This file is part of the BCNN and is made available under
% the terms of the BSD license (see the COPYING file).
%
% This file modified from CNN_IMAGENET_GET_BATCH of MatConvNet


for i=1:numel(varargin{1})
    opts(i).imageSize = [227, 227] ;
    opts(i).border = [0, 0] ;
    opts(i).keepAspect = true;
    opts(i).numAugments = 1 ;
    opts(i).transformation = 'none' ;
    opts(i).averageImage = [] ;
    opts(i).rgbVariance = zeros(0,3,'single') ;
    opts(i).interpolation = 'bilinear' ;
    opts(i).numThreads = 1 ;
    opts(i).prefetch = false;
    opts(i).scale = 1;
    opts(i) = vl_argparse(opts(i), {varargin{1}(i),varargin{2:end}});
    
    if(i==1)
        
        switch opts(i).transformation
            case 'none'
                tfs = [.5 ; .5 ; 0 ];
            case 'f2'
                tfs = [...
                    0.5 0.5 ;
                    0.5 0.5 ;
                    0   1];
            case 'f5'
                tfs = [...
                    .5 0 0 1 1 .5 0 0 1 1 ;
                    .5 0 1 0 1 .5 0 1 0 1 ;
                    0 0 0 0 0  1 1 1 1 1] ;
            case 'f25'
                [tx,ty] = meshgrid(linspace(0,1,5)) ;
                tfs = [tx(:)' ; ty(:)' ; zeros(1,numel(tx))] ;
                tfs_ = tfs ;
                tfs_(3,:) = 1 ;
                tfs = [tfs,tfs_] ;
            case 'stretch'
            otherwise
                error('Uknown transformations %s', opts.transformation) ;
        end
		
		[~,transformations] = sort(rand(size(tfs,2), numel(images)), 1) ;
        
        
        if ~isempty(opts(i).rgbVariance) && isempty(opts(i).averageImage)
            opts(i).averageImage = zeros(1,1,3) ;
        end
        if numel(opts(i).averageImage) == 3
            opts(i).averageImage = reshape(opts(i).averageImage, 1,1,3) ;
        end
        if ~isempty(opts(i).rgbVariance)
            rgbjitt = opts(i).rgbVariance * randn(3,numel(images));
        else
            rgbjitt = [];
        end
        
        
        
        % fetch is true if images is a list of filenames (instead of
        % a cell array of images)
        % fetch = numel(images) > 1 && ischar(images{1}) ;
        fetch = ischar(images{1}) ;
        
        % prefetch is used to load images in a separate thread
        prefetch = fetch & opts(i).prefetch ;
        
        
        if prefetch
            vl_imreadjpeg(images, 'numThreads', opts(i).numThreads, 'prefetch') ;
            imo = [] ;
            return ;
        end
        if fetch
            im = vl_imreadjpeg(images,'numThreads', opts(i).numThreads) ;
        else
            im = images ;
        end

    end
    
    % preprocess images for the ith network
    imo{i} = get_batch_fun(images, im,  opts(i), transformations, tfs, rgbjitt);
    
end




function imo = get_batch_fun(images, im, opts, transformations, tfs, rgbjitt)

opts.imageSize(1:2) = round(opts.imageSize(1:2).*opts.scale);
opts.border = round(opts.border.*opts.scale);
if(opts.scale ~= 1)
    opts.averageImage = mean(mean(opts.averageImage, 1),2);
end


imo = zeros(opts.imageSize(1), opts.imageSize(2), 3, ...
            numel(images)*opts.numAugments, 'single') ;


si=1;
for i=1:numel(images)

  trf = transformations(:,i);
    
  % acquire image
  if isempty(im{i})
    imt = imread(images{i}) ;
    imt = single(imt) ; % faster than im2single (and multiplies by 255)
  else
    imt = im{i} ;
  end
  if size(imt,3) == 1
      imt = cat(3, imt, imt, imt) ;
  end
  
  w = size(imt,2) ;
  h = size(imt,1) ;
  factor = [(opts.imageSize(1)+opts.border(1))/h ...
      (opts.imageSize(2)+opts.border(2))/w];
  
  % resize
  if opts.keepAspect
      factor = max(factor) ;
  end
  if any(abs(factor - 1) > 0.0001)
      
      imt = imresize(imt, ...
          'scale', factor, ...
          'method', opts.interpolation) ;
  end
  % crop & flip
  w = size(imt,2) ;
  h = size(imt,1) ;
  for ai = 1:opts.numAugments
    switch opts.transformation
      case 'stretch'
        sz = round(min(opts.imageSize(1:2)' .* (1-0.1+0.2*rand(2,1)), [w;h])) ;
        dx = randi(w - sz(2) + 1, 1) ;
        dy = randi(h - sz(1) + 1, 1) ;
        flip = rand > 0.5 ;
      otherwise
        tf = tfs(:, trf(mod(ai-1, numel(trf)) + 1)) ;
        sz = opts.imageSize(1:2) ;
        dx = floor((w - sz(2)) * tf(2)) + 1 ;
        dy = floor((h - sz(1)) * tf(1)) + 1 ;
        flip = tf(3) ;
    end
    sx = round(linspace(dx, sz(2)+dx-1, opts.imageSize(2))) ;
    sy = round(linspace(dy, sz(1)+dy-1, opts.imageSize(1))) ;
    if flip, sx = fliplr(sx) ; end

    if ~isempty(opts.averageImage)
      offset = opts.averageImage ;
      if ~isempty(opts.rgbVariance)
        offset = bsxfun(@plus, offset, reshape(rgbjitt(:,i), 1,1,3)) ;
      end
      imo(:,:,:,si) = bsxfun(@minus, imt(sy,sx,:), offset) ;
    else
      imo(:,:,:,si) = imt(sy,sx,:) ;
    end
    si = si + 1 ;
  end
    
end
