function code = get_rcnn_features(net, im, varargin)
% GET_RCNN_FEATURES
%    This function gets the fc7 features for an image region,
%    extracted from the provided mask.

opts.batchSize = 96 ;
opts.regionBorder = 0.05;
opts = vl_argparse(opts, varargin) ;

if ~iscell(im)
  im = {im} ;
end

res = [] ;
cache = struct() ;
resetCache() ;

    % for each image
    function resetCache()
        cache.images = cell(1,opts.batchSize) ;
        cache.indexes = zeros(1, opts.batchSize) ;
        cache.numCached = 0 ;
    end

    function flushCache()
        if cache.numCached == 0, return ; end
        images = cat(4, cache.images{:}) ;
        images = bsxfun(@minus, images, net.meta.normalization.averageImage) ;
        if net.useGpu
            images = gpuArray(images) ;
        end
        res = vl_simplenn(net, images, ...
                        [], res, ...
                        'conserveMemory', true, ...
                        'sync', true) ;
        code_ = squeeze(gather(res(end).x)) ;
        code_ = bsxfun(@times, 1./sqrt(sum(code_.^2)), code_) ;
        for q=1:cache.numCached
            code{cache.indexes(q)} = code_(:,q) ;
        end
        resetCache() ;
    end

    function appendCache(i,im)
        cache.numCached = cache.numCached + 1 ;
        cache.images{cache.numCached} = im ;
        cache.indexes(cache.numCached) = i;
        if cache.numCached >= opts.batchSize
            flushCache() ;
        end
    end

    code = {} ;
    for k=1:numel(im)
        appendCache(k, getImage(opts, single(im{k}), net.meta.normalization.imageSize(1), net.meta.normalization.keepAspect));
    end
    flushCache() ;
end

% -------------------------------------------------------------------------
function reg = getImage(opts, im, regionSize, keepAspect)
% -------------------------------------------------------------------------
 
    if keepAspect
        w = size(im,2) ;
        h = size(im,1) ;
        factor = [regionSize/h,regionSize/w];
        
        
        factor = max(factor);
        %if any(abs(factor - 1) > 0.0001)
        
        im_resized = imresize(im, ...
            'scale', factor, ...
            'method', 'bicubic') ;
        %end
        
        w = size(im_resized,2) ;
        h = size(im_resized,1) ;
        
        reg = imcrop(im_resized, [fix((w-regionSize)/2)+1, fix((h-regionSize)/2)+1,...
            round(regionSize)-1, round(regionSize)-1]);
    else
        reg = imresize(im, [regionSize, regionSize], 'bicubic') ;
    end
end
