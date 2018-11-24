function [im, r, pad] = center_image(images, params, interp)
% take image and rescale and fit to a predefined size defined in pars under
% maxsize. The output is guaranteed to have that size
  if ~exist('interp','var')
    interp = 'bicubic';
  end
  
  if ~isstruct(params)
    % FIXME workaround for the API
    pars.maxsize = params;
  else
    pars = params;
  end
  
  [H, W, d] = size(images);
  r = min(pars.maxsize(1)/H,pars.maxsize(2)/W);
  imres = imresize(images,r, interp);
  [H, W, d] = size(imres);
  
  if pars.maxsize(1)~=H || pars.maxsize(2)~=W
    if size(imres,1) > pars.maxsize(1)
      imres(pars.maxsize(1)+1:end,:,:)= [];
    elseif size(imres,2) > pars.maxsize(2)
      imres(:,pars.maxsize(2)+1:end,:)= [];
    end
    [H, W, d] = size(imres);
  end
  
  pad = [0 0];
  im = zeros(pars.maxsize(1),pars.maxsize(2),d);
  for j = 1:d
    if H==pars.maxsize(1)
      pad(1) = floor((pars.maxsize(2)-W)/2);
      if pad(1)>=0
        im(:,:,j) = [zeros(pars.maxsize(1),pad(1)) imres(:,:,j) zeros(pars.maxsize(1),pars.maxsize(2)- pad(1) -size(imres,2))];
      else
        im(:,:,j) = imres(:,1:end-1,j);
      end
    elseif pars.maxsize(2)==W
      pad(2) = floor((pars.maxsize(1)-H)/2);
      if pad(2)>=0
        im(:,:,j) = [zeros(pad(2),pars.maxsize(2)); imres(:,:,j); zeros(pars.maxsize(1)- pad(2) -size(imres,1),pars.maxsize(2))];
      else
        im(:,:,j) = imres(1:end-1,:,j);
      end
    end
  end
  
  [H, W, d] = size(im);

  assert(~any(pars.maxsize-[H W]))