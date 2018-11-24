function imfin = pad_image_to_size(im, H, W)
% pad image
[Him, Wim, Dim] = size(im);

for j = 1: Dim  
  lim(:,:,j) = [zeros(Him,ceil(.5*(W-Wim))) im(:,:,j) zeros(Him,W-Wim-ceil(.5*(W-Wim)))]; % FIXME do the color scaling thing
  imfin(:,:,j) = [zeros(ceil(.5*(H-Him)),W); lim(:,:,j); zeros(H-Him-ceil(.5*(H-Him)),W)]; 
end
end