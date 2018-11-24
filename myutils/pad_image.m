function imfin = pad_image(im, H, W)
% pad image
[Him, Wim, Dim] = size(im);

Wfull = 2*W + Wim;

for j = 1: Dim  
  lim(:,:,j) = [zeros(Him,W) im(:,:,j) zeros(Him,W)]; % FIXME do the color scaling thing
  imfin(:,:,j) = [zeros(H,Wfull); lim(:,:,j); zeros(H,Wfull)]; 
end
end