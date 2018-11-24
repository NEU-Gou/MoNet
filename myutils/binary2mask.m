function masks = binary2mask(bin_masks,type)
  masks = zeros(size(bin_masks,1),size(bin_masks,2),type);
  umasks = 1:size(bin_masks,3);
  for i = 1: length(umasks)
    masks(bin_masks(:,:,i)>0) = i;
  end
end