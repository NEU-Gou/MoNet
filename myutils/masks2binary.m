function bin_masks = masks2binary(masks,type)
  umasks = unique(masks);

  bin_masks = zeros(size(masks,1),size(masks,2),length(umasks),type);
  for i = 1: length(umasks)
    bin_masks(:,:,i) = masks==umasks(i);
  end
end