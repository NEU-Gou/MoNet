function Quality = sp_quality(superpixels,labels,masks)
  debug = 0;
  i = 1;
  tic;
  
  un_sp = unique(superpixels);
  n_sp = length(un_sp);
  n_masks = length(labels);
%   n_pixels = numel(superpixels);
  n_gt_masks = size(masks,3);
  
  sp_pixels = cell(numel(un_sp),1);
  for k=1:numel(un_sp)
    sp_pixels{k} = find(superpixels==un_sp(k));
  end
  
  sp_app = false(n_sp,n_masks);
  for m = 1: n_masks
    sp_app(labels{m},m) = true;
  end
  
  sp_size = zeros(numel(un_sp),1);
  for k=1:numel(un_sp)
    sp_size(k) = sum(superpixels(:)==un_sp(k));
  end
  
  Quality{1} = [];
  for j=1:n_gt_masks
    mask = single(masks(:,:,j));
    mask_area = sum(mask(:));
    
    Quality{1}(j).q = zeros(numel(labels),1,'single');
    Quality{1}(j).bbox = [];
    Quality{1}(j).area = mask_area;

    inters = zeros(numel(un_sp),1);

    for k=1:numel(un_sp)                    
      inters(k) = sum(mask(sp_pixels{k}));
    end

    for k=1:numel(labels)
      Quality{1}(j).q(k) = sum(inters(sp_app(:,k)))/(mask_area+sum(sp_size(sp_app(:,k)))-sum(inters(sp_app(:,k))));
    end
    
    if debug
      % checking the results
      dims = size(superpixels);
      seg_masks = false(dims(1),dims(2),length(labels));
      for l = 1: size(seg_masks,3)
        tmpmask = false(dims(1),dims(2));
        tmpmask(cell2mat(sp_pixels(labels{l}))) = true;
        seg_masks(:,:,l) = tmpmask;
      end
      [~,~,oo] = myCalcCandScoreFigureGroundAll(seg_masks,mask,'overlap',true(size(mask)));
      figure; plot(oo-Quality{1}(j).q,'r')
      if debug>1
        [~,ind] = sort(oo,'descend'); 
        im = imread(fullfile(dataDir,'JPEGImages',[img_names{i} '.jpg']));
        figure; subplot_auto_transparent(seg_masks(:,:,ind(1:10)),im);
      end
      pause
    end
    
    assert(all(1 >= Quality{1}(j).q ));
  end
