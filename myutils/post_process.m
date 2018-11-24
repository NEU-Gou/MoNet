function seg_orig = post_process(seg_orig,varargin)
  for i = 1: length(varargin)
    switch varargin{i}
      case 'scc'
        un_l = unique(seg_orig);
        NumObjects = max(un_l);
        for l = 2: length(un_l)
          bwmask = seg_orig==un_l(l);
          comps = bwconncomp(bwmask);
          for j = 2: comps.NumObjects
            if length(comps.PixelIdxList{j})>4*16*16 % 4 pixels of 16x16
              NumObjects = NumObjects + 1;
              seg_orig(comps.PixelIdxList{j}) = NumObjects;
            else
              % assign to neighbor with most pixels
              bwmaskcomp = false(size(seg_orig));
              bwmaskcomp(comps.PixelIdxList{j}) = true;
              bwcomps = imdilate(bwmaskcomp,strel('square',5))&~bwmaskcomp;
              [~,hneighb] = max(hist(seg_orig(bwcomps),1:NumObjects));
              neighb = 1:NumObjects;
              nid = neighb(hneighb);
              seg_orig(comps.PixelIdxList{j}) = nid;
            end
          end
        end
      case 'none'
      otherwise 
        error('Unknown post processing !');
    end
  end
end