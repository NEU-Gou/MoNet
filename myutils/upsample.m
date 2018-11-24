function seg_orig = upsample(segs, im_dims, coords, im)

  if 0
    % densify by colorization
    solver = 2;    
    gI = repmat(rgb2gray(double(gather(im+100))/255),[1 1 3]); % this is the gray image we use to propagate
    cI = gI(:,:,1); cI(sub2ind(floor(im_dims),coords(1,:),coords(2,:)))= segs/max(segs(:));
    cI = repmat(cI,[1 1 3]);
    
    colorIm=(sum(abs(gI-cI),3)>0.01);
    colorIm=double(colorIm);
    sgI=rgb2ntsc(gI);
    scI=(cI);
    ntscIm(:,:,1)=sgI(:,:,1);
    ntscIm(:,:,2)=scI(:,:,2);
    ntscIm(:,:,3)=scI(:,:,3);

    max_d=floor(log(min(size(ntscIm,1),size(ntscIm,2)))/log(2)-2);
    iu=floor(size(ntscIm,1)/(2^(max_d-1)))*(2^(max_d-1));
    ju=floor(size(ntscIm,2)/(2^(max_d-1)))*(2^(max_d-1));
    id=1; jd=1;
    colorIm=colorIm(id:iu,jd:ju,:);
    ntscIm=ntscIm(id:iu,jd:ju,:);

    if (solver==1)
      nI=getVolColor(colorIm,ntscIm,[],[],[],[],5,1);
      nI=ntsc2rgb(nI);c
    else
      [nI,snI]=getColorExact(colorIm,ntscIm);
    end
    seg_orig = zeros(im_dims);
    a = snI(:,:,2); [c,i]=myfwkmeans(a(:),max(segs(:))); c = reshape(c,size(a));
    seg_orig(1:size(snI,1),1:size(snI,2)) = c;
  else
    % densify by pasting
    seg_orig = zeros(im_dims,'single');
    up = [2 4 8 16];
    [~,ii] = min(abs(size(segs,1)*up/im_dims(1)-1));
    upsz = up(ii)/2;
    for k = -upsz:upsz, % this is not quite right but it easier like this for alexnet when the spacing is slightly larger than 16
      for l = -upsz:upsz, 
        seg_orig(sub2ind(im_dims,max(1,coords(1,:)+k),max(1,coords(2,:)+l))) = segs; 
      end; 
    end
    % paste in borders
    L = max(1,coords(1,1)-upsz);
    seg_orig(1:L,:) = repmat(seg_orig(L,:),L,1); % top
    L = max(1,coords(1,end)+upsz);
    seg_orig(L+1:end,:) = repmat(seg_orig(L,:),im_dims(1)-L,1); % bottom
    L = max(1,coords(2,1)-upsz);
    seg_orig(:,1:L) = repmat(seg_orig(:,L),1,L); % left
    L = max(1,coords(2,end)+upsz);
    seg_orig(:,L+1:end) = repmat(seg_orig(:,L),1,im_dims(2)-L); % right
  end
end