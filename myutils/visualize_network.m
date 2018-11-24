function visualize_network(vistype,net,im,masks,mean_image)
  switch vistype
    case 'grad'
%       res(1).x = gpuArray(im);
      res = vl_mysimplenn(net, gpuArray(im), 1, [], masks);
      figure; subplot(121); imshow(im); subplot(122); 
      %imagesc(max(abs(res(1).dzdx),[],3)); axis equal; axis off;
      maxim = max(abs(res(1).dzdw{1}(:)));
      imshow(max(abs(res(1).dzdw{1}),[],3)/maxim);
      mm = vl_nnconvt(permute(res(2).dzdx,[1 2 4 3]),permute(net.layers{1}.filters,[1 2 4 3]),shiftdim(net.layers{1}.biases,-1));
      
    case 'dream'
      addpath('external_src/minFunc');
%       im = bsxfun(@minus,gpuArray.rand(size(im),'single')*255,mean_image);
      im = gpuArray(im)+50*gpuArray.randn(size(im),'single');
      opts.Method = 'sd';
%       opts.Corr = 100000;
%       opts.LS_init = 1;
%       opts.LS_init = 1;
%       opts.DerivativeCheck = 1;
        newmasks = masks;
%       ind = 7;
%       newmasks{1} = masks{1}(:,:,ind);
%       newmasks{1} = cat(3,~any(newmasks{1},3),newmasks{1});
      [newim] = minFunc(@(x)fwrapper(x,net,newmasks,size(im)),im(:),opts);
      figure; subplot(121); imshow(bsxfun(@plus,im,mean_image)/255); subplot(122); imshow(bsxfun(@plus,reshape(newim,size(im)),mean_image)/255);
      figure; imagesc(max(abs(bsxfun(@plus,im,mean_image)/255-bsxfun(@plus,reshape(newim,size(im)),mean_image)/255),[],3));
      pause; 
      close all;
  end
  
end

function [y,grad] = fwrapper(x,net,masks,imsize)
  im = reshape(x,imsize);
  res = vl_mysimplenn(net,im,1,[],masks);
  y = gather(res(end).x);
  grad = gather(res(1).dzdx(:));
end

