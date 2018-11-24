function res = vl_GlbGaussian_nn(net, x, dzdy, res, varargin)

% This code is used to compute forward and backward propagations
% for Global Gaussian Distribution Embeding Network. This file is 
% modified from code of BCNN (https://bitbucket.org/tsungyu/bcnn).
  
% Qilong Wang, Peihua Li, Lei Zhang. G^2DeNet: Global Gaussian 
% Distribution Embeding Network and Its Application to Visual
% Recognition. In CVPR, 2017.

% Copyright (C) 2017 Qilong Wang, Peihua Li, Lei Zhang.
% All rights reserved.

opts.res = [] ;
opts.conserveMemory = false ;
opts.sync = false ;
opts.mode = 'normal' ;
opts.accumulate = false ;
opts.cudnn = true ;
opts.backPropDepth = +inf ;
opts.gradClip = false;
% opts.doforward = true;
opts = vl_argparse(opts, varargin);

ropts.alpha = 0.5;  %%alpha = 0.3~0.6 is ok!
ropts.epsilon = 1e-3;

n = numel(net.layers) ;


if (nargin <= 2) || isempty(dzdy)
  doder = false ;
else
  doder = true ;
end


if opts.cudnn
  cudnn = {'CuDNN'} ;
else
  cudnn = {'NoCuDNN'} ;
end


switch lower(opts.mode)
  case 'normal'
    testMode = false ;
  case 'test'
    testMode = true ;
  otherwise
    error('Unknown mode ''%s''.', opts. mode) ;
end

docrosslayer = false;
for i=1:n
  l = net.layers{i} ;
  if(strcmp(l.type, 'bilinearclpool'))
      docrosslayer = true;
      crlayer1 = l.layer1;
      crlayer2 = l.layer2;
  end
end


gpuMode = isa(x, 'gpuArray') ;

if nargin <= 3 || isempty(res)
  res = struct(...
    'x', cell(1,n+1), ...
    'dzdx', cell(1,n+1), ...
    'dzdw', cell(1,n+1), ...
    'aux', cell(1,n+1), ...
    'time', num2cell(zeros(1,n+1)), ...
    'backwardTime', num2cell(zeros(1,n+1))) ;
end
res(1).x = x ;


for i=1:n
  l = net.layers{i} ;
  res(i).time = tic ;
%   % debug -- Gou
%   fprintf('layer--%s, aver--%.5f\n',l.name,(mean(mean(res(i).x(:,:,1,1)))));
  
  switch l.type
    case 'conv'
      res(i+1).x = vl_nnconv(res(i).x, l.weights{1}, l.weights{2}, ...
        'pad', l.pad, ...
        'stride', l.stride, ...
        l.opts{:}, ...
        cudnn{:}) ;
    case 'convt'
      res(i+1).x = vl_nnconvt(res(i).x, l.weights{1}, l.weights{2}, ...
        'crop', l.crop, ...
        'upsample', l.upsample, ...
        'numGroups', l.numGroups, ...
        l.opts{:}, ...
        cudnn{:}) ;
    case 'pool'
      res(i+1).x = vl_nnpool(res(i).x, l.pool, ...
        'pad', l.pad, 'stride', l.stride, ...
        'method', l.method, ...
        l.opts{:}, ...
        cudnn{:}) ;
    case {'normalize', 'lrn'}
      res(i+1).x = vl_nnnormalize(res(i).x, l.param) ;
    case 'softmax'
      res(i+1).x = vl_nnsoftmax(res(i).x) ;
    case 'loss'
      res(i+1).x = vl_nnloss(res(i).x, l.class) ;
    case 'softmaxloss'
      res(i+1).x = vl_nnsoftmaxloss(res(i).x, l.class) ;
    case 'relu'
      if l.leak > 0, leak = {'leak', l.leak} ; else leak = {} ; end
      res(i+1).x = vl_nnrelu(res(i).x,[],leak{:}) ;
    case 'sigmoid'
      res(i+1).x = vl_nnsigmoid(res(i).x) ;
    case 'noffset'
      res(i+1).x = vl_nnnoffset(res(i).x, l.param) ;
    case 'spnorm'
      res(i+1).x = vl_nnspnorm(res(i).x, l.param) ;
    case 'dropout'
      if testMode
        res(i+1).x = res(i).x ;
      else
        [res(i+1).x, res(i+1).aux] = vl_nndropout(res(i).x, 'rate', l.rate) ;
      end
    case 'bnorm'
      if testMode
        res(i+1).x = vl_nnbnorm(res(i).x, l.weights{1}, l.weights{2}, 'moments', l.weights{3}) ;
      else
        res(i+1).x = vl_nnbnorm(res(i).x, l.weights{1}, l.weights{2}) ;
      end
    case 'pdist'
      res(i+1) = vl_nnpdist(res(i).x, l.p, 'noRoot', l.noRoot, 'epsilon', l.epsilon) ;
    case 'GlbGaussian'
        res(i+1) = nnfspool_forward_Gaussian(ropts, res(i), res(i+1)) ;
        res(i+1).cpu = 1;  
    case 'sqrt'
      res(i+1).x = vl_nnsqrt(res(i).x, 1e-8);
    case 'l2norm'
      res(i+1).x = vl_nnl2norm(res(i).x, 1e-10);
    case 'fsqrt'
      res(i+1).x = vl_nnfsqrt(res(i).x,1e-6);   
    case 'custom'
      res(i+1) = l.forward(l, res(i), res(i+1)) ;
    otherwise
      error('Unknown layer type %s', l.type) ;
  end
  % optionally forget intermediate results
  forget = opts.conserveMemory ;
  forget = forget & (~doder || strcmp(l.type, 'relu')) ;
  forget = forget & ~(strcmp(l.type, 'loss') || strcmp(l.type, 'softmaxloss')) ;
  forget = forget & (~isfield(l, 'rememberOutput') || ~l.rememberOutput) ;
  forget = forget & (~docrosslayer || (i~=(crlayer1+1) && i~=(crlayer2+1)));
  if forget
    res(i).x = [] ;
  end
  if gpuMode & opts.sync
    % This should make things slower, but on MATLAB 2014a it is necessary
    % for any decent performance.
    wait(gpuDevice) ;
  end
  res(i).time = toc(res(i).time) ;
end

if doder
  res(n+1).dzdx = dzdy ;
  for i=n:-1:max(1, n-opts.backPropDepth+1)
    l = net.layers{i} ;
    res(i).backwardTime = tic ;
    switch l.type
        case 'conv'
            [backprop, dzdw{1}, dzdw{2}] = ...
                vl_nnconv(res(i).x, l.weights{1}, l.weights{2}, ...
                res(i+1).dzdx, ...
                'pad', l.pad, 'stride', l.stride, ...
                l.opts{:}, cudnn{:}) ;
            res(i).dzdx = updateGradient(res(i).dzdx, backprop);
            clear backprop

      case 'convt'
          [backprop, dzdw{1}, dzdw{2}] = ...
              vl_nnconvt(res(i).x, l.weights{1}, l.weights{2}, ...
              res(i+1).dzdx, ...
              'crop', l.crop, 'upsample', l.upsample, ...
              'numGroups', l.numGroups, l.opts{:}, cudnn{:}) ;
            res(i).dzdx = updateGradient(res(i).dzdx, backprop);
            clear backprop


      case 'pool'
        backprop = vl_nnpool(res(i).x, l.pool, res(i+1).dzdx, ...
                                'pad', l.pad, 'stride', l.stride, ...
                                'method', l.method, ...
                                l.opts{:}, cudnn{:}) ;
        res(i).dzdx = updateGradient(res(i).dzdx, backprop);
        clear backprop
      case {'normalize', 'lrn'}
        backprop = vl_nnnormalize(res(i).x, l.param, res(i+1).dzdx) ;
        res(i).dzdx = updateGradient(res(i).dzdx, backprop);
        clear backprop
      case 'softmax'
        backprop = vl_nnsoftmax(res(i).x, res(i+1).dzdx) ;
        res(i).dzdx = updateGradient(res(i).dzdx, backprop);
        clear backprop
      case 'loss'
        backprop = vl_nnloss(res(i).x, l.class, res(i+1).dzdx) ;
        res(i).dzdx = updateGradient(res(i).dzdx, backprop);
        clear backprop
      case 'softmaxloss'
        backprop = vl_nnsoftmaxloss(res(i).x, l.class, res(i+1).dzdx) ;
        res(i).dzdx = updateGradient(res(i).dzdx, backprop);
        clear backprop
      case 'relu'
        if l.leak > 0, leak = {'leak', l.leak} ; else leak = {} ; end
        if ~isempty(res(i).x)
          backprop = vl_nnrelu(res(i).x, res(i+1).dzdx, leak{:}) ;
        else
          % if res(i).x is empty, it has been optimized away, so we use this
          % hack (which works only for ReLU):
          backprop = vl_nnrelu(res(i+1).x, res(i+1).dzdx, leak{:}) ;
        end
        res(i).dzdx = updateGradient(res(i).dzdx, backprop);
        clear backprop
      case 'sigmoid'
        backprop = vl_nnsigmoid(res(i).x, res(i+1).dzdx) ;
        res(i).dzdx = updateGradient(res(i).dzdx, backprop);
        clear backprop
      case 'noffset'
        backprop = vl_nnnoffset(res(i).x, l.param, res(i+1).dzdx) ;
        res(i).dzdx = updateGradient(res(i).dzdx, backprop);
        clear backprop
      case 'spnorm'
        backprop = vl_nnspnorm(res(i).x, l.param, res(i+1).dzdx) ;
        res(i).dzdx = updateGradient(res(i).dzdx, backprop);
        clear backprop
      case 'dropout'
        if testMode
          backprop = res(i+1).dzdx ;
        else
          backprop = vl_nndropout(res(i).x, res(i+1).dzdx, ...
                                     'mask', res(i+1).aux) ;
        end
        res(i).dzdx = updateGradient(res(i).dzdx, backprop);
        clear backprop
      case 'bnorm'
          [backprop, dzdw{1}, dzdw{2}, dzdw{3}] = ...
              vl_nnbnorm(res(i).x, l.weights{1}, l.weights{2}, ...
              res(i+1).dzdx) ;
          res(i).dzdx = updateGradient(res(i).dzdx, backprop);
          dzdw{3} = dzdw{3} * size(res(i).x,4) ;
          clear backprop
      case 'pdist'
        backprop = vl_nnpdist(res(i).x, l.p, res(i+1).dzdx, ...
                                 'noRoot', l.noRoot, 'epsilon', l.epsilon) ;
        res(i).dzdx = updateGradient(res(i).dzdx, backprop);
        clear backprop
      case 'GlbGaussian'
        backprop = nnfspool_backward_Gaussian(ropts, res(i), res(i+1)) ;    
        res(i).dzdx = updateGradient(res(i).dzdx, backprop);
      case 'sqrt'
        backprop = vl_nnsqrt(res(i).x, 1e-8, res(i+1).dzdx);
        res(i).dzdx = updateGradient(res(i).dzdx, backprop);
        clear backprop
      case 'l2norm'
        backprop = vl_nnl2norm(res(i).x, 1e-10, res(i+1).dzdx);
        res(i).dzdx = updateGradient(res(i).dzdx, backprop);
        clear backprop
      case 'fsqrt'
        backprop = vl_nnfsqrt(res(i).x, 1e-6, res(i+1).dzdx);
        res(i).dzdx = updateGradient(res(i).dzdx, backprop);
        clear backprop  
      case 'custom'
        res(i) = l.backward(l, res(i), res(i+1)) ;
    end
    
    switch l.type
      case {'conv', 'convt', 'bnorm'}
        if ~opts.accumulate
          res(i).dzdw = dzdw ;
        else
          for j=1:numel(dzdw)
            res(i).dzdw{j} = res(i).dzdw{j} + dzdw{j} ;
          end
        end
        dzdw = [] ;
    end
   
    % gradient clip
    if opts.gradClip
    	res(i).dzdx(res(i).dzdx>1) = 1;
    	res(i).dzdx(res(i).dzdx<-1) = -1;
    end

    if opts.conserveMemory
      res(i+1).dzdx = [] ;
    end
    if gpuMode & opts.sync
      wait(gpuDevice) ;
    end
    res(i).backwardTime = toc(res(i).backwardTime) ;
  end
  if opts.conserveMemory
      res(1).dzdx = [] ;
  end
end


% add up the gradient 
function g = updateGradient(y, backprop)

if isempty(y)
    g = backprop;
else
    g = y + backprop;
end
