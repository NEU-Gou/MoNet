function my_test_nnlayers(gpu,tests)
% my_test_nnlayers
% layer testing routine for numerical gradients
% 
% (c) 2015 -- catalin ionescu - catalin.ionescu@ins.uni-bonn.de

run(fullfile(fileparts(mfilename('fullpath')), '../matlab/vl_setupnn.m')) ;
addpath('cnn_o2p/');
addpath('myutils');
rng('default')

range = 1 ;

if nargin < 1, gpu = false ; end
if gpu
  grandn = @(varargin) (range * gpuArray.randn(varargin{:})) ;
  grand = @(varargin) (range * gpuArray.rand(varargin{:})) ;
else
  grandn = @(varargin)(range*randn(varargin{:}));
%   grandn = @(varargin) (range * randn(varargin{:})) ;
  grand = @(varargin)((2*range*rand(varargin{:})-1));
%   grand = @(varargin) (range * rand(varargin{:})) ;
end

if nargin < 2
  tests = 1;
end


for l = tests
  fprintf('======= test %0d start! =======================\n',l);
  % resets random number generator to obtain reproducible results
  if gpu, parallel.gpu.rng(0, 'combRecursive');
  else rng(0, 'combRecursive') ; end
  
  switch l
    case 1
      fprintf('vector function check\n');
      fprintf('=============================================\n');
      
      X = grandn(10,1);
      check_table = {'id    ',@(x)myfunc(x,'id');...
                     'log   ',@(x)myfunc(x,'log',1e-3);...
                     'tanh  ',@(x)myfunc(x,'tanh',0,2/3,1.7159)...
                     };
      
      for i = 1: size(check_table,1)
        fprintf('%s...\t\t',check_table{i,1});
        fun = check_table{i,2};
        [~,ndev] = autoGrad(X,0,fun);
        [~,adev] = fun(X);
        err = max(abs(ndev-adev));
        if err<tol(ndev,adev)
          fprintf('pass with %f at %f!\n',err,tol(ndev,adev));
        else
          fprintf('fail with %f at %f!\n',err,tol(ndev,adev));
        end
      end
      
    case 2 
      type = 'svd';
      fprintf('====> %s <=====\n',type);
      
      ind = 1:20;
      deltas = 2.^(-ind);
      
      N = 10;
      X = grand(N); dX = grand(N); W = grand(N);
      fwd_vars = cell(1,3); fwd_vars_dX = cell(1,3); ddev_vars = cell(1,2); bwd_vars = cell(1,1);

      %%%%%%% svd
      [fwd_vars{:}]   = matDecomp(type,'forward' ,X);
      [bwd_vars{:}]   = matDecomp(type,'backward',fwd_vars{:},W,W,W);
      [ddev_vars{:}]  = matDecomp(type,'directional',fwd_vars{:},dX);
      for d = 1: length(deltas)
        delta = deltas(d);
        
        [fwd_vars_dX{:}]      = matDecomp(type,'forward'      ,X+delta*dX);
        err_analytical(d,1)   = norm(ddev_vars{1}-(fwd_vars_dX{3}-fwd_vars{3})/delta,'fro'); % only the left eigenvectors
        err_analytical(d,2)   = norm(ddev_vars{2}-(fwd_vars_dX{2}-fwd_vars{2})/delta,'fro'); % eigenvalues
%         err1(d,3)             = norm(ddev_vars{3}-(fwd_vars_dX{3}-fwd_vars{3})/delta,'fro');
        err_numerical(d)      = abs(dp(W,(fwd_vars_dX{2}-fwd_vars{2})/delta)+...
                                    dp(W,(fwd_vars_dX{3}-fwd_vars{3})/delta)-...
                                    dp(bwd_vars{1},dX));
      end
      err                     = abs(dp(W,ddev_vars{2})+dp(W,ddev_vars{1})-dp(bwd_vars{1},dX));
      
      figure; plot(deltas,err_analytical); set(gca,'yscale','log','xscale','log'); legend('U','S');
      figure; plot(deltas,err_numerical);  set(gca,'yscale','log','xscale','log'); legend('dLdX');
      fprintf('Backward accuracy %f\n',err);

    case 3      
      type = 'eig';
      fprintf('====> %s <=====\n',type);

      ind = 1:20;
      deltas = 2.^(-ind);
      N = 10;
      
      X = randsym([N N],grand); dX = randsym([N N],grand); W = randsym([N N],grand);
      fwd_vars = cell(1,2); fwd_vars_dX = cell(1,2); ddev_vars = cell(1,2); bwd_vars = cell(1,1);
      
      %%%%%%% eig
      [fwd_vars{:}]     = matDecomp(type,'forward'    ,X);
      [bwd_vars{:}]     = matDecomp(type,'backward'   ,fwd_vars{:},W,W);
      [ddev_vars{:}]    = matDecomp(type,'directional',fwd_vars{:},dX);      
      for d = 1: length(deltas)
        delta = deltas(d);

        [fwd_vars_dX{:}]      = matDecomp(type,'forward' ,X+delta*dX);
        err_analytical(d,1)   = norm(ddev_vars{1}-(fwd_vars_dX{1}-fwd_vars{1})/delta,'fro');
        err_analytical(d,2)   = norm(ddev_vars{2}-(fwd_vars_dX{2}-fwd_vars{2})/delta,'fro');

        err_numerical(d)      = abs(dp(W,(fwd_vars_dX{2}-fwd_vars{2})/delta)+...
                                    dp(W,(fwd_vars_dX{1}-fwd_vars{1})/delta)-...
                                    dp(bwd_vars{1},dX));
      end
      err                     = abs(dp(W,ddev_vars{1})+dp(W,ddev_vars{2})-dp(bwd_vars{1},dX));
      
      figure; plot(deltas,err_analytical); set(gca,'yscale','log','xscale','log'); legend('U','S');
      figure; plot(deltas,err_numerical);  set(gca,'yscale','log','xscale','log'); legend('dLdX');
      fprintf('Backward accuracy %f\n',err);
      
    case 4
      fprintf('svdFun check\n');
      
      ind = 1:20;
      deltas = 2.^(-ind);
      
      % this is the data
      N = 10; % FIXME make it asymetric, multiple images to make sure things are kosher
      X = grand(N);
      dV = grand(N); 
      dS = dDiag(grand(N));     
      dLdC = grand(N);
      fwd_vars = cell(1,1); fwd_vars_dX = cell(1,1); bwd_vars = cell(1,2);
      
      type = {'id    ',@(x)myfunc(x,'id');...
             'log   ',@(x)myfunc(x,'log',1e-3);...
             'tanh  ',@(x)myfunc(x,'tanh',0,2/3,1.7159)...
             };
      
      [U,S,V] = svd(X);
      for i = 1: size(type,1)
        % forward test
        fprintf('====> %s <=====\n',type{i,1});

        % compute the function and derivatives at X
        [fwd_vars{:}]   = svdFun(type{i,2},'forward' ,U,S,V);
        [bwd_vars{:}]   = svdFun(type{i,2},'backward',U,S,V,dLdC);
        for d = 1: length(deltas)
          delta = deltas(d);
          
          [fwd_vars_dX{:}]        = svdFun(type{i,2},'forward' ,U,S,V+proj(V,delta*dV));
          err(i,d,1)              = abs((dp(dLdC,fwd_vars_dX{1})-dp(dLdC,fwd_vars{1}))/delta-dp(bwd_vars{1},dV));
          
          [fwd_vars_dX{:}]        = svdFun(type{i,2},'forward' ,U,S+delta*dS,V);
          err(i,d,2)              = abs((dp(dLdC,fwd_vars_dX{1})-dp(dLdC,fwd_vars{1}))/delta-dp(bwd_vars{2},dS));  
        end
        figure; plot(deltas,squeeze(err(i,:,:))); set(gca,'yscale','log','xscale','log'); title(['Convergence of ' type{i,1}]); legend('dV','d\Sigma');
      end
      
    case 5
      fprintf('eigFun check\n');
    
      ind = 1:20;
      deltas = 2.^(-ind);
      
      % this is the data
      N = 10;
      X = randsym([N,N],grand);
      dU = grand(N); 
      dS = dDiag(grand(N));     
      dLdC = grand(N);
      fwd_vars = cell(1,1); fwd_vars_dX = cell(1,1); bwd_vars = cell(1,2);
      type = {'id    ',@(x)myfunc(x,'id');...
             'log   ',@(x)myfunc(x,'log',1e-3);...
             'tanh  ',@(x)myfunc(x,'tanh',0,2/3,1.7159)...
             };
      
      [U,S] = eig(X);
      for i = 1: size(type,1)
        % forward test
        fprintf('====> %s <=====\n',type{i,1});

        % compute the function and derivatives at X
        [fwd_vars{:}]   = eigFun(type{i,2},'forward' ,U,S);
        [bwd_vars{:}]   = eigFun(type{i,2},'backward',U,S,dLdC);
        for d = 1: length(deltas)
          delta = deltas(d);
          
          [fwd_vars_dX{:}]        = eigFun(type{i,2},'forward' ,U+proj(U,delta*dU),S);
          err(i,d,1)              = abs((dp(dLdC,fwd_vars_dX{1})-dp(dLdC,fwd_vars{1}))/delta-dp(bwd_vars{1},dU));
          
          [fwd_vars_dX{:}]        = eigFun(type{i,2},'forward' ,U,S+delta*dS);
          err(i,d,2)              = abs((dp(dLdC,fwd_vars_dX{1})-dp(dLdC,fwd_vars{1}))/delta-dp(bwd_vars{2},dS));  
        end
        figure; plot(deltas,squeeze(err(i,:,:))); set(gca,'yscale','log','xscale','log'); title(type{i,1}); legend('dU','d\Sigma');
      end
      
    case 6
      fprintf('=====> reshape layer\n');
      res(1).x = grand(8,5,30,2,'single','gpuArray');
      for i = 1:size(res(1).x,4), 
        masks{i} = ones(size(res(1).x,1),size(res(1).x,2),2,'single'); 
        masks{i}(:,:,1) = masks{i}(:,:,1)./sum(sum(masks{i}(:,:,1)));
        masks{i}(1:2,:,2) = 0;  masks{i}(:,:,2) = masks{i}(:,:,2)./sum(sum(masks{i}(:,:,2)));
      end
      res(2).x = []; res(2).aux = {};
      layer{1} = struct('type','custom_masks','forward',@reshape_forward,'backward',@reshape_backward,'shape','oprod','data_type','double');
      layer{2} = struct('type','custom_masks','forward',@reshape_forward,'backward',@reshape_backward,'shape','iprod','data_type','double');
      layer{3} = struct('type','custom_masks','forward',@reshape_forward,'backward',@reshape_backward,'shape','simple','data_type','double');
      
      fprintf('========> without masks\n');
      for i = 1: length(layer)
        fprintf('%s   ',layer{i}.shape);
        res(2) = layer{i}.forward(layer{i},res(1),res(2)) ;
        res(2).dzdx = grand(size(res(2).x),'single');
        res(1) = layer{i}.backward(layer{i},res(1),res(2)) ;
        vl_testder_custom(@(x) layer{i}.forward(layer{i},x,res(2)), res(1), res(2), 5e-4*range) ;
      end
      
      fprintf('========> with masks\n');
      for i = 1: length(layer)
        fprintf('%s   ',layer{i}.shape);
        res(2) = layer{i}.forward(layer{i},res(1),res(2),masks) ;
        res(2).dzdx = grand(size(res(2).x),'single');
        res(1) = layer{i}.backward(layer{i},res(1),res(2),masks) ;
        vl_testder_custom(@(x) layer{i}.forward(layer{i},x,res(2),masks), res(1), res(2), 5e-4*range) ;
      end
      
    case 7 % check o2p
      fprintf('=====> o2p layer\n');
      res(1).x = grand(4,3,10,2,'single','gpuArray');
      for i = 1:size(res(1).x,4), 
        masks{i} = ones(size(res(1).x,1),size(res(1).x,2),2,'single'); 
        masks{i}(:,:,1) = masks{i}(:,:,1)./sum(sum(masks{i}(:,:,1)));
        masks{i}(1:2,:,2) = 0;  masks{i}(:,:,2) = masks{i}(:,:,2)./sum(sum(masks{i}(:,:,2)));
      end
      res(2).x = []; res(2).aux = {};
      res(3).x = []; res(3).aux = {}; res(3).dzdx = randsym([size(res(1).x,3),size(res(1).x,3),4,1],grand);
      layer{1}{1} = struct('type','custom_masks','forward',@reshape_forward,'backward',@reshape_backward,'shape','oprod','data_type','double');
      layer{1}{2} = struct('type','custom','method','o2p_avg_eig_log','forward',@deepO2P_forward,'backward',@deepO2P_backward);
      layer{2}{1} = struct('type','custom_masks','forward',@reshape_forward,'backward',@reshape_backward,'shape','simple','data_type','double');
      layer{2}{2} = struct('type','custom','method','o2p_avg_svd_log','forward',@deepO2P_forward,'backward',@deepO2P_backward);
      layer{3}{1} = struct('type','custom_masks','forward',@reshape_forward,'backward',@reshape_backward,'shape','oprod','data_type','double');
      layer{3}{2} = struct('type','custom','method','o2p_avg_eig','forward',@deepO2P_forward,'backward',@deepO2P_backward);
      layer{4}{1} = struct('type','custom_masks','forward',@reshape_forward,'backward',@reshape_backward,'shape','simple','data_type','double');
      layer{4}{2} = struct('type','custom','method','o2p_avg_svd','forward',@deepO2P_forward,'backward',@deepO2P_backward);

      for i = 1: length(layer)
        fprintf('%s %s ',layer{i}{1}.shape,layer{i}{2}.method);
        net.layers = layer{1};
        res = vl_mysimplenn(net,res(1).x,res(3).dzdx,[],masks);
        vl_testder_subnet(@(x) vl_mysimplenn(net,x(1).x,[],[],masks), res, 5e-2*range, 2) ;
      end
      
    case 8 % check seg
  end
  
  fprintf('======= test %0d done! ========================\n',l);
end
