function net = init_alexnet(scal,init_bias)
  addpath ../../o2p-release1/external_src/
  DefaultVal('*scal','1');
  DefaultVal('*init_bias','.1');

  net.layers = {} ;

  % Block 1
  net.layers{end+1} = struct('name', 'conv1', ...
                             'type', 'conv', ...
                             'filters', 0.01/scal * randn(11, 11, 3, 96, 'single'), ...
                             'biases', zeros(1, 96, 'single'), ...
                             'stride', 4, ...
                             'pad', 0, ...
                             'filtersLearningRate', 1, ...
                             'biasesLearningRate', 2, ...
                             'filtersWeightDecay', 1, ...
                             'biasesWeightDecay', 0) ;
  net.layers{end+1} = struct('name', 'relu1', ...
                             'type', 'relu') ;
  net.layers{end+1} = struct('name', 'mpool1', ...
                             'type', 'pool', ...
                             'method', 'max', ...
                             'pool', [3 3], ...
                             'stride', 2, ...
                             'pad', 0) ;
  net.layers{end+1} = struct('name', 'lcn1', ...
                             'type', 'normalize', ...
                             'param', [5 1 0.0001/5 0.75]) ;

  % Block 2
  net.layers{end+1} = struct('name', 'conv2', ...
                             'type', 'conv', ...
                             'filters', 0.01/scal * randn(5, 5, 48, 256, 'single'), ...
                             'biases', init_bias*ones(1, 256, 'single'), ...
                             'stride', 1, ...
                             'pad', 2, ...
                             'filtersLearningRate', 1, ...
                             'biasesLearningRate', 2, ...
                             'filtersWeightDecay', 1, ...
                             'biasesWeightDecay', 0) ;
  net.layers{end+1} = struct('name', 'relu2', ...
                             'type', 'relu') ;
  net.layers{end+1} = struct('name', 'mpool2', ...
                             'type', 'pool', ...
                             'method', 'max', ...
                             'pool', [3 3], ...
                             'stride', 2, ...
                             'pad', 0) ;
  net.layers{end+1} = struct('name', 'lcn2', ...
                             'type', 'normalize', ...
                             'param', [5 1 0.0001/5 0.75]) ;

  % Block 3
  net.layers{end+1} = struct('name', 'conv3', ...
                             'type', 'conv', ...
                             'filters', 0.01/scal * randn(3,3,256,384,'single'), ...
                             'biases', init_bias*ones(1,384,'single'), ...
                             'stride', 1, ...
                             'pad', 1, ...
                             'filtersLearningRate', 1, ...
                             'biasesLearningRate', 2, ...
                             'filtersWeightDecay', 1, ...
                             'biasesWeightDecay', 0) ;
  net.layers{end+1} = struct('name', 'conv3', ...
                             'type', 'relu') ;

  % Block 4
  net.layers{end+1} = struct('name', 'conv4', ...
                             'type', 'conv', ...
                             'filters', 0.01/scal * randn(3,3,192,384,'single'), ...
                             'biases', init_bias*ones(1,384,'single'), ...
                             'stride', 1, ...
                             'pad', 1, ...
                             'filtersLearningRate', 1, ...
                             'biasesLearningRate', 2, ...
                             'filtersWeightDecay', 1, ...
                             'biasesWeightDecay', 0) ;
  net.layers{end+1} = struct('name', 'relu4', ...
                             'type', 'relu') ;

  % Block 5
  net.layers{end+1} = struct('name', 'conv5', ...
                             'type', 'conv', ...
                             'filters', 0.01/scal * randn(3,3,192,256,'single'), ...
                             'biases', init_bias*ones(1,256,'single'), ...
                             'stride', 1, ...
                             'pad', 1, ...
                             'filtersLearningRate', 1, ...
                             'biasesLearningRate', 2, ...
                             'filtersWeightDecay', 1, ...
                             'biasesWeightDecay', 0) ;
  net.layers{end+1} = struct('name', 'relu5', ...
                             'type', 'relu') ;
  net.layers{end+1} = struct('name', 'mpool3', ...
                             'type', 'pool', ...
                             'method', 'max', ...
                             'pool', [3 3], ...
                             'stride', 2, ...
                             'pad', 0) ;

  % Block 6
  net.layers{end+1} = struct('name', 'fc6', ...
                             'type', 'conv', ...
                             'filters', 0.01/scal * randn(6,6,256,4096,'single'),...
                             'biases', init_bias*ones(1,4096,'single'), ...
                             'stride', 1, ...
                             'pad', 0, ...
                             'filtersLearningRate', 1, ...
                             'biasesLearningRate', 2, ...
                             'filtersWeightDecay', 1, ...
                             'biasesWeightDecay', 0) ;
  net.layers{end+1} = struct('name', 'relu6', ...
                             'type', 'relu') ;
  net.layers{end+1} = struct('type', 'dropout', ...
                             'rate', 0.5) ;

  % Block 7
  net.layers{end+1} = struct('name', 'fc7', ...
                             'type', 'conv', ...
                             'filters', 0.01/scal * randn(1,1,4096,4096,'single'),...
                             'biases', init_bias*ones(1,4096,'single'), ...
                             'stride', 1, ...
                             'pad', 0, ...
                             'filtersLearningRate', 1, ...
                             'biasesLearningRate', 2, ...
                             'filtersWeightDecay', 1, ...
                             'biasesWeightDecay', 0) ;
  net.layers{end+1} = struct('name', 'relu7', ...
                             'type', 'relu') ;
  net.layers{end+1} = struct('type', 'dropout', ...
                             'rate', 0.5) ;

  % Block 8
  net.layers{end+1} = struct('name', 'fc8', ...
                             'type', 'conv', ...
                             'filters', 0.01/scal * randn(1,1,4096,1000,'single'), ...
                             'biases', zeros(1, 1000, 'single'), ...
                             'stride', 1, ...
                             'pad', 0, ...
                             'filtersLearningRate', 1, ...
                             'biasesLearningRate', 2, ...
                             'filtersWeightDecay', 1, ...
                             'biasesWeightDecay', 0) ;

  % Block 9
  net.layers{end+1} = struct('type', 'softmaxloss') ;
end