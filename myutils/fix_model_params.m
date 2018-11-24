function net = fix_model_params(net)
% this just fixes the parameters

  for j= 1:length(net.layers)        
    if isfield(net.layers{j}, 'pad') && numel(net.layers{j}.pad) == 1, net.layers{j}.pad = repmat(net.layers{j}.pad, [1 4]); end
    if isfield(net.layers{j}, 'stride') && numel(net.layers{j}.stride) == 1, net.layers{j}.stride = repmat(net.layers{j}.stride, [1 2]); end
    if isfield(net.layers{j}, 'pool') && numel(net.layers{j}.pool) == 1, net.layers{j}.pool= repmat(net.layers{j}.pool, [1 2]); end
  end 
end