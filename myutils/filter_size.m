function [outfilt, infilt, xsupp, ysupp] = filter_size(net)
  for i = length(net.layers):-1:1
    if isfield(net.layers{i},'filters')
      [xsupp, ysupp,infilt,outfilt] = size(net.layers{i}.filters);
      break;
    end
  end
end