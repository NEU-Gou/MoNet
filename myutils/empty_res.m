function res = empty_res(num_layers)
  for i = 1: num_layers
    res(i).x = []; 
    res(i).aux = {}; 
  end
end