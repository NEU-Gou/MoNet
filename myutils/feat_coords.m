function F = feat_coords(net,sz_im,layer_id)
% this implementation treats each dimension independently 

  if nargin == 2
    layer_id = length(net.layers);
  end
  x = 1: sz_im(1);
  y = 1: sz_im(2);


  for j= 1:layer_id
    l = net.layers{j};

    if(strcmp(l.type,'conv'))
      pad_dist_x = x(2)-x(1); % we assume spacing is equal between points
      pad_left = x(1)-pad_dist_x:-pad_dist_x: x(1)-pad_dist_x*l.pad(1); % compute coordinates for the padding to the left
      pad_right = x(end)+pad_dist_x:pad_dist_x: x(end)+pad_dist_x*l.pad(3); % compute coordinates for the padding to the right
      x = [pad_left x pad_right]; % add the paddings
      x = conv(x,ones(1,size(l.filters,1))/size(l.filters,1),'valid');
      x = x(1:l.stride(1):end);  % leave the room for the last one
      
      pad_dist_y = y(2)-y(1); % we assume spacing is equal between points
      pad_left = y(1)-pad_dist_y:-pad_dist_y: y(1)-pad_dist_y*l.pad(2); % compute coordinates for the padding to the left
      pad_right = y(end)+pad_dist_y:pad_dist_y: y(end)+pad_dist_y*l.pad(4); % compute coordinates for the padding to the right
      y = [pad_left y pad_right]; % add the paddings
      y = conv(y,ones(1,size(l.filters,2))/size(l.filters,2),'valid');
      y = y(1:l.stride(2):end);  % leave the room for the last one
%       x = x ((size(l.filters,1)/2) - l.pad(1):l.stride(1):length(x)-(size(l.filters,1)/2)+ l.pad(3));  % leave the room for the last one
%       y = y ((size(l.filters,2)/2) - l.pad(2):l.stride(2):length(y)-(size(l.filters,2)/2)+ l.pad(4));
    elseif (strcmp(l.type,'pool'))
      x = conv(x,ones(1,l.pool(1))/l.pool(1),'valid');
      x = x(1:l.stride(1):end);
      y = conv(y,ones(1,l.pool(2))/l.pool(2),'valid');
      y = y(1:l.stride(2):end);
%       x = x ((l.pool(1)/2) - l.pad(1):l.stride(1):length(x)-l.pool(1)/2 + l.pad(3));  % leave the room for the last one
%       y = y ((l.pool(2)/2) - l.pad(2):l.stride(2):length(y)-l.pool(2)/2 + l.pad(4));
    else
        % do nothing 
    end

    if 0&&strcmp(l.type,'pool')
      hold on;
      line([1 sz_im(1) sz_im(1) 1 1],[1 1 sz_im(2) sz_im(2) 1]); hold on;
      xall = reshape(repmat(x',[1 length(y)]),length(x)*length(y),1);
      yall = reshape(repmat(y,[length(x) 1]),length(x)*length(y),1);
      plot(xall, yall, 'o'); axis equal; axis ij;
      x
    end    
  end

  F(1,:) = reshape(repmat(x',[1 length(y)]),length(x)*length(y),1); 
  F(2,:) = reshape(repmat(y,[length(x) 1]),length(x)*length(y),1);

  F = floor(F);
  
  if 0
    figure; line([1 sz_im(1) sz_im(1) 1 1],[1 1 sz_im(2) sz_im(2) 1]); hold on;
    plot(F(1,:), F(2,:), 'o'); axis equal; axis ij;
  end    
end
