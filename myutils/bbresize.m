function bb = bbresize(bb,ratio)
  % bbresize is a function that resizes a bounding box to a certain size
  % given the original bounding box and the image size. This allows us to
  % crop with background instead of cropping then padding with black
  if bb(3)>bb(4)
      delta = (bb(3)-bb(4))*.5;
      bb(2) = max(1,bb(2)-delta);
      bb(4) = bb(4)+2*delta;
  else
      delta = (bb(4)-bb(3))*.5;
      bb(1) = max(1,bb(1)-delta);
      bb(3) = bb(3)+2*delta;
  end
end