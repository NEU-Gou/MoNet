function [y,dzdx_] = autoGradMat(x,fun,dzdy,delta)
% autoGrad - this needs to be verified

  y = fun(x);
  dzdx_=zeros(size(x));
  for i=1:numel(x)
    x_ = x ;
    x_(i) = x_(i) + delta ;    
    y_ = fun(x_) ;
    factors = dzdy .* (y_ - y)/delta ;
    dzdx_(i) = dzdx_(i) + sum(factors(:)) ;
  end
end