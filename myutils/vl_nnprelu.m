function [y,dzdp] = vl_nnprelu(x,dzdy,p)
% VL_NNPRELU CNN rectified linear unit parametric

if nargin <= 1 || isempty(dzdy)
  y = max(x, single(0)) + p *min(x,single(0));
else
  y = dzdy .* (x > single(0)) + p*(dzdy.*(x<=single(0)));
  dzdp = dzdy.*(x.*(x<=single(0)));
end
