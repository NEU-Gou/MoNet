function pass = vl_testder_subnet(g,res,delta,tau)
% test a set of layers

dzdy = res(end).dzdx;
% dzdx = res(1).dzdw;
dzdx = res(1).dzdx;
x = res(1).x;

if nargin < 3
  delta = 1e-3 ;
end

if nargin < 4
  tau = [] ;
end

dzdy = gather(dzdy) ;
dzdx = gather(dzdx) ;

y = g(res);
dzdx_=zeros(size(dzdx));
fprintf('%06d of %06d\n',0,numel(dzdx));
for i=1:numel(dzdx)
  fprintf('\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b%06d of %06d\n',i,numel(dzdx));
  res_.x = x ;
  res_.x(i) = res_.x(i) + delta ;
  y_ = g(res_) ;
  factors = dzdy .* (gather(y_(end).x) - gather(y(end).x))/delta ;
  dzdx_(i) = dzdx_(i) + sum(factors(:)) ;
end
fprintf('\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b');
[pass,tau, error] = vl_testsim(dzdx , dzdx_, tau);
if pass
  fprintf('PASS at tolerance %e -- error %e\n',tau,error);
else
  fprintf('FAIL at tolerance %e -- error %e\n',tau,error);
end

