function vl_testder_custom(g,lower,upper,delta,tau)
% test a layer that is setup in the format of the 'custom' layer

dzdy = upper.dzdx;
dzdx = lower.dzdx;
x = lower.x;

if nargin < 4
  delta = 1e-3 ;
end

if nargin < 5
  tau = [] ;
end

dzdy = gather(dzdy) ;
dzdx = gather(dzdx) ;

y = gather(g(lower)) ;
dzdx_=zeros(size(dzdx));
for i=1:numel(x)
  lower_.x = x ;
  lower_.x(i) = lower_.x(i) + delta ;
  y_ = g(lower_) ;
  factors = dzdy .* (gather(y_.x) - gather(y.x))/delta; % delta
  dzdx_(i) = dzdx_(i) + sum(factors(:)) ;
end
[pass,tau, error] = vl_testsim(dzdx , dzdx_, tau);
if pass
  fprintf('PASS at tolerance %e -- error %e\n',tau,error);
else
  fprintf('FAIL at tolerance %e -- error %e\n',tau,error);
end

