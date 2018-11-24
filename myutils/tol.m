function tau = tol(a,b)
  maxv = max([max(a(:)), max(b(:))]) ;
  minv = min([min(a(:)), min(b(:))]) ;
  tau = 1e-2 * (maxv - minv) + 1e-4 * max(maxv, -minv) ;