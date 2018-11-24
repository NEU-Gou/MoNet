function f = randortho(n,m,l,p,randfun,type)
    if ~exist('type','var')
      type = 'double';
    end
    
    f = randfun(n,m,l,p,type);
    for i = 1: p
      [~,ftmp] = qr(reshape(f(:,:,:,i),m*n,l));
      f(:,:,:,i) = reshape(ftmp,[m n l]);
    end
end