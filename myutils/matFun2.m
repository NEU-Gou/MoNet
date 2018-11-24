function [f,fdev,U,S2e,V,dLdU,dLdS,dLdV,dLdZ] = matFun2(x,type,fun,epsilon,dLdY,U,S2e,V)
% matFun - compute a matrix function using singular or eigen decomposition
%
% input :
% x    - input matrix
% type - x (asym1, asym2) and xx' (sym1), x'x (sym2) and x (sym3)
% fun  - diagonal function
% 
% output :
% f    - matrix function
% fdev - matrix derivative function

if ~exist('epsilon','var')||isempty(epsilon), epsilon = 0 ; end

[M,N] = size(x);
% FIXME reduce the matrices based to make things numerically more stable

f = []; fdev = []; U = []; S2e = []; V = []; dLdU = []; dLdS = []; dLdV = []; dLdZ = [];

switch type
  case {'asym1','asym2'}
    D = N; I = eye(D);
    if strcmp(type,'asym2'), x = x'; D = M; end
    
    if ~exist('U','var')||isempty(U)
      [U,S,V] = tsvd(x); 
      S2e = S'*S+epsilon*eye(D);
      [F,Fdev] = diagFun(S2e,fun);
      f = V*F*V';
    end
    
    if exist('dLdY','var')&&~isempty(dLdY)
      dLdV = 2*symmetric(dLdY)*V*F;
      dLdS = 2*S*Fdev*(V'*dLdY*V);
      K    = 1./(diag(S2e)*ones(1,D)-(diag(S2e)*ones(1,D))'+I)-I;
      dLdZ = [];
      fdev = U*(2*S*symmetric(K'.*(V'*dLdV))+dDiag(dLdS))*V';
      if strcmp(type,'asym2'), fdev = fdev'; end
    end
    
  case {'sym1','sym2','sym3'}
    if strcmp(type,'sym1'),     I = eye(M); Z = x*x'+epsilon*I; % x*x'
    elseif strcmp(type,'sym2'), I = eye(N); Z = x'*x+epsilon*I; % x'*x
    else                        I = eye(N); Z = x   +epsilon*I; % already symmetric
    end
    
    if ~exist('U','var')||isempty(U)
      [U,S2e] = teig(Z);
      [F,Fdev] = diagFun(S2e,fun);
      f = U*F*U';
    end
    
    if  exist('dLdY','var')&&~isempty(dLdY)
      dLdU = 2*symmetric(dLdY)*U*F; dLdV = [];
      dLdS = Fdev*(U'*dLdY*U);
      Kt   = 1./(diag(S2e)*ones(1,size(S2e,1))-(diag(S2e)*ones(1,size(S2e,1)))'+I)-I;
      dLdZ = U*((2*Kt.*symmetric(U'*dLdU))+dDiag(dLdS))*U'; 
      if strcmp(type,'sym1')
      	fdev = 2*symmetric(dLdZ)*x;
      elseif strcmp(type,'sym2')
        fdev = 2*x*symmetric(dLdZ);
      else
        fdev = dLdZ;
      end
    end
    V = [];
    
  case 'quad' % quadratic
    I     = eye(M);
    f    = x*x'+epsilon*I;
    
    if exist('dLdY','var')&&~isempty(dLdY)
      fdev = 2*symmetric(dLdY)*x;
    end
    
  case 'quadt' % quadratic transposed
    I     = eye(N);
    f    = x'*x+epsilon*I;
    
    if exist('dLdY','var')&&~isempty(dLdY)
      fdev = 2*x*symmetric(dLdY);
    end
    
  otherwise
    error('Type is unrecognized!');
end
