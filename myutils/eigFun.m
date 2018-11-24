function varargout = eigFun(fun,direction,varargin)
% matFun - compute a matriX function using singular or eigen decomposition
%           This function is tensor aware.
%
% input :
% X    - input matrix
% type - eigfun,svdfun
% fun  - diagonal function
% 
% output : 
%

U               = varargin{1};
S               = varargin{2};

if strcmp(direction,'forward')
  if nargin==2, varargout{1} = 1; return; end

  F             = diagFun(S,fun);
  [M,N,L]       = size(S); assert(M==N,'Dimension mismatch!');
  Z             = zeros(M,M,L);
  
  parfor i = 1: L
    Z(:,:,i) = U(:,:,i)*F(:,:,i)*U(:,:,i)';
  end
  
  varargout{1} = Z;
  
elseif strcmp(direction,'backward')
  if nargin==2, varargout{1} = 2; return; end
  dLdC          = varargin{3};

  [F,Fdev]      = diagFun(S,fun);

  if 1
    [M,N,L] = size(U);
    dLdU = zeros(M,M,L);
    dLdZ = zeros(M,M,L);
    parfor i = 1: L
      dLdU(:,:,i)  = 2*symmetric(dLdC(:,:,i))*U(:,:,i)*F(:,:,i);
      dLdZ(:,:,i)  = Fdev(:,:,i)*(U(:,:,i)'*dLdC(:,:,i)*U(:,:,i));
    end
    
    varargout{1} = dLdU;
    varargout{2} = dLdZ;
  else
    varargout{1}  = 2*proj(U,symmetric(dLdC)*U*F);
    varargout{2}  = dDiag(U'*symmetric(dLdC)*U)*Fdev;
  end
  
end
    
