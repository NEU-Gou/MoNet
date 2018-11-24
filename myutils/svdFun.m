function varargout = svdFun(fun,direction,varargin)
% matFun - compute a matrix function using singular or eigen decomposition
%
% input :
% fun  - multivariate function to apply to diagonalized version of the
%        input matrix
% direction - forward, backward or directional
% varargin - $1=U,$2=S,$3=V,$4=dLdY
% 
% output :
% varargout - 

  if strcmp(direction,'forward')
    if nargin==2, varargout{1} = 1; return; end
    U               = varargin{1};
    S               = varargin{2};
    V               = varargin{3};
    
    F             = diagFun(S'*S,fun);

    varargout{1}  = V*F*V';

  elseif strcmp(direction,'backward')
    if nargin==2, varargout{1} = 2; return; end
    U             = varargin{1};
    S             = varargin{2};
    V             = varargin{3};
    dLdC          = varargin{4};

    [F,Fdev]      = diagFun(S'*S,fun);

    if 0
      % FIXME this should be fixed it's not there yet
      varargout{1}  = 2*symmetric(dLdC)*V*F;
      varargout{2}  = 2*S*dDiag(V'*symmetric(dLdC)*V)*Fdev;
    else
      varargout{1}  = 2*proj(V,symmetric(dLdC)*V*F);
      varargout{2}  = 2*S*dDiag(V'*symmetric(dLdC)*V)*Fdev;
    end
  end
