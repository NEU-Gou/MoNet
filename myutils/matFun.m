function varargout = matFun(type,direction,varargin)
% matFun - compute a matriX function using singular or eigen decomposition
%
% input :
% X    - input matrix
% type - fro, opnorm, quad, quadt
% fun  - diagonal function
% 
% output :
% varargout - 

switch type
%   %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%   case 'fro'
%     if strcmp(direction,'forward')
%       if nargin==2, varargout{1} = 1; return; end
%       
%       X             = varargin{1};
%       
%       varargout{1}  = .5 * norm(X,'fro').^2;
%       
%     elseif strcmp(direction,'backward')
%       if nargin==2, varargout{1} = 1; return; end
% 
%       X             = varargin{1};
%       dLdY          = varargin{2};
%       
%       varargout{1}  = dLdY*X;
%       
%     elseif strcmp(direction,'directional')
%       if nargin==2, varargout{1} = 1; return; end
%       
%       varargout{1}  = ;
%     end
    
  %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
  case 'quad' % quadratic
    if strcmp(direction,'forward')
      if nargin==2, varargout{1} = 1; return; end
      
      I             = eye(M);
      
      varargout{1}  = X*X'+epsilon*I;
      
    elseif strcmp(direction,'backward')
      if nargin==2, varargout{1} = 1; return; end

      varargout{1}  = 2*symmetric(dLdY)*X;
      
    elseif strcmp(direction,'directional')
      if nargin==2, varargout{1} = 1; return; end
      
      dX            = varargin{1};
      
      varargout{1}  = 2*symmetric(X*dX');
    end
    
  %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
  case 'quadt' % quadratic transposed
    if strcmp(direction,'forward')
      if nargin==2, varargout{1} = 1; return; end

      I             = eye(N);
      
      varargout{1}  = X'*X+epsilon*I;
      
    elseif strcmp(direction,'backward')
      if nargin==2, varargout{1} = 1; return; end

      varargout{1}  = 2*X*symmetric(dLdY);
      
    elseif strcmp(direction,'directional')
      if nargin==2, varargout{1} = 1; return; end
      
      dX            = varargin{1};
      
      varargout{1}  = 2*symmetric(X'*dX);
    end
    
  otherwise
    error('Type is unrecognized!');
end
