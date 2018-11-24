function varargout = matDecomp(type,direction,varargin)
% matDecomp - matrix decomposition routines. We implement here the forward 
%           pass with the possibility of saving all the variables, the forward
%           as well as the backward sensitivities. This code was and can be
%           numerically checked using test_nnlayers.
%
% input:
% x        - input data
% type     - decomposition type (eig,[svd])
% varargin - other variables
% 
% output:
% varargout - the decomposition result
% 
% (c) 2015 -- catalin ionescu - catalin.ionescu@ins.uni-bonn.de

switch type
  %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
  case 'eig'
    if strcmp(direction,'forward')
      if nargin==2, varargout{1} = 2; return; end
      
      X                           = varargin{1};
      [M,N,L] = size(X); assert(M==N,'Symmetric eigenvalue requires a square matrix!');
      
      U = zeros(M,N,L,'double');
      S = zeros(M,N,L,'double');
      
      parfor i = 1: L
        [U(:,:,i),S(:,:,i)] = eig(X(:,:,i));
      end
      
      varargout{1} = U;
      varargout{2} = S;

    elseif strcmp(direction,'backward')
      if nargin==2, varargout{1} = 1; return; end
      
      U             = varargin{1};
      S             = varargin{2};
      dLdU          = varargin{3};
      dLdS          = varargin{4};
      
      D = size(U,1); I = eye(D); one = ones(1,D); L = size(U,3);
      
      parfor i = 1: L
        Kt            = 1./((diag(S(:,:,i))*one)'-diag(S(:,:,i))*one+I)-I;      
      
        if 0
          Z(:,:,i)  = U(:,:,i)*(2*(Kt'.*symmetric(U(:,:,i)'*dLdU(:,:,i)))+dDiag(dLdS(:,:,i)))*U(:,:,i)';
        else
          Z(:,:,i)  = U(:,:,i)'\(Kt'.*(U(:,:,i)'*dLdU(:,:,i))+dDiag(dLdS(:,:,i)))*U(:,:,i)';
        end
      end
      
      varargout{1} = Z;
      
    elseif strcmp(direction,'directional')
      if nargin==2, varargout{1} = 1; return; end  
      U             = varargin{1};
      S             = varargin{2};
      dX            = varargin{3};
            
      D = size(U,1); I = eye(D);
      Kt            = 1./((diag(S)*ones(1,D))'-diag(S)*ones(1,D)+I)-I;      
      
      if 1
        dP            = U'*dX*U;
        varargout{1}  = U*(Kt'.*symmetric(dP));       % dU      
        varargout{2}  = dDiag(dP);                    % dS     
      else
        dP            = U\(dX*U);        
        varargout{1}  = U*(Kt'.*dP);                  % dU      
        varargout{2}  = dDiag(dP);                    % dS     
      end
    end
    
  %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
  case 'svd'
    if strcmp(direction,'forward')
      if nargin==2, varargout{1} = 3; return; end
      
      X                                         = varargin{1};
      
      [M,N,L] = size(X);
      
      U = zeros(M,M,L,'double');
      S = zeros(M,N,L,'double');
      V = zeros(N,N,L,'double');
      
      parfor i = 1: L
        [U(:,:,i),S(:,:,i),V(:,:,i)] = svd(X(:,:,i));
      end
      
      varargout{1} = U;
      varargout{2} = S;
      varargout{3} = V;
      
    elseif strcmp(direction,'backward')
      if nargin==2, varargout{1} = 1; return; end
      
      U             = varargin{1};
      S             = varargin{2};
      V             = varargin{3};
      dLdV          = varargin{4};
      dLdS          = varargin{5};
      
      D = size(U,1); I = eye(D);
      
      % FIXME get support for the svd with U as well instead of just V
      K               = 1./(diag(S).^2*ones(1,D)-(diag(S).^2*ones(1,D))'+I)-I;      
      if 1
        varargout{1}  = U*(2*S*symmetric(K'.*(V'*dLdV))+dDiag(dLdS))*V';
      else
        varargout{1}  = U*((2*S*symmetric(K'.*(V'*dLdV))+dDiag(dLdS))*V');
      end
      
    elseif strcmp(direction,'directional')
      if nargin==2, varargout{1} = 1; return; end  
      U             = varargin{1};
      S             = varargin{2};
      V             = varargin{3};
      dX            = varargin{4};
            
      D = size(U,1); I = eye(D);
      K               = 1./(diag(S).^2*ones(1,D)-(diag(S).^2*ones(1,D))'+I)-I;      

      dP = U'*dX*V;
      varargout{1}  = 2*V*(K'.*symmetric(S'*dP)); % dV      
      varargout{2}  = dDiag(dP);                  % dS
      varargout{3}  = 2*U*(K'.*symmetric(S*dP')); % dU
    end
    
  %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    
  otherwise
    error('Decomposition not (yet) supported!');
end

end