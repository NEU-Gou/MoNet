function [f,fdev] = myfunc(x,type,varargin)
% myfunc - implements a coordinate-wise function and their corresponding 
%          derivatives
%
% input: 
%       x - input vector
%       type - one of the supported types
%       varargin - other parameters parsed by the function. Parameters
%       below are in bash notation ($1, $2, etc)
% 
% output:
%       f - function value (if x is empty then the number of params)
%       fdev - derivative of the function at x
%
% supported types are:
% tanh      - implements .5*$3*(tanh((x+$1)/$2)+1)
% log       - log(x+$1)
% id        - identity
% heaviside - step function 0, x\in(-\infty,$1]; 1,otherwise
% custom    - custom function with handles to function ($1) and the function
%             derivative ($2) passed in varargin
%
% (c) 2015 -- catalin ionescu - catalin.ionescu@ins.uni-bonn.de

  switch type
    case 'tanh'
      if isempty(x), f = 2; return; end
      alpha = varargin{1};
      beta = varargin{2};
      gamma = varargin{3};
      f = tanh((x+alpha)./beta);
      if nargout==2
        fdev = .5*gamma*(1-f.^2)./beta;
      end
      f = .5*gamma*(f+1);
    case 'log'
      if isempty(x), f = 1; return; end
      epsilon = varargin{1};
      f = log(x+epsilon); 
      if nargout==2
        fdev = 1./(x+epsilon);
      end
    case 'id'
      if isempty(x), f = 0; return; end
      f = x;
      if nargout==2
        fdev = ones(size(x));
      end
    case 'quad'
      if isempty(x), f = 1; return; end
      f = x*x'+varargin{1}*eye(size(x,1));
      if nargout==2
        fdev = 2*x;
      end
    case 'heaviside'
      if isempty(x), f = 1; return; end
      epsilon = varargin{1};
      f = double(x>epsilon); 
      if nargout==2
        fdev = zeros(size(f)); 
      end
    case 'custom'
      fun = varargin{1};
      devfun = varargin{2};
      f = fun(x);
      fdev = devfun(x);
    otherwise
      error('Unsupported function type!');
  end
end