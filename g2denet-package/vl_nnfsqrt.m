function y = vl_nnfsqrt(x, param, varargin)
% VL_NNFSQRT perform square normalization for the input features
% at each location
%
% Author: Qilong Wang, Peihua Li, Lei Zhang

%
% This file is part of the G2DeNet 


thresh = param(1);

backMode = numel(varargin) > 0 && ~isstr(varargin{1}) ;
if backMode
  dzdy = varargin{1} ;
end

[M,N,D,L] = size(x);
n = M * N;
X = reshape(x, [n, D, L]);

if backMode
      GX = (0.5).*(X+thresh).^(0.5-1);
      XM = reshape(X,size(X,1)*size(X,2),size(X,3));
      GXM = reshape(GX,size(GX,1)*size(GX,2),size(GX,3));
      GXM(XM<=thresh) = 1;
      GX = reshape(GXM,size(GX,1),size(GX,2),size(GX,3));
      y = reshape(GX,[M,N,D,L]);
      y = y.*dzdy;
      
else
     y = sign(x).*sqrt(abs(x));
%     y = sign(x).*abs(x).^(0.5);
end

