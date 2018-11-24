function y = vl_nnfsqrt_m(x, param, varargin)
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

alpha = 0.5;
[M,N,D,L] = size(x);
n = M * N;
X = reshape(x, [n, D, L]);

if backMode
%       X = eval(['double' '(reshape(gather(x),[M*N,D,L]));']); 
%       up_dzdy = double(gather(dzdy));
    
      GX = (X+thresh).^(alpha-1);
      XM = reshape(X,size(X,1)*size(X,2),size(X,3));
      GXM = reshape(GX,size(GX,1)*size(GX,2),size(GX,3));
      GXM(XM<=thresh) = 1;
      GX = reshape(GXM,size(GX,1),size(GX,2),size(GX,3));
      y = reshape(GX,[M,N,D,L]);
      y = y.*dzdy;
      
else
%     for mm =1:size(X,3)
%       X(:,:,mm) = sign(X(:,:,mm)).abs(X(:,:,mm)).^(0.5);
%     end
%     y = reshape(X,[M,N,D,L]);
    
%      y = sign(x).*sqrt(abs(x));
%     x = double(gather(x));
    y = sign(x).*abs(x).^(alpha);
end
% y = gpuArray(y);
