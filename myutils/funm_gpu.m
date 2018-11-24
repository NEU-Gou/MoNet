function [lM,U,S,V] = funm_gpu(M,fun,diag_method,epsilon)
% log matrix on the gpu. This should be done better using QR decomposition
% but this is quick and dirty
% eigen decomposition
if ~exist('diag_method','var'), diag_method = 'svd'; end
% if size(M,1)~=size(M,2)
if ~exist('epsilon','var')
  epsilon = 1;
end

switch diag_method
  case 'svd'
    try
      [U,S,V] = svd(M);
    catch E
%       warning('bad svd putting on some noise!');
%       [U,S,V] = svd(M+eps(max(M(:)))*ones(size(M)));

      warning('Bad SVD setting things to 0!');
      S = eye(size(M));
      U = zeros(size(M,1));
      V = zeros(size(M,2));
    end
    S2 = S'*S+epsilon*eye(size(S,2)); 
    lM = V*diag(fun(diag(S2)))*V';%funm(M'*M,fun);%
  case 'eig'
%     [U,S] = schur(M+epsilon*eye(size(M,1))); V = U;
%     [U,S,V] = svd(M+epsilon*eye(size(M,1)));
    [U,S] = eig(M+epsilon*eye(size(M,1))); V = U;
    diagS = diag(S);
    ind = diagS>epsilon+eps(max(diagS));
    diagS(~ind)= epsilon; S = diag(diagS); % chopping eigen values
%     lM = V*diag(fun(diag(S)))*V';
    lM = V(:,ind)*diag(fun(diagS(ind)))*V(:,ind)';
  otherwise
    error('unknown diag_method!');
end

%   S(S>0) = S(S>0)+epsilon;
  
% else
%   diagS = diag(S); [m,n] = size(M);
%   threshold = max(m,n) * eps(max(diagS)); % compute the threshold
%   % threshold = max(m,n) * eps('single');
%   r = gather(sum(diagS>threshold));% cap the small eigenvalues
%   diagS(r:end) = threshold;
%   U(:,r:end) = 0; V(:,r:end) = 0; 
%   lM = U*diag(fun(diagS))*V';
%   S = diag(diagS);
end
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%
% max(max(abs((C+eI)-(U(:,1:r)*diag(diagS(1:r))*V(:,1:r)'))))
% max(max(abs((C+eI)*(U(:,1:r)*diag(1./diagS(1:r))*V(:,1:r)')-eye(size(C)))))
% mean(C(:)+eI(:))

%% older attemps
% [v,s] = eig(M); 
% [u,v,s] = svd(M);
% diagS = diag(s);
% % threshold = 1e-10;
% [m,n] = size(M);
% threshold = max(m,n) * eps(max(diagS)); % compute the threshold
% r = gather(sum(diagS>threshold));% cap the small eigenvalues
% v = v(:,1:r); u = u(:,1:r);
% diagS = diagS(1:r);
% lM = v*diag(log(diagS))*u';
% s = diag(diagS);

% figure; imagesc(M-v*s*v');