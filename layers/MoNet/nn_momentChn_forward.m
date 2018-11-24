function now = nn_momentChn_forward(layer,pre,now)

order = layer.order;
powers = layer.powers;

[h,w,c,n] = size(pre.x);

dim = nchoosek(2+order,2)^2;
now.x = gpuArray(zeros(1,1,dim*c,n));

[H,W] = meshgrid(1:h,1:w);
data = cat(2,H(:),W(:));
data = gpuArray(single(data));
for i = 1:n
    for j = 1:c
        tmpw = pre.x(:,:,j,i);
        tmpw = tmpw./sum(tmpw(:));
        
        V = veronese(cat(2,ones(h*w,1),data)',order,powers);
        V = bsxfun(@times, V, 1./sqrt(tmpw(:))');
        tmpx = V*V';
        
        now.x(1,1,dim*(j-1)+1:dim*j) = tmpx(:);
    end
end