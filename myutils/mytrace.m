function v = mytrace(A,B)
  A=reshape(A,size(A,1)*size(A,2),size(A,3),size(A,4));
  B=reshape(B,size(B,1)*size(B,2),size(B,3),size(B,4));
  for i = 1: size(A,3)
    % dot product then flatten and sum
    v(i) = sum(reshape(A(:,:,i).*B(:,:,i),size(A,1)*size(A,2),1));
  end
end