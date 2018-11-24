function o = dp(A,B)
% tr(A*B)
% o = trace(A'*B);
o = sum(sum(A.*B));
end