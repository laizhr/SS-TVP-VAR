function sigma = vcov( epsilon )
%已知先验条件为均值0的条件下估计协方差
[m,n] = size(epsilon);
sum = zeros(n,n);
for i = 1:m
    sum = sum + epsilon(i,:)'*epsilon(i,:);
end
sigma = 1/m .*sum;
end

