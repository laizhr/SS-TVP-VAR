function sigma = vcov( epsilon )
%��֪��������Ϊ��ֵ0�������¹���Э����
[m,n] = size(epsilon);
sum = zeros(n,n);
for i = 1:m
    sum = sum + epsilon(i,:)'*epsilon(i,:);
end
sigma = 1/m .*sum;
end

