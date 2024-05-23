X = vol(1:1209,1:29);
beta = beta_vol_0308;

%参数设置
[time,firm] = size(X);
Y = X(2:time,:);
Z = X(1:time-1,:);%p = 1
%e.g. if p = 3,Xp =[X(3:time-1,:),X(2:time-2,:),X(1:time-3,:)]
epsilon = Y-Z*beta;
lambda = [0.1,0.99];
lengl = length(lambda);
kappa = [0.96,0.98];
lengk = length(kappa);

%变量声明


%shrink，剔除变量
V = cell(time-1,firm);
L = V;
shrink = cell(firm,1);
ZZ = cell(firm,1);
for i = 1:firm
    if all(beta(:,i)==0)
        continue
    else
        shrink{i} = find(beta(:,i));
        ZZ{i} = Z(:,shrink{i});
        V{1,i} = (epsilon(:,i)'*epsilon(:,i))*(ZZ{i}(:,:)'*ZZ{i}(:,:))^(-1);
    end
end

%初始值

vcovepsilon=vcov(epsilon);


TV_epsilon = zeros(time-1,firm);
TV_beta = cell(time-1,lengl,lengk);
TV_sigma = TV_beta;
for l = 1:lengl
        for k =1:lengk
            for i = 2:time-1
                TV_beta{i,l,k}=zeros(firm);
                TV_sigma{i,l,k}=TV_beta{i,l,k};
            end
TV_beta{1,l,k} = beta;
TV_sigma{1,l,k} = vcovepsilon;
        end
end





%TV_beta{1} = beta;
%TV_sigma{1} = vcovepsilon;
TV_epsilon(1,:) = Y(1,:) - Z(1,:) * beta;

TV_beta_fin=cell(time-1,1);
for i = 1:time-1
    TV_beta_fin{i,1}=zeros(firm);
end
TV_sigma_fin = TV_beta_fin;
TV_beta_fin{1} = beta;
TV_sigma_fin{1} = vcovepsilon;



likelihood = cell(time-1,1);%转换概率矩阵
for i = 1:time-1
    likelihood{i,1}=zeros(lengl,lengk);
end
%lam_kap = zeros(time-1,2);



%Kalman filtering
for j = 1:time-2
    for l = 1:lengl
        for k =1:lengk
            for i = 1:firm
                if all(beta(:,i)==0)
                    continue
                else
                    if j == 1
                        V{j,i} = V{1,i};
                    else
                        V{j,i} = (1/lambda(l))*(eye(size(shrink{i},1))-L{j-1,i}*ZZ{i}(j-1,:))*V{j-1,i};
                    end
                    L{j,i} = V{j,i}*ZZ{i}(j,:)'*(ZZ{i}(j,:)*V{j,i}*ZZ{i}(j,:)'+TV_sigma{j,l,k}(i,i))^(-1);
                    TV_beta{j+1,l,k}(shrink{i},i) = TV_beta{j,l,k}(shrink{i},i)+L{j,i}*TV_epsilon(j,i);
                    TV_epsilon(j+1,i) = Y(j+1,i) - Z(j+1,:) * TV_beta{j+1,l,k}(:,i);
                end
            end
            TV_sigma{j+1,l,k} = kappa(k) * TV_sigma{j,l,k} + (1-kappa(k)) * TV_epsilon(j+1,:)'*TV_epsilon(j+1,:);
            %Variance = abs(diag(diag(TV_sigma{j+1})));
            likelihood{j+1}(l,k) = mvnpdf(TV_epsilon(j+1,:),0,TV_sigma{j+1,l,k});
        end
    end
    [lw,kw] = find(likelihood{j+1} == max(max(likelihood{j+1})));
  %  lam_kap(j,:) = [lambda(lw(1)),kappa(kw(1))];
    TV_beta_fin{j+1}=TV_beta{j+1,lw(1),kw(1)};
    TV_sigma_fin{j+1}=TV_sigma{j+1,lw(1),kw(1)};
  %  TV_beta_fin=TV_beta(:,lw(1),kw(1));
  %  TV_sigma_fin=TV_sigma(:,lw(1),kw(1));
  
    
    
    
end

%variance decompsition
%方差分解和VAR阶数相关，这里为1阶VAR
H = 5;%predictive horizon
A = cell(time-1,H);
for t = 1:time-1
    A{t,1} = eye(firm);
    for h = 1:H-1
        A{t,h+1} = TV_beta_fin{t}*A{t,h};
    end
end
%connectedness table
con = cell(time-1,1);
for t = 1:time-1
    con{t} = zeros(firm+1);
end
for t = 1:time-1
    for i = 1:firm
        for j = 1:firm
            fz = 0;
            for h = 1:H
                fz = fz+(A{t,h}(i,:)*TV_sigma_fin{t}(:,j))^2;
            end
            fm = 0;
            for h = 1:H
                fm = fm+A{t,h}(i,:)*TV_sigma_fin{t}*A{t,h}(i,:)';
            end    
            con{t}(i,j) = (TV_sigma_fin{t}(j,j)^(-1)*fz)/(fm);
        end
        
        c = sum(con{t}(i,1:firm));
        for j = 1:firm
            con{t}(i,j) = con{t}(i,j)/c;
        end
    end
    con{t}(1:firm,firm+1) = sum(con{t}(1:firm,1:firm),2)-diag(con{t}(1:firm,1:firm));%from others to i
    con{t}(firm+1,1:firm) = sum(con{t}(1:firm,1:firm))-diag(con{t}(1:firm,1:firm))';%from j to others
    con{t}(firm+1,firm+1) = 1/firm*sum(con{t}(:,firm+1));%total connectedness
end

%total connectedness 绘图
 total_con = zeros(time-1,1);
for t =1:time-1
    total_con(t) = con{t}(firm+1,firm+1);
end
plot(1:time-1,total_con)

