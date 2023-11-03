% The code is written by Jie Wen, 
% if you have any problems, please don't hesitate to contact me: wenjie@hrbeu.edu.cn 
% If you find the code is useful, please cite the following reference:
% Jie Wen , Xiaozhao Fang, Yong Xu, Chunwei Tian, Lunke Fei, 
% Low-Rank Representation with Adaptive Graph Regularization [J], 
% Neural Networks, 2018.
% homepage: https://sites.google.com/view/jerry-wen-hit/home
clear all
%clc
clear memory;
max_iter= 200;
rho = 1.1;
miulist = [0.0001 0.001 0.01 0.1];
load MSRCv1
c = length(unique(Y));   
gnd = Y;
for i = 1:length(X)
    fea = X{i}';
    fea = fea./repmat(sqrt(sum(fea.^2)),[size(fea,1) 1]);
    X{i} = fea;
end

%F_ini = cell(1,length(X));
Z_ini = cell(1,length(X));
Ctg = cell(1,length(X));
L_t = sparse(length(gnd));
for i = 1:length(X)
% ---------- initilization for Z and F -------- %
options = [];
options.NeighborMode = 'KNN';
options.k = 10;
options.WeightMode = 'Binary';      % Binary  HeatKernel
Z = constructW(X{i}',options);
Z = full(Z);
Z1 = Z-diag(diag(Z));         
Z = (Z1+Z1')/2;
DZ= diag(sum(Z));
LZ = DZ - Z;  
L_t = L_t + LZ;
Z_ini{i} = Z;
clear LZ DZ Z fea Z1
Ctg{i} = inv(X{i}'*X{i}+2*eye(size(X{i},2)));
end
[F_ini, ~, evs]=eig1(L_t, c, 0);
lambda1list = [0.000001 0.00001 0.0001];
lambda2list = [0.000001 0.00001 0.0001 0.001 0.01 0.1 1 10 100 1000];
lambda3list = [0.000001 0.00001 0.0001 0.001 0.01 0.1 1 10 100 1000];
for i = 1:length(miulist)
for ii = 1:length(lambda1list)
    for iii = 1:length(lambda2list)
        for iiii = 1:length(lambda3list)
            miu = miulist(i);
            lambda1 = lambda1list(ii);
            lambda2 = lambda2list(iii);
            lambda3 = lambda3list(iiii);
            tic;
[Z,S,U,F,E,obj] = LRR_AGR(X,F_ini,Z_ini,c,lambda1,lambda2,lambda3,max_iter,rho,miu,Ctg);
toc
label = litekmeans(F, c, 'MaxIter', 100, 'Replicates', 20);
result = ClusteringMeasure(gnd, label);
dlmwrite('result_MSRCv1_F.txt',[lambda1, lambda2, lambda3, miu, rho, length(obj), result],'-append','delimiter','\t','newline','pc');

Zt = sparse(length(gnd));
for jj = 1:length(Z)
    Zt = Zt + abs(Z{jj});
end
Zt = Zt/length(Z);

C = max(Zt,Zt');

addpath('Ncut_9');
Z_out = C;
A = Z_out;
A = A - diag(diag(A));
A = (A+A')/2;  
[NcutDiscrete,NcutEigenvectors,NcutEigenvalues] = ncutW(A,c);
[value,result_label] = max(NcutDiscrete,[],2);
result = ClusteringMeasure(gnd, result_label);
dlmwrite('result_MSRCv1_NCut.txt',[lambda1, lambda2, lambda3, miu, rho, length(obj), result],'-append','delimiter','\t','newline','pc');
        end
    end
end
end