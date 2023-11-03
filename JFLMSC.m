function [Z,M,J,F,E,obj] = JFLMSC(X,F_ini,Z_ini,lambda1,lambda2,lambda3,max_iter,rho,miu,Ctg)
% If you find the code is useful, please cite the following reference:
% Simultaneously Learning Feature-wise Weights and Local Structures for Multi-view Subspace Clustering 
% KnowledgeBased Systems, 2020.
max_miu = 1e8;
Obj = zeros(1,max_iter);

dwf = sparse(1:size(X,1),1:size(X,1),1/size(X,1) * ones(1,size(X,1)));


[m,n] = size(X);
% ---------- Initilization -------- %
C1 = zeros(m,n);
C2 = zeros(n,n);
C3 = zeros(n,n);
E  = zeros(m,n);
S = zeros(n, n);
distX = L2_distance_1(X,X);
S = similarityS(X);
for iter = 1:max_iter
    if iter == 1
        Z = Z_ini;
        F = F_ini;
        M = Z_ini;
        J = Z_ini;
        clear Z_ini F_ini
    end
    
   
    % -------- Update Z --------- %
    Z = Ctg*(X'*(X-E+C1/miu)+2*lambda1*S+M+J-(C2+C3)/miu);
    Z = Z- diag(diag(Z));
    % -------- Update M --------- %
    distF = L2_distance_1(F',F');            % ¼ÆËãvij
    distX = L2_distance_1(dwf*X,dwf*X);
%     dist  = distX+lambda1*distF;
    M    = Z+(C2-distX)/miu;
    M     = M- diag(diag(M));
    for ic = 1:n
        idx    = 1:n;
        idx(ic) = [];
        M(ic,idx) = EProjSimplex_new(M(ic,idx));          % 
    end
    % -------- Update w --------- %
    A0 = (M+M')/2;
    D0 = diag(sum(A0));
    L0 = D0 - A0;
    dw = diag(X*L0*X');
    tdw = dw.^-1;
    dwf = sparse(1:size(X,1),1:size(X,1),(dw * sum(tdw)).^-1);
    % -------- Update J --------- %
      
    
     temp = Z+C3/miu;
%     [UH,sigmaH,VH] = svd(temp,'econ');
%     UH(isnan(UH)) = 0;
%     VH(isnan(VH)) = 0;
%     sigmaH(isnan(sigmaH)) = 0;
%     sigmaH = diag(sigmaH);
%     SVP = length(find(sigmaH>lambda2/miu));
%     if SVP >= 1
%         sigmaH = sigmaH(1:SVP)-lambda2/miu;
%     else
%         SVP = 1;
%         sigmaH = 0;
%     end
%     U = UH(:,1:SVP)*diag(sigmaH)*VH(:,1:SVP)';
        [UH,sigmaH,VH] = svd(temp,'econ');
        SU = sigmaH;
        canshu = lambda2;
        mu = miu;
        for time=1:size(sigmaH,1)
         if sum(diag(sigmaH(1:time,1:time)))>canshu/mu
            if time<size(sigmaH,1)
                if ((sum(diag(sigmaH(1:time,1:time)))-canshu/mu)/time)>=sigmaH(time+1,time+1)
                    tsigma=(sum(diag(sigmaH(1:time,1:time)))-canshu/mu)/time;
                    break
                end
            else
                tsigma=(sum(diag(sigmaH(1:time,1:time)))-canshu/mu)/time;
            end
         else
            if time==size(sigmaH,1)
                tsigma=0;
            end
         end
        end
    
       for cba=1:time
        sigmaH(cba,cba)=tsigma;
       end
       J = UH*sigmaH*VH';
    % ------- Update E ---------- %
%     temp1 = X-X*Z+C1/miu;
%     temp2 = lambda3/miu;
%     E = max(0,temp1-temp2) + min(0,temp1+temp2);    

    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%E NORM21
    P = X-X*Z+C1/miu;
     for i=1:1:m
       if (lambda3/miu)<norm(P(i,:),2) 
         E(i,:)= ((norm(P(i,:),2)-lambda2/mu)/norm(P(i,:),2))*P(i,:);
       else
         E(i,:)=0;
       end
     end
    % -------- Update C1 C2 C3 miu -------- %
    L1 = X-X*Z-E;
    L2 = Z-M;
    L3 = Z-J;
    C1 = C1+miu*L1;
    C2 = C2+miu*L2;
    C3 = C3+miu*L3;
    
    % ---------- Update F ----------- %
    L = sparse(n);
   
    LS = (M+M')/2;
    L = L + diag(sum(LS)) - LS;
  
    [F, ~, ev] = eig1(L, 0);
    
    LL1 = sparse(1,length(X));
    LL2 = LL1;
    LL3 = LL1;
    %LL4 = LL1;
    
    LL1 = norm(X-X*Z-E,inf);
    LL2 = norm(Z-M,inf);
    LL3 = norm(Z-J,inf);
   
    

    % --------- obj ---------- %
    t_1 = 0;
    L_Z = cell(1,length(X));
    
    L = (abs(Z)+abs(Z'))/2;
    L_Z =diag(sum(L)) - L;
    t_1 = t_1 + trace(X*L_Z*X'); 
 
    t_2 = 0;
     
    t_2 = t_2 + max(svd(Z));
    
    t_3 = 0;
     
    t_3 = t_3 + norm(E,1);
    
    t_4 = 0;
     
    t_4 = t_4 + trace(F'*L_Z*F);
     
    Obj(iter) = Obj(iter) + t_1 + t_2 * lambda2 + t_3*lambda3...
        +t_4*lambda1;
    % ---------- miu ------------- %
    miu = min(rho*miu,max_miu);
    if ((max(LL1) < 1e-6 && max(LL2) < 1e-6 && max(LL3) < 1e-6) ||...
            iter > 2 && ((abs(Obj(iter)-Obj(iter-1))/Obj(iter-1) < 1e-6)))
        iter
        break;
    end
end
obj = Obj(1:iter);