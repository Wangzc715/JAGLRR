function [score]=similarityS(A)

[m,n]=size(A);
%  D=pdist(A,'euclidean');%欧几里德距离
% D =pdist(A,'correlation');%相关距离
 D =pdist(A,'chebychev');%切比雪夫距离
% D = pdist(A,'cosine');%夹角余弦距离
% D = pdist(A,'spearman');
% D = pdist(A,'jaccard');%杰卡德距离
% D = pdist(A,'seuclidean');%	标准化欧氏距离
% D = pdist(A,'mahalanobis');%马哈拉诺比斯距离
% D = pdist(A,'cityblock');%城市街区距离 曼哈顿距离
% D = pdist(A,'minkowski');%明考斯基距离 闵可夫斯基距离
% D = pdist(A,'hamming');%汉明距离


score=zeros(n,n);
m=1;
for i=1:n
    for j=i+1:n
        score(i,j)=D(1,m);
        m=m+1;
    end
end
score=score./max(max(score));
score=score+score';
score=score./max(max(score));
score=1./(score+1);

end

