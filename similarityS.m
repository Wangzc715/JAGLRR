function [score]=similarityS(A)

[m,n]=size(A);
%  D=pdist(A,'euclidean');%ŷ����¾���
% D =pdist(A,'correlation');%��ؾ���
 D =pdist(A,'chebychev');%�б�ѩ�����
% D = pdist(A,'cosine');%�н����Ҿ���
% D = pdist(A,'spearman');
% D = pdist(A,'jaccard');%�ܿ��¾���
% D = pdist(A,'seuclidean');%	��׼��ŷ�Ͼ���
% D = pdist(A,'mahalanobis');%�����ŵ��˹����
% D = pdist(A,'cityblock');%���н������� �����پ���
% D = pdist(A,'minkowski');%����˹������ �ɿɷ�˹������
% D = pdist(A,'hamming');%��������


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

