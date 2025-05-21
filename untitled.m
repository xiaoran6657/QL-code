clc, clear;
rng(12);

n=100;
nchoose2 = nchoosek(n,2);
A=rand(n,n+nchoose2+n^2+n*nchoose2+nchoose2^2,'gpuArray');
A_Delta=rand(n,n+nchoose2+n^2+n*nchoose2+nchoose2^2,'gpuArray');

lambda = 0.3;
alpha = log(1-lambda);
S = randi([0,1],[n,1]);

X1 = alpha*S;
X2 = X1*X1'; X2 = X2(:);

Y1 = A(:,1:n)*X1 + A(:,n+1:n+nchoose2)*X2 + A(:,n+nchoose2+1:n+nchoose2+n^2)*kron(X1,X1) + ...
    2*A(:,n+nchoose2+n^2+1:n*nchoose2+n^2+n*nchoose2)*kron(X1,X2)+A(:,n*nchoose2+n^2+n*nchoose2+1:n*nchoose2+n^2+n*nchoose2+nchoose2^2)*kron(X2,X2);


tic;
[X1_opt, X2_opt, history] = admm_bfgs(A1, A2, Y, 4000, 1e-6);
toc;
tic;
[X1_opt, X2_opt, history] = ADMM_GPU(A1, A2, Y, 50000, 1e-6);
toc;

[X1_opt, history] = ADMM_GPU2(A1, A2, Y, 2000, 1e-6);

X1_pred = zeros(n,1);
temp = X1_opt(X1_opt<0);
thresh2 = graythresh(-temp);  % Otsu's Method, 最大类间方差法

X1_pred(X1_opt<-thresh2) = 1;  % X1_opt是负的



t= [1,2,3,4,5];  % [n,n]
temp = t'*t;
A3=(temp(logical(tril(ones(size(temp)),-1))))';