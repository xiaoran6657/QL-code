function neig = find_neig(X, n)
    % X=[X1, X2, X3], (N+N+(N-1)*N/2)x1
    % neig: prossible two-body neighbor index

    X1 = X(1:n);
    X2 = X(n+1:2*n);
    X3 = X(2*n+1:end);

    X4 = X1.*X2;

    %thresh1 = graythresh(X1);  % Otsu's Method, 最大类间方差法
    %thresh2 = graythresh(X2);  % Otsu's Method, 最大类间方差法
    unique_Pl = sort(unique(X1, "rows"),'descend');  % 降序排列
    [~, index] = max(unique_Pl(1:end-1).*(unique_Pl(1:end-1)-unique_Pl(2:end))./unique_Pl(2:end)); %Truncation method
    thresh1 = unique_Pl(index+1);
    unique_Pl = sort(unique(X2, "rows"),'descend');  % 降序排列
    [~, index] = max(unique_Pl(1:end-1).*(unique_Pl(1:end-1)-unique_Pl(2:end))./unique_Pl(2:end)); %Truncation method
    thresh2 = unique_Pl(index);

    neig1 = find(X1<=thresh1);
    neig2 = find(X2>=thresh2);
    
    neig = intersect(neig1, neig2);
end
