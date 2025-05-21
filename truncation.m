function [ori_A_bin, triangles_pred] = truncation(ori_A_adj, P3_tensor)
    n = size(ori_A_adj, 1);
    ori_A_bin = zeros(n, n);
    
    %Truncate the final two-body matrices
    Pl = ori_A_adj(:);
    %Pl(Pl<0)=0; Pl(Pl>1)=1;  % preprocessing
    %unique_Pl = sort(unique(Pl, "rows"),'descend');  % 降序排列
    %[~, index] = max(unique_Pl(1:end-1).*(unique_Pl(1:end-1)-unique_Pl(2:end))./unique_Pl(2:end)); %Truncation method
    %thresh2 = unique_Pl(index);
    thresh2 = graythresh(Pl);  % Otsu's Method, 最大类间方差法
    for i = 1:n
        a = ori_A_adj(i,:);
        a(a>=thresh2) = 1;
        a(a<thresh2) = 0;
        ori_A_bin(i,:) = a;
    end
    ori_A_bin = ori_A_bin + ori_A_bin';
    ori_A_bin(ori_A_bin==2) = 1;
    
    %Truncate the final three-body
    triangles = [];  % 记录重构的每个三角形的节点
    for i = 1:n
        neig = find(ori_A_bin(i,:)==1);
        P3 = P3_tensor(:,:,i);  % 获取节点i的P3矩阵
        Pl = P3(:);
        Pl(Pl>1) = 1; Pl(Pl<0)=0;
        thresh2 = graythresh(Pl);  % Otsu's Method, 最大类间方差法

        [row, col] = find(P3 >= thresh2); % 找到小于阈值元素的位置索引，直接使用row和col作为节点编号
        %row = []; col = [];
        %for j = 1:n
        %   for k = 1:n
        %        if P3(j,k)>thresh2 && (ismember(j,neig) || ismember(k,neig))
        %            row = [row; j];
        %            col = [col; k];
        %        end
        %    end
        %end

        % 这里直接使用row和col作为节点编号
        triangles_i = [repmat(i, length(row), 1), row, col]; % 生成临时矩阵
        triangles = [triangles; triangles_i]; % 竖着摞在一起
    end

    % triangles:给triangles排序去重
    %[triangles, ~, ic] = unique(sort(triangles, 2), 'rows', 'stable');
    triangles2 = sortrows(sort(triangles,2),1); 
    triangles_pred = unique(triangles2,'rows');

    %Set the conditions for the existence of three-body
    num_triangles_pred=size(triangles_pred,1);  
    if num_triangles_pred ~= 0
        number_triangles_pred=[triangles_pred,zeros(num_triangles_pred,1)]; %Add a column of record times
        for j=1:num_triangles_pred
            number_triangles_pred(j,4)=length(find(triangles2(:,1)==triangles_pred(j,1) & triangles2(:,2)==triangles_pred(j,2)  & triangles2(:,3)==triangles_pred(j,3) )); 
        end
        triangles_pred(number_triangles_pred(:,4)==1,:)=[];
    end
end
