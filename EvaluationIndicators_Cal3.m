function [ACC, F1, ACC_tri, F1_tri] = EvaluationIndicators_Cal3(A1, A2, ori_A_adj, P3_tensor)
    %Input:
      % A1: Real adjacency matrix
      % A2：Real triangle
      % ori_A_adj: Original predicted adjacency matrix
      % P3_tensor: Original predicted tensor of three-body
    %Output: Evaluation indicators

    n = size(A1, 1);  % 节点数量
    A1 = full(A1);
    A2 = full(A2);
    ori_A_bin = zeros(n, n);
    
    %A2_tensor = zeros(n, n, n);
    %for i = 1:size(A2, 1)
    %    A2_tensor(A2(i,1), A2(i,2), A2(i,3)) = 1;
    %    A2_tensor(A2(i,1), A2(i,3), A2(i,2)) = 1;
    %    A2_tensor(A2(i,2), A2(i,1), A2(i,3)) = 1;
    %    A2_tensor(A2(i,2), A2(i,3), A2(i,1)) = 1;
    %    A2_tensor(A2(i,3), A2(i,1), A2(i,2)) = 1;
    %    A2_tensor(A2(i,3), A2(i,2), A2(i,1)) = 1;
    %end

    %Truncate the final two-body matrices
    Pl = ori_A_adj(:);
    thresh2 = threshold_PR(A1(:), Pl);  % PR曲线选择最佳阈值
    %thresh2 = graythresh(Pl);  % Otsu's Method, 最大类间方差法
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
        %A2i = A2_tensor(:,:,i);
        %thresh2 = threshold_PR(A2i(:), Pl);  % PR曲线选择最佳阈值
        thresh2 = graythresh(Pl);  % Otsu's Method, 最大类间方差法

        %[row, col] = find(P3 >= thresh2); % 找到小于阈值元素的位置索引，直接使用row和col作为节点编号
        row = []; col = [];
        for j = 1:n
           for k = 1:n
                if P3(j,k)>thresh2 && (ismember(j,neig) && ismember(k,neig))
                    row = [row; j];
                    col = [col; k];
                end
            end
        end
        
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

    
    % 计算指标 
    tp = sum(A1(:) & ori_A_bin(:));    % 真正例
    fp = sum(~A1(:) & ori_A_bin(:));   % 假正例
    fn = sum(A1(:) & ~ori_A_bin(:));   % 假反例
    tn = sum(~A1(:) & ~ori_A_bin(:));  % 真反例

    ACC = (tn + tp) / (tp + fp + fn + tn);  % 避免除以0
    precision = tp / (tp + fp);
    recall = tp / (tp + fn);
    F1 = 2 * (precision * recall) / (precision + recall + eps);

    disp('一阶：\n');
    fprintf('tp: %d, fp: %d, fn: %d, tn: %d\n', tp, fp, fn, tn);
    fprintf('ACC: %.4f, Precision: %.4f, Recall: %.4f, F1: %.4f\n', ACC, precision, recall, F1);
    
    if ~isempty(triangles_pred)
        A2t = sortrows(sort(A2,2),1);  % 三元组内升序排列，三元组间升序排列
        num1 = size(A2t, 1); num2 = size(triangles_pred, 1);

%{
 tp=0;
        for i = 1:num1
            for j = 1:num2
                if isequal(A2t(i,:), triangles_pred(j,:))
                    tp = tp+1;  % 计算 TP（A 和 B 的交集）
                    break
                end
            end
        end
        fprintf("for循环：%d\n",tp); 
%}

        % 使用 ismember 替代双重循环
        [~, idx] = ismember(A2t, triangles_pred, 'rows');
        tp = nnz(idx);
        
        fp = num2 - tp;  % 计算 FP（B 中不在 A 中的部分）
        fn = num1 - tp;  % 计算 FN（A 中不在 B 中的部分）
        tn = n*n*(n-1) - tp - fp - fn;

        ACC_tri = (tn + tp) / (tp + fp + fn + tn);  % 避免除以0
        precision_tri = tp / (tp + fp);
        recall_tri = tp / (tp + fn);
        F1_tri = 2 * (precision_tri * recall_tri) / (precision_tri + recall_tri + eps);
    else
        tp=0; fp=0; fn=0; tn=0;
        ACC_tri=0; precision_tri=0; recall_tri=0; F1_tri=0;
    end
    
    disp('二阶：\n');
    fprintf('tp: %d, fp: %d, fn: %d, tn: %d\n', tp, fp, fn, tn);
    fprintf('ACC: %.4f, Precision: %.4f, Recall: %.4f, F1: %.4f\n', ACC_tri, precision_tri, recall_tri, F1_tri);
end