function [ACC, F1, ACC_tri, F1_tri] = EvaluationIndicators_Cal2(A1, A2, ori_A_bin, triangles_pred)
    %Input:
      % A1: Real adjacency matrix
      % A2：Real triangle
      % ori_A_bin: Predicted adjacency matrix
      % triangles_pred: Predicted triangle
    %Output:Evaluation indicators

    n = size(A1, 1);  % 节点数量
    A1 = full(A1);
    A2 = full(A2);
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