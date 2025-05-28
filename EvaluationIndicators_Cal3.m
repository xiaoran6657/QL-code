function [ACC, F1, ACC_tri, F1_tri] = EvaluationIndicators_Cal3(A1, A2, ori_A_adj, P3_tensor)
    %// TODO 替换PR曲线技术，避免真实信息泄露等学术风险问题

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

    % ---------- Truncate the final two-body matrices ----------
    Pl = ori_A_adj(:);
    %thresh2 = threshold_PR(A1(:), Pl, 1);  % PR曲线选择最佳阈值
    thresh2 = graythresh(Pl);  % Otsu's Method, 最大类间方差法
    for i = 1:n
        a = ori_A_adj(i,:);
        a(a>=thresh2) = 1;
        a(a<thresh2) = 0;
        ori_A_bin(i,:) = a;
    end
    ori_A_bin = ori_A_bin + ori_A_bin';
    ori_A_bin(ori_A_bin==2) = 1;
    

    % ---------- Truncate the final three-body tensor ----------
    
%{
 triangles = [];  % 记录重构的每个三角形的节点
    for i = 1:n
        %neig = find(ori_A_bin(i,:)==1);
        P3 = P3_tensor(:,:,i);  % 获取节点i的P3矩阵
        Pl = P3(:);
        %Pl(Pl>1) = 1; Pl(Pl<0)=0;
        %thresh2 = graythresh(Pl);  % Otsu's Method, 最大类间方差法
        %A2i = A2_tensor(:,:,i);
        %thresh2 = threshold_PR(A2i(:), Pl, 0);  % PR曲线选择最佳阈值
        Pl_sorted = sort(Pl, 'descend');
        k = round(0.005 * numel(Pl_sorted));
        thresh2 = Pl_sorted(k);  % 基于密度的阈值选取

        %[row, col] = find(P3 >= thresh2); % 找到小于阈值元素的位置索引，直接使用row和col作为节点编号
        row = []; col = [];
        for j = 1:n
           for k = 1:n
                if P3(j,k)>thresh2 && ori_A_bin(i,j) && ori_A_bin(i,k) && ori_A_bin(j,k)
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
%}

    % 截断二阶边（数值稳定性优化版）
    candidates = [];
    scores = [];
    epsilon = 1e-10;
    scale_factor = 100;  % 根据实际数据调整
    
    % 生成候选三元组并计算缩放后的对数总分
    parfor i = 1:n
        for j = i+1:n
            for k = j+1:n
                if ori_A_bin(i,j) + ori_A_bin(i,k) + ori_A_bin(j,k) > 1  % 增加判断减少无效候选，放宽条件减少fn
                %if 1
                    % 数值缩放与对数转换
                    log_P3_ijk = log(P3_tensor(j,k,i) * scale_factor + P3_tensor(k,j,i) * scale_factor + epsilon);
                    log_P3_ikj = log(P3_tensor(i,k,j) * scale_factor + P3_tensor(k,i,j) * scale_factor + epsilon);
                    log_P3_jki = log(P3_tensor(i,j,k) * scale_factor + P3_tensor(j,i,k) * scale_factor + epsilon);
                    % 总分计算
                    total_log_score = log_P3_ijk + log_P3_ikj + log_P3_jki;
                    total_score = exp(total_log_score);
                    
                    candidates = [candidates; [i,j,k]];
                    scores = [scores; total_score];
                end
            end
        end
    end

    
%{
 % 计算稀疏性和分数分布特征
    num_edges = sum(ori_A_bin(:)) / 2;
    sparsity = num_edges / nchoosek(n,2);
    skew = skewness(scores);
    kurt = kurtosis(scores);

    % 动态调整k_final
    alpha = 1.5;
    beta = 0.3;
    gamma = -0.2;
    k_base = alpha * sparsity;
    k_skew = k_base * (1 + beta * skew);
    k_final = k_skew * (1 + gamma * (kurt - 3));
    k_final = max(0.01, min(0.1, k_final)); 
%}


    % 全局阈值选择
    if ~isempty(scores)
        [scores_sorted, idx] = sort(scores, 'descend');
        k = round(0.032 * numel(scores_sorted));
        %k = round(k_final * numel(scores_sorted));
        thresh2 = scores_sorted(max(1, k));
        valid_idx = idx(scores_sorted >= thresh2);
        triangles_pred = candidates(valid_idx, :);
    else
        triangles_pred = [];
    end

    % ---------- 补充一阶边 ----------
    


    
    % ---------- 计算指标 ----------
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

        % 使用 ismember 替代双重循环
        tp = sum(ismember(A2t, triangles_pred, 'rows'));
        fp = size(triangles_pred, 1) - tp;  % 计算 FP（B 中不在 A 中的部分）
        fn = size(A2t, 1) - tp;  % 计算 FN（A 中不在 B 中的部分）
        tn = nchoosek(n,3) - tp - fp - fn;  % 正确计算所有可能三元组数量

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