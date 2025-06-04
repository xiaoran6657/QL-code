function [ACC, F1, ACC_tri, F1_tri] = EvaluationIndicators_Cal3(A1, A2, ori_A_adj, P3_tensor)
    %// TODO 替换PR曲线技术，避免真实信息泄露等学术风险问题

    %Input:
      % A1: Real adjacency matrix, [n, n]
      % A2：Real triangle, [triangle_num, 3]
      % ori_A_adj: Original predicted adjacency matrix, [n,n]
      % P3_tensor: Original predicted tensor of three-body, [n,n,n]
    %Output: Evaluation indicators

    n = size(A1, 1);  % 节点数量
    A1 = full(A1);
    A2 = full(A2);
    ori_A_bin = zeros(n, n);

    % ---------- Truncate the final two-body matrices ----------
    Pl = ori_A_adj(:);
    thresh2 = graythresh(Pl);  % Otsu's Method, 最大类间方差法
    for i = 1:n
        a = ori_A_adj(i,:);
        a(a>=thresh2) = 1;
        a(a<thresh2) = 0;
        ori_A_bin(i,:) = a;
    end
    ori_A_bin = ori_A_bin + ori_A_bin';
    ori_A_bin(ori_A_bin==2) = 1;

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
    

    % 截断二阶边（数值稳定性优化版）
    candidates = [];
    scores = [];
    epsilon = 1e-10;
    scale_factor = 10;  % 根据实际数据调整
    
    % 生成候选三元组并计算缩放后的对数总分
    parfor i = 1:n
        for j = i+1:n
            for k = j+1:n
                if ori_A_bin(i,j) + ori_A_bin(i,k) + ori_A_bin(j,k) >= 1  % 增加判断减少无效候选，放宽条件减少fn
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


    % 计算稀疏性和分数分布特征
    num_edges = sum(ori_A_bin(:)) / 2;
    sparsity = num_edges / nchoosek(n,2);
    skew = skewness(scores);
    kurt = kurtosis(scores);

    % 动态调整k_final
    alpha = 0.02;  % 调参
    beta = 0.01;
    gamma = -0.1;

    % 基础计算
    k_base = alpha * sparsity;
    k_skew = k_base * (1 + beta * skew);
    
    % 峰度处理策略选择
    if kurt > 100
        % 策略1：对数归一化（适用于极端峰度）
        kurt_log = log10(kurt);
        kurt_normalized = 2 * (kurt_log - 1) / 3 - 1;  % 映射log10(100)~log10(10000)到[-1,1]
        kurt_normalized = max(-1, min(1, kurt_normalized));
        k_final = k_skew * (1 + gamma * kurt_normalized);
    elseif kurt > 10
        % 策略2：渐近函数处理
        kurt_adjustment = gamma * sign(kurt-3) * (1 - exp(-0.1*abs(kurt-3)));
        k_final = k_skew * (1 + kurt_adjustment);
    else
        % 策略3：原始公式
        k_final = k_skew * (1 + gamma * (kurt - 3));
    end
    
    % 最终约束
    k_final = max(0.001, min(0.1, k_final));

    % 全局阈值选择
    if ~isempty(scores)
        [scores_sorted, idx] = sort(scores, 'descend');
        %k = round(0.003 * numel(scores_sorted));
        %k = round(k_final * numel(scores_sorted));
        %thresh2 = scores_sorted(max(1, k));
        thresh2 = graythresh(scores_sorted);
        valid_idx = idx(scores_sorted >= thresh2);
        triangles_pred = candidates(valid_idx, :);
    else
        triangles_pred = [];
    end


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


    ori_A_bin_bu = ori_A_bin;
    % ---------- 补充一阶边 ----------
    for idx=1:length(triangles_pred)
        i = triangles_pred(idx, 1);
        j = triangles_pred(idx, 2);
        k = triangles_pred(idx, 3);

        ori_A_bin_bu(i,j)=1; ori_A_bin_bu(j,i)=1;
        ori_A_bin_bu(i,k)=1; ori_A_bin_bu(k,i)=1;
        ori_A_bin_bu(k,j)=1; ori_A_bin_bu(j,k)=1;
    end

    % ---------- 计算指标 ----------
    tp = sum(A1(:) & ori_A_bin_bu(:));    % 真正例
    fp = sum(~A1(:) & ori_A_bin_bu(:));   % 假正例
    fn = sum(A1(:) & ~ori_A_bin_bu(:));   % 假反例
    tn = sum(~A1(:) & ~ori_A_bin_bu(:));  % 真反例

    ACC = (tn + tp) / (tp + fp + fn + tn);  % 避免除以0
    precision = tp / (tp + fp);
    recall = tp / (tp + fn);
    F1 = 2 * (precision * recall) / (precision + recall + eps);

    disp('补充一阶：\n');
    fprintf('tp: %d, fp: %d, fn: %d, tn: %d\n', tp, fp, fn, tn);
    fprintf('ACC: %.4f, Precision: %.4f, Recall: %.4f, F1: %.4f\n', ACC, precision, recall, F1);

end