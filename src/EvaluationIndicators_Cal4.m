function [ACC, F1, ACC_tri, F1_tri] = EvaluationIndicators_Cal4(A1, A2, ori_A_adj, P3_tensor, opt)
    %%% PR曲线查看重建结果分离度，实验结果不可用于论文表述 %%%

    %Input:
      % A1: Real adjacency matrix, [n,n]
      % A2：Real triangle, [triangles_num, 3]
      % ori_A_adj: Original predicted adjacency matrix, [n,n]
      % P3_tensor: Original predicted tensor of three-body, [n,n,n]
    %Output: Evaluation indicators

    n = size(A1, 1);  % 节点数量
    A1 = full(A1);
    A2 = full(A2);
    ori_A_bin = zeros(n, n);

    % ---------- Truncate the final two-body matrices ----------
    Pl = ori_A_adj(:);
    if opt == "PR"
        thresh2 = threshold_PR(A1(:), Pl, 0);  % PR曲线选择最佳阈值
    else
        thresh2 = graythresh(Pl);  % Otsu's Method, 最大类间方差法
    end

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
    

    % ---------- Truncate the final three-body tensor ----------
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

    if opt == "PR"
        metrics = threshold_PR3D(A2, candidates, scores, n, 0);  % PR曲线选择最佳阈值
        disp('二阶：\n');
        fprintf('tp: %d, fp: %d, fn: %d, tn: %d\n', metrics.TP, metrics.FP, metrics.FN, metrics.TN);
        fprintf('ACC: %.4f, Precision: %.4f, Recall: %.4f, F1: %.4f\n', metrics.Acc, metrics.Precision, metrics.Recall, metrics.F1);
    
        ACC_tri = metrics.Acc;
        F1_tri = metrics.F1;
    else
        %{
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
        %}

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
        for idx=1:size(triangles_pred,1)
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
end

function optimal_threshold = threshold_PR(y_true, y_prob, show)

    % 生成示例数据（稀疏网络，正类占比5%）
    rng(42);
    %N = 1000;
    %y_true = [ones(50,1); zeros(950,1)]; % 50正样本，950负样本
    %y_prob = [0.02 + 0.1*rand(50,1); 0.1*rand(950,1)]; % 正类概率高，负类概率低
    
    % 步骤1：计算PR曲线数据
    [y_prob_sorted, sort_idx] = sort(y_prob, 'descend');
    y_true_sorted = y_true(sort_idx);
    thresholds = unique(y_prob_sorted);
    %thresholds = [inf; thresholds]; 
    
    num_thresholds = length(thresholds);
    precision = zeros(num_thresholds, 1);
    recall = zeros(num_thresholds, 1);
    
    for i = 1:num_thresholds
        T = thresholds(i);
        y_pred = (y_prob >= T);
        
        TP = sum(y_true(y_pred) == 1);
        FP = sum(y_true(y_pred) == 0);
        FN = sum(y_true(~y_pred) == 1);
        
        precision(i) = TP / (TP + FP + eps);
        recall(i) = TP / (TP + FN + eps);
    end
    
    % 步骤2：选择最佳阈值（最大化F1）
    f1_scores = 2 * (precision .* recall) ./ (precision + recall + eps);
    [~, optimal_idx] = max(f1_scores);
    optimal_threshold = thresholds(optimal_idx);
    
    % 步骤3：可视化
    if show
        figure;
        plot(recall, precision, 'LineWidth', 2);
        hold on;
        scatter(recall(optimal_idx), precision(optimal_idx), 100, 'r*');
        xlabel('Recall');
        ylabel('Precision');
        title('PR Curve (Imbalanced Data)');
        grid on;
        legend('PR Curve', 'Optimal Threshold', 'Location', 'southwest');
    end
end

function metrics = threshold_PR3D(A2, candidates, scores, n, show)
    % 数据准备与预处理
    gt_set = unique(A2, 'rows'); % 去除重复的真实三元组
    total_positives = size(gt_set, 1);
    
    % 按得分排序候选三元组
    [sorted_scores, sort_idx] = sort(scores, 'descend');
    sorted_candidates = candidates(sort_idx, :);
    
    % 初始化变量
    tp = 0;
    fp = 0;
    matched_gt = false(total_positives, 1); % 跟踪已匹配的真实三元组
    precisions = zeros(size(scores));
    recalls = zeros(size(scores));
    
    % 动态计算PR曲线点
    for i = 1:size(sorted_candidates, 1)
        candidate = sorted_candidates(i, :);
        
        % 检查是否命中真实三元组
        is_match = false;
        for j = 1:size(gt_set, 1)
            if isequal(candidate, gt_set(j, :)) && ~matched_gt(j)
                is_match = true;
                matched_gt(j) = true;
                break;
            end
        end
        
        if is_match
            tp = tp + 1;
        else
            fp = fp + 1;
        end
        
        % 计算当前累积指标
        current_tp = tp;
        current_fp = fp;
        
        if (current_tp + current_fp) > 0
            precisions(i) = current_tp / (current_tp + current_fp);
        else
            precisions(i) = 0;
        end
        
        if total_positives > 0
            recalls(i) = current_tp / total_positives;
        else
            recalls(i) = 0;
        end
    end
    
    % 寻找最佳阈值（最大化F1）
    best_f1 = -1;
    best_idx = -1;
    best_threshold = -1;
    
    for i = 1:length(precisions)
        p = precisions(i);
        r = recalls(i);
        if (p + r) > 0
            f1 = 2 * p * r / (p + r);
        else
            f1 = 0;
        end
        
        if f1 > best_f1
            best_f1 = f1;
            best_idx = i;
            best_threshold = sorted_scores(i);
        end
    end
    
    % 使用最佳阈值生成最终预测
    threshold_mask = scores >= best_threshold;
    A2_pred = candidates(threshold_mask, :);
    
    % 计算混淆矩阵指标
    tp_set = intersect(A2_pred, gt_set, 'rows');
    fp_set = setdiff(A2_pred, gt_set, 'rows');
    fn_set = setdiff(gt_set, A2_pred, 'rows');
    
    tp_count = size(tp_set, 1);
    fp_count = size(fp_set, 1);
    fn_count = size(fn_set, 1);
    tn_count = nchoosek(n,3) - tp_count - fp_count - fn_count;
    
    % 计算评估指标
    acc = (tn_count + tp_count) / nchoosek(n,3);
    if (tp_count + fp_count) > 0
        precision = tp_count / (tp_count + fp_count);
    else
        precision = 0;
    end
    
    if (tp_count + fn_count) > 0
        recall = tp_count / (tp_count + fn_count);
    else
        recall = 0;
    end
    
    if (precision + recall) > 0
        f1 = 2 * precision * recall / (precision + recall);
    else
        f1 = 0;
    end
    
    % 存储指标
    metrics = struct(...
        'TP', tp_count, ...
        'FP', fp_count, ...
        'FN', fn_count, ...
        'TN', tn_count, ...
        'Acc', acc, ...
        'Precision', precision, ...
        'Recall', recall, ...
        'F1', f1);
    
    % 绘制PR曲线
    if show
        figure;
        plot(recalls, precisions, 'b-', 'LineWidth', 1.5);
        hold on;
        plot(recalls(best_idx), precisions(best_idx), 'ro', ...
            'MarkerSize', 10, 'LineWidth', 2);
        hold off;
        
        title(sprintf('Precision-Recall Curve (Best F1=%.3f @ Threshold=%.4f)', best_f1, best_threshold));
        xlabel('Recall');
        ylabel('Precision');
        legend('PR Curve', 'Best Threshold', 'Location', 'best');
        grid on;
    end
end