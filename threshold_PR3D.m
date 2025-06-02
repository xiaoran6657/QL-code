function [best_threshold, best_f1, A2_pred, metrics] = threshold_PR3D(A2, candidates, scores)
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
    
    % 计算评估指标
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
        'Precision', precision, ...
        'Recall', recall, ...
        'F1', f1);
    
    % 绘制PR曲线
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
    
    % 添加阈值分布信息
    figure;
    histogram(scores, 50, 'FaceColor', [0.5, 0.5, 0.9]);
    hold on;
    line([best_threshold, best_threshold], ylim, 'Color', 'r', 'LineWidth', 2);
    hold off;
    title(sprintf('Score Distribution with Best Threshold (%.4f)', best_threshold));
    xlabel('Prediction Score');
    ylabel('Frequency');
    legend('Score Distribution', 'Best Threshold', 'Location', 'best');
    
    % 显示关键指标
    fprintf('=== 最佳阈值评估结果 ===\n');
    fprintf('最佳阈值: %.4f\n', best_threshold);
    fprintf('F1分数: %.4f\n', best_f1);
    fprintf('\n=== 最终预测评估 ===\n');
    fprintf('TP: %d\n', tp_count);
    fprintf('FP: %d\n', fp_count);
    fprintf('FN: %d\n', fn_count);
    fprintf('Precision: %.4f\n', precision);
    fprintf('Recall: %.4f\n', recall);
    fprintf('F1: %.4f\n', f1);
end