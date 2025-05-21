function optimal_threshold = threshold_PR(y_true, y_prob)

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
    figure;
    plot(recall, precision, 'LineWidth', 2);
    hold on;
    scatter(recall(optimal_idx), precision(optimal_idx), 100, 'r*');
    xlabel('Recall');
    ylabel('Precision');
    title('PR Curve (Imbalanced Data)');
    grid on;
    legend('PR Curve', 'Optimal Threshold', 'Location', 'southwest');
    
    % 输出结果
    %y_pred = (y_prob >= optimal_threshold);
        
    %TP = sum(y_true(y_pred) == 1);
    %FP = sum(y_true(y_pred) == 0);
    %FN = sum(y_true(~y_pred) == 1);
    %TN = size(y_true, 1) - TP - FP - FN;

    %fprintf('最佳阈值: %.3f\n', optimal_threshold);
    %fprintf('Precision: %.3f, Recall: %.3f, F1: %.3f\n TP:%d, FP:%d, FN:%d, TN:%d', ...
    %        precision(optimal_idx), recall(optimal_idx), f1_scores(optimal_idx), TP, FP, FN, TN);

end