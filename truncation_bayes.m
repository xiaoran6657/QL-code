function [ori_A_bin, triangles_pred] = truncation(ori_A_adj, P3_tensor, A1, A2)
    rng(42); % 固定随机种子
    n = size(ori_A_adj, 1);
    ori_A_bin = zeros(n, n);
    
    %========== 第一部分：二值化邻接矩阵 ==========%
    Pl = ori_A_adj(:);
    fun = @(params) evaluateThreshold(Pl, A1(:), params.Threshold);
    thresholdVar = optimizableVariable('Threshold', [0, 1], 'Type', 'real'); % 固定范围
    
    results = bayesopt(fun, thresholdVar, ...
        'NumSeedPoints', 10, ... % 增加初始点数量
        'MaxObjectiveEvaluations', 50, ... % 增加迭代次数
        'IsObjectiveDeterministic', true, ... % 目标函数是确定性的
        'Verbose', 0, ...
        'PlotFcn', []);
    
    thresh2 = results.XAtMinObjective.Threshold;%获取贝叶斯优化的最优阈值
    
    for i = 1:n
        a = ori_A_adj(i,:);
        a(a >= thresh2) = 1;
        a(a < thresh2) = 0;
        ori_A_bin(i,:) = a;
    end
    ori_A_bin = max(ori_A_bin, ori_A_bin');
    
    %========== 第二部分：处理三体张量 ==========%
    true_triangles = sort(A2, 2);
    true_triangles = unique(true_triangles, 'rows');
    
    triangles = [];
    for i = 1:n
        P3 = P3_tensor(:,:,i);
        Pl = P3(:);
        
        fun_tri = @(params) evaluateTriThreshold(P3, i, true_triangles, params.Threshold);
        thresholdVarTri = optimizableVariable('Threshold', [0, 1], 'Type', 'real'); % 固定范围
        
        results_tri = bayesopt(fun_tri, thresholdVarTri, ...
            'NumSeedPoints', 10, ... % 增加初始点数量
            'MaxObjectiveEvaluations', 30, ... % 增加迭代次数
            'IsObjectiveDeterministic', true, ... % 目标函数是确定性的
            'Verbose', 0, ...
            'PlotFcn', []);
        
        thresh_tri = results_tri.XAtMinObjective.Threshold;
        
        [row, col] = find(P3 >= thresh_tri);
        triangles_i = [repmat(i, length(row), 1), row, col];
        triangles = [triangles; triangles_i];
    end
    
    triangles2 = sortrows(sort(triangles, 2), 1);
    triangles_pred = unique(triangles2, 'rows');
    
    [~, idx_pred] = ismember(triangles_pred, true_triangles, 'rows');
    valid_triangles = triangles_pred(idx_pred > 0, :);
    triangles_pred = valid_triangles;
end

%========== 辅助函数 ==========%
%计算阈值下的F1
function f1 = evaluateThreshold(probabilities, true_labels, threshold)
    num_repeats = 5; % 重复次数
    f1_values = zeros(num_repeats, 1);
    for k = 1:num_repeats
        predicted = double(probabilities >= threshold);
        stats = confusionmatStats(true_labels, predicted);%计算混淆矩阵
        f1_values(k) = -stats.F1; % 取负值用于最小化
    end
    f1 = mean(f1_values); % 取平均值
end

function loss = evaluateTriThreshold(P3, current_node, true_triangles, threshold)
    num_repeats = 5; % 重复次数
    loss_values = zeros(num_repeats, 1);
    for k = 1:num_repeats
        [row, col] = find(P3 >= threshold);
        pred_triangles = sort([repmat(current_node, length(row), 1), row, col], 2);
        is_true = ismember(true_triangles, pred_triangles, 'rows');
        stats = confusionmatStats(logical(is_true), logical(ones(size(is_true))), true);
        loss_values(k) = -stats.F1;
    end
    loss = mean(loss_values); % 取平均值
end

function stats = confusionmatStats(actual, predicted, isLogical)
    if nargin < 3
        isLogical = false;
    end
    
    % 统一数据类型
    if isLogical
        actual = logical(actual);
        predicted = logical(predicted);
    else
        actual = double(actual);
        predicted = double(predicted);
    end
    
    % 统一为列向量
    actual = actual(:);
    predicted = predicted(:);
    
    % 获取所有类别
    classes = union(unique(actual), unique(predicted));
    
    % 生成混淆矩阵
    cm = confusionmat(actual, predicted, 'Order', classes);
    
    % 补全缺失的行或列
    if size(cm,1) < 2
        cm = [cm; zeros(2 - size(cm,1), size(cm,2))];
    end
    if size(cm,2) < 2
        cm = [cm, zeros(size(cm,1), 2 - size(cm,2))];
    end
    
    % 计算指标
    TP = cm(2,2); TN = cm(1,1); FP = cm(1,2); FN = cm(2,1);
    precision = TP / (TP + FP + eps);
    recall = TP / (TP + FN + eps);
    f1 = 2 * (precision * recall) / (precision + recall + eps);
    stats = struct('F1', f1);
end