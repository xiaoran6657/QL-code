clear,clc;

%%
pathname1 = '..\matData\';  % 单纯复形
pathname3 = '..\RematData\';

pathname2 = '..\matData2\';  % 超图
pathname4 = '..\RematData2\';

% 获取指定文件夹下的所有 .mat 文件
files = dir(fullfile(pathname1, '*.mat'));
% 提取文件名
fileNames = {files.name};
networkType = 'ER';  % 选定网络类型
Lambda = 1e-3;  % lasso parameter
Timespan = [1000, 2000, 4000, 7000, 10000, 15000, 20000, 30000, 40000, 50000];

%% Reconstruction Network
for i = 1:length(fileNames)
    filename = fileNames{i};

    result = strsplit(filename, '_');
    % Extraction networks parameters
    numbers = regexp(result{3}, '\d+', 'match');
    nNodes = str2double(numbers);
    %numbers = regexp(result{5}, '\d+', 'match');
    %kB = str2double(numbers);

    %matches = regexp(filename, 'm(\d+)', 'tokens');
    %numbers = cellfun(@(x) str2double(x{1}), matches); % 将捕获的数字字符串转换为数值
    %if startsWith(filename, networkType) && endsWith(filename, strcat(num2str(numTriangles), '.mat'))
    if startsWith(filename, networkType) && nNodes==100
        disp(filename)
        load(strcat(pathname1, filename));
        %[ori_A_adj, P3_tensor] = Reconstruction_TS(UAU_state_nodes, SIS_state_nodes, Lambda);
        %[ori_A_adj, P3_tensor] = Reconstruction_TS_GPU(UAU_state_nodes, SIS_state_nodes);
        %[ori_A_adj, P3_tensor] = Reconstruction_TStaylor2_GPUUUU(UAU_state_nodes, SIS_state_nodes);
        [ori_A_adj, P3_tensor] = Reconstruction_TStaylor6_GPU(UAU_state_nodes, SIS_state_nodes);
        save(strcat(pathname3, 'TSL', num2str(Lambda), '_', filename), 'A1', 'A2', 'B', 'ori_A_adj', 'P3_tensor');
    end
end

%% Evaluation Indicators Calculation
files2 = dir(fullfile(pathname3, '*.mat'));  % 获取指定文件夹下的所有 .mat 文件
fileNames2 = {files2.name};  % 提取文件名

ACC_two_body = zeros(length(Timespan), 2);
ACC_three_body = zeros(length(Timespan), 2);
F1_two_body = zeros(length(Timespan), 2);
F1_three_body = zeros(length(Timespan), 2);
k=1;
for i = 1:length(fileNames2)
    filename = fileNames2{i};
    result = strsplit(filename, '_');
    % Extraction networks parameters
    numbers = regexp(result{5}, '\d+', 'match');
    kA = str2double(numbers);
    numbers = regexp(result{6}, '\d+', 'match');
    kB = str2double(numbers);
    
    if startsWith(filename, strcat('L', num2str(Lambda), '_', networkType)) && kA==4 && kB==4 && endsWith(filename, strcat(num2str(numTriangles), '.mat'))
        disp(filename)
        
        matches = regexp(filename, 'm(\d+)', 'tokens');
        numbers = cellfun(@(x) str2double(x{1}), matches); % 将捕获的数字字符串转换为数值

        load(strcat(pathname3, filename));
        %P3_tensor = -P3_tensor; % 重构结果由负变正
        %ori_A_adj = -ori_A_adj; % 重构结果由负变正
        %[ori_A_bin, triangles_pred] = truncation(ori_A_adj, P3_tensor);
        %[ori_A_bin, triangles_pred] = truncation_bayes(ori_A_adj, P3_tensor, A1, A2);
        %[ACC, F1, ACC_tri, F1_tri] = EvaluationIndicators_Cal2(A1, A2, ori_A_bin, triangles_pred);
        [ACC, F1, ACC_tri, F1_tri] = EvaluationIndicators_Cal4(A1, A2, ori_A_adj, P3_tensor);
        ACC_two_body(k, :) = [numbers, ACC]; ACC_three_body(k, :) = [numbers, ACC_tri];
        F1_two_body(k, :) = [numbers, F1]; F1_three_body(k, :) = [numbers, F1_tri];
        k = k+1;
    end
end
ACC_two_body = sortrows(ACC_two_body, 1);
ACC_three_body = sortrows(ACC_three_body, 1);
F1_two_body = sortrows(F1_two_body, 1);
F1_three_body = sortrows(F1_three_body, 1);

%% Figures
color_two_body = [15/255, 52/255, 255/255]; % 自定义二体：蓝色
color_three_body = [255/255, 80/255, 80/255]; % 自定义三体：红色

% ACC
figure;
plot(Timespan, ACC_two_body(:,2), 'o-', 'LineWidth', 1.5, 'MarkerFaceColor', color_two_body, 'Color', color_two_body);
hold on;
plot(Timespan, ACC_three_body(:,2), '^-', 'LineWidth', 1.5, 'MarkerFaceColor', color_three_body, 'Color', color_three_body);

% 添加图例和图表标题，使用 'Location' 选项自动放置图例，避免遮挡数据点
legend('two-body', 'three-body', 'Location', 'best', 'FontSize', 14);
xlabel('T', 'FontSize', 20);
ylabel('Accuracy', 'FontSize', 20);

% 设置坐标轴范围，使其比数据的最大值大一点
xlim([0 max(Timespan) + 2000]); % 横坐标留出更多空间
ylim([-0.05 1.05]); % 纵坐标留出更多空间

% 设置纵坐标的刻度间隔为0.1
yticks(0:0.1:1);

% 设置坐标轴上刻度标签的字号
set(gca, 'FontSize', 14);


% F1
figure;
plot(Timespan, F1_two_body(:,2), 'o-', 'LineWidth', 1.5, 'MarkerFaceColor', color_two_body, 'Color', color_two_body);
hold on;
plot(Timespan, F1_three_body(:,2), '^-', 'LineWidth', 1.5, 'MarkerFaceColor', color_three_body, 'Color', color_three_body);

% 添加图例和图表标题，使用 'Location' 选项自动放置图例，避免遮挡数据点
legend('two-body', 'three-body', 'Location', 'best', 'FontSize', 14);
xlabel('T', 'FontSize', 20);
ylabel('F1 score', 'FontSize', 20);

% 设置坐标轴范围，使其比数据的最大值大一点
xlim([0 max(Timespan) + 2000]); % 横坐标留出更多空间
ylim([-0.05 1.05]); % 纵坐标留出更多空间

% 设置纵坐标的刻度间隔为0.1
yticks(0:0.1:1);

% 设置坐标轴上刻度标签的字号
set(gca, 'FontSize', 14);
