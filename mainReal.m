%%% 真实网络重建
%%% v1: 信息层为添加二阶边的真实网络，物理层为构造的ER网络(Avgk=8)

%% construct the network
clear, clc

pathname = '.\RealData\';
load(strcat(pathname, 'Thiers12_w.mat'));
load(strcat(pathname, 'Thiers12_triangles.mat'));


% Get adjacency matrices for both networks
A1 = full(w);  % 网络邻接矩阵保存在名为w的稀疏矩阵中，且为对称矩阵
A2 = triangles;  % 三元组，保存网络二阶边信息

nNodes = size(A1, 1);
numTriangles = size(A2, 1);
Prob_A = 8 / (nNodes - 1);
network = erdos_renyi(nNodes, Prob_A);
B = adjacency(network);

%Virtual a
lambda1 = 0.1;  % Lambda: probability of informed between two-body
lambda2 = 0.9;  % Lambda_Delta: probability of informed between three-body
delta = 0.8;    % Oblivion rate
%Physical b
beta1 = 0.2;    % Beta_U: probability of infection in the U-state
beta2 = 0.05;   % Beta_A: probability of infection in the A-state
mu = 0.8;       % Recovery rate

rhoa = 0.2;     % initial density of A
rhob = 0.25;    % initial density of I

Timespan = [1000, 2000, 4000, 7000, 10000, 15000, 20000, 30000, 40000, 50000];

for T = 1:length(Timespan)
    [UAU_state_nodes,SIS_state_nodes]=UAU_SIS_state(A1,A2,lambda1,lambda2,delta,B,beta1,beta2,mu,Timespan(T),rhoa,rhob);
    filename = strcat('Thiers12', '_m', num2str(Timespan(T)), '_n', num2str(nNodes), '_tri', num2str(numTriangles));
    disp(filename)
    save(strcat(pathname, filename), 'A1', 'A2', 'B', 'UAU_state_nodes', 'SIS_state_nodes');
end


%% Reconstruction Network
Lambda = 1e-2;  % lasso parameter
files = dir(fullfile(pathname, '*.mat'));
fileNames = {files.name};

for i = 1:length(fileNames)
    filename = fileNames{i};
    result = strsplit(filename, '_');

    if startsWith(filename, 'hypertext') && (~ismember(strcat('TSL', num2str(Lambda), '_', filename), fileNames)) && length(result)==4
        disp(filename)
        load(strcat(pathname, filename));
        [ori_A_adj, P3_tensor] = Reconstruction_TS(UAU_state_nodes, SIS_state_nodes, Lambda);
        save(strcat(pathname, 'TSL', num2str(Lambda), '_', filename), 'A1', 'A2', 'B', 'ori_A_adj', 'P3_tensor');
    end
end


%% Evaluation Indicators Calculation
F1_mat = zeros(2,length(Timespan));
files = dir(fullfile(pathname, '*.mat'));
fileNames = {files.name};
for i = 1:length(fileNames)
    filename = fileNames{i};
    result = strsplit(filename, '_');
    if startsWith(filename, 'TSL0.01') && isequal(result{2}, 'hypertext2009')
        disp(filename)
        
        result = strsplit(filename, '_');
        % Extraction networks parameters
        numbers = regexp(result{3}, '\d+', 'match');
        m = str2double(numbers);

        load(strcat(pathname, filename));
        P3_tensor = -P3_tensor; % 重构结果由负变正
        ori_A_adj = -ori_A_adj; % 重构结果由负变正
        [ori_A_bin, triangles_pred] = truncation(ori_A_adj, P3_tensor);
        [ACC, F1, ACC_tri, F1_tri] = EvaluationIndicators_Cal2(A1, A2, ori_A_bin, triangles_pred);
        F1_mat(:, find(Timespan==m)) = [F1, F1_tri];
    end
end


%% Figures
color_two_body   = [15/255, 52/255, 255/255]; % 自定义二体：蓝色
color_three_body = [255/255, 80/255, 80/255]; % 自定义三体：红色

% F1
figure;
plot(Timespan, F1_mat(1,:), 'o-', 'LineWidth', 1.5, 'MarkerFaceColor', color_two_body, 'Color', color_two_body);
hold on;
plot(Timespan, F1_mat(2,:), '^-', 'LineWidth', 1.5, 'MarkerFaceColor', color_three_body, 'Color', color_three_body);

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


