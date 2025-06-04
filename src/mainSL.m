%%% 多次重复跑出最佳结果，smooth line plot

clear, clc
% Set a fixed random seed for reproducibility
rng(12);

networkType = 'BA';
numTriangles = 40;
% Parameters for the networks
nNodes = 50;  % Number of nodes
Lambda = 1e-5;  % lasso parameter
cycle = 10;  % Number of cycles

% ER
    Prob_A = 0.2;  % Connection probability for ER network
    Prob_B = 0.12;
% BA
    degree_A = 6;  % dA：Average degree, mLinks should be half of this
    degree_B = 6;  % dB：Average degree, mLinks should be half of this
% SW
    k_A = 3;       % Each node is connected to k nearest neighbors in ring topology
    k_B = 3; 
    beta = 0.10;   % Rewiring probability

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

Timespan = [1000, 2000, 4000, 7000, 10000, 15000, 20000, 30000, 40000, 50000, 60000,70000,80000,90000,100000];
%Timespan=[60000,70000,80000,90000,100000];
F1_two_body = zeros(cycle, length(Timespan));
F1_three_body = zeros(cycle, length(Timespan));

for i=1:cycle
    disp(i)
    if isequal(networkType, 'ER')
        network1 = erdos_renyi(nNodes, Prob_A);
        network2 = erdos_renyi(nNodes, Prob_B);
    elseif isequal(networkType, 'BA')
        network1 = barabasi_albert(nNodes, degree_A/2);
        network2 = barabasi_albert(nNodes, degree_B/2);
    elseif isequal(networkType, 'SW')
        network1 = watts_strogatz(nNodes, k_A, beta);
        network2 = watts_strogatz(nNodes, k_B, beta);
    end

    % Get adjacency matrices for both networks
    A1 = adjacency(network1);
    B = adjacency(network2);

    % Add and highlight second-order edges in the first layer
    [network1, A2] = addSecondOrderEdges(network1, numTriangles);

    % Node status generation
    parfor T = 1:length(Timespan)
        disp(Timespan(T))

        [UAU_state_nodes,SIS_state_nodes]=UAU_SIS_state(A1,A2,lambda1,lambda2,delta,B,beta1,beta2,mu,Timespan(T),rhoa,rhob);
        [ori_A_adj, P3_tensor] = Reconstruction_TS(UAU_state_nodes, SIS_state_nodes, Lambda);
        P3_tensor = -P3_tensor; % 重构结果由负变正
        ori_A_adj = -ori_A_adj; % 重构结果由负变正
        [ori_A_bin, triangles_pred] = truncation(ori_A_adj, P3_tensor);
        [ACC, F1, ACC_tri, F1_tri] = EvaluationIndicators_Cal2(A1, A2, ori_A_bin, triangles_pred);

        F1_two_body(i,T) = F1;
        F1_three_body(i,T) = F1_tri;
    end
end

F1_two_body = temp1;
F1_three_body = temp2;


% 定义表头
headers = arrayfun(@(x) ['m', num2str(x)], Timespan, 'UniformOutput', false);
% 将矩阵转换为表格对象，并指定表头
T_two = array2table(F1_two_body, 'VariableNames', headers);
T_three = array2table(F1_three_body, 'VariableNames',headers);

% 写入 Excel 文件
writetable(T_two, '.\xlsxData\BA_F1two_SL.xlsx');
writetable(T_three, '.\xlsxData\BA_F1three_SL.xlsx');


%% Figures
color_two_body = [15/255, 52/255, 255/255]; % 自定义二体：蓝色
color_three_body = [255/255, 80/255, 80/255]; % 自定义三体：红色

% F1
figure;
plot(Timespan, max(F1_two_body), 'o-', 'LineWidth', 1.5, 'MarkerFaceColor', color_two_body, 'Color', color_two_body);
hold on;
plot(Timespan, max(F1_three_body), '^-', 'LineWidth', 1.5, 'MarkerFaceColor', color_three_body, 'Color', color_three_body);

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