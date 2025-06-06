
Timespan_list = 1:20;
Timespan_list = Timespan_list*10000;

pathname2 = '.\redata\';

filename = "ERm200000n100ka18kb5.mat";
load(strcat(pathname2, "TST12_",filename));

parpool(20)

evalIndicators = zeros(size(recon_state,2), 5);

for i = 1:size(recon_state,2)
    fprintf("m%d:\n", recon_state(i).Timespan);
    [ACC, F1, ACC_tri, F1_tri] = EvaluationIndicators_Cal4(A1, A2, recon_state(i).ori_A_adj, recon_state(i).P3_tensor, "PR");
    evalIndicators(i, :) = [recon_state(i).Timespan, ACC, F1, ACC_tri, F1_tri];
end

%% Figures
color_two_body = [15/255, 52/255, 255/255]; % 自定义二体：蓝色
color_three_body = [255/255, 80/255, 80/255]; % 自定义三体：红色

% ACC
figure;
plot(evalIndicators(:,1), evalIndicators(:,2), 'o-', 'LineWidth', 1.5, 'MarkerFaceColor', color_two_body, 'Color', color_two_body);
hold on;
plot(evalIndicators(:,1), evalIndicators(:,4), '^-', 'LineWidth', 1.5, 'MarkerFaceColor', color_three_body, 'Color', color_three_body);

% 添加图例和图表标题，使用 'Location' 选项自动放置图例，避免遮挡数据点
legend('two-body', 'three-body', 'Location', 'best', 'FontSize', 14);
xlabel('T', 'FontSize', 20);
ylabel('Accuracy', 'FontSize', 20);

% 设置坐标轴范围，使其比数据的最大值大一点
xlim([0 max(evalIndicators(:,1)) + 1000]); % 横坐标留出更多空间
ylim([-0.05 1.05]); % 纵坐标留出更多空间

% 设置纵坐标的刻度间隔为0.1
yticks(0:0.1:1);

% 设置坐标轴上刻度标签的字号
set(gca, 'FontSize', 14);


% F1
figure;
plot(evalIndicators(:,1), evalIndicators(:,3), 'o-', 'LineWidth', 1.5, 'MarkerFaceColor', color_two_body, 'Color', color_two_body);
hold on;
plot(evalIndicators(:,1), evalIndicators(:,5), '^-', 'LineWidth', 1.5, 'MarkerFaceColor', color_three_body, 'Color', color_three_body);

% 添加图例和图表标题，使用 'Location' 选项自动放置图例，避免遮挡数据点
legend('two-body', 'three-body', 'Location', 'best', 'FontSize', 14);
xlabel('T', 'FontSize', 20);
ylabel('F1 score', 'FontSize', 20);

% 设置坐标轴范围，使其比数据的最大值大一点
xlim([0 max(evalIndicators(:,1)) + 1000]); % 横坐标留出更多空间
ylim([-0.05 1.05]); % 纵坐标留出更多空间

% 设置纵坐标的刻度间隔为0.1
yticks(0:0.1:1);

% 设置坐标轴上刻度标签的字号
set(gca, 'FontSize', 14);
