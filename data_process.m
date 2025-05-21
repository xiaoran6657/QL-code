% Generate the node status of the real network and save it(单纯复形)
clc, clear;

%dynamic parameters
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

pathname = '.\RealData\';
Timespan = [1000, 2000, 4000, 7000, 10000, 15000, 20000, 30000, 40000, 50000];

realnetworkname = 'KapfererTailorShop';
filepath = strcat(pathname, realnetworkname, '.edges');
% 读取 .edges 文件
edgesData = readtable(filepath, 'FileType', 'text');  % 使用 FileType 选项读取为文本文件

nNodes = max(max(edgesData{:, 2}), max(edgesData{:, 3}));  % 获取节点数
A1 = zeros(nNodes, nNodes);  % 初始化邻接矩阵
B = zeros(nNodes, nNodes);
for j = 1:height(edgesData)
    if edgesData{j,1}==1 || edgesData{j,1}==2
        A1(edgesData{j, 2}, edgesData{j, 3}) = 1;  % 生成邻接矩阵
        A1(edgesData{j, 3}, edgesData{j, 2}) = 1;  % 生成邻接矩阵
    else
        B(edgesData{j, 2}, edgesData{j, 3}) = 1;
        B(edgesData{j, 3}, edgesData{j, 2}) = 1;
    end
end
kA = sum(sum(A1))/nNodes;  % 计算平均度
kB = 1;  % 1or2
Prob_A2 = 2*kB/((nNodes-1)*(nNodes-2));  % Connection probability of three-body interaction in Virtual layer

% Add and highlight second-order edges in the first layer
A2 = [];
network1 = graph(A1);
while isempty(A2)
    [network1, A2] = addSecondOrderEdges(network1, Prob_A2);
end
A1 = adjacency(network1);

% Node status generation and save
parfor T = 1:length(Timespan)
    [UAU_state_nodes,SIS_state_nodes]=UAU_SIS_state(A1,A2,lambda1,lambda2,delta,B,beta1,beta2,mu,Timespan(T),rhoa,rhob);
    filename = strcat(realnetworkname, '_m', num2str(Timespan(T)), '_n', num2str(nNodes), '_kA', num2str(kA), '_kB', num2str(kB));
    disp(filename)
    fun_save(pathname, filename, A1, A2, B, UAU_state_nodes, SIS_state_nodes);
end


function fun_save(pathname, filename, A1, A2, B, UAU_state_nodes, SIS_state_nodes)
  save(strcat(pathname, filename,  '.mat'), 'A1', 'A2', 'B', 'UAU_state_nodes', 'SIS_state_nodes');
end