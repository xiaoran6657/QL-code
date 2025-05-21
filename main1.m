%% 第一步：生成网络和节点状态

clear, clc
% Set a fixed random seed for reproducibility
rng(12);

pathname = '.\matData\';  % 单纯复形
%pathname = '.\matData2\';  % 超图
networkType = 'BA';
% Parameters for the networks
nNodes_list = [100, 150];  % Number of nodes
K1s = [10, 12, 14, 16, 18];  % Average degree of two-body interaction
K2s = [2, 3, 4, 5, 6];    % Average degree of three-body interaction

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

Timespan = [1000, 2000, 4000, 7000, 10000, 15000, 20000, 30000, 40000, 50000];
for nNodes=nNodes_list
    for k1=K1s
        for k2 = K2s
            if (k1-2*k2)<=0  % avoid Prob_A1<=0 and mLinks<=0
                continue
            end
            Prob_A1 = (k1-2*k2)/(nNodes-1-2*k2);  % Connection probability of two-body interaction in Virtual layer
            Prob_A2 = 2*k2/((nNodes-1)*(nNodes-2));  % Connection probability of three-body interaction in Virtual layer
            Prob_B = 0.12;
            mLinks = ceil(nNodes*(k1-2*k2)/(2*nNodes-4*k2));  % the mLinks edges each new node connects to the old nodes with degree preference
    
            % Generate two networks
            if isequal(networkType, 'ER')
                network1 = erdos_renyi(nNodes, Prob_A1);
                network2 = erdos_renyi(nNodes, Prob_A1);
            elseif isequal(networkType, 'BA')
                network1 = barabasi_albert(nNodes, mLinks);
                network2 = barabasi_albert(nNodes, mLinks);
            elseif isequal(networkType, 'SW')
                network1 = watts_strogatz(nNodes, 2*mLinks, Prob_A1);
                network2 = watts_strogatz(nNodes, 2*mLinks, Prob_A1);
            end

            % Add and highlight second-order edges in the first layer
            A2 = [];
            while isempty(A2)
                [network1, A2] = addSecondOrderEdges(network1, Prob_A2);
            end

            % Get adjacency matrices for both networks
            A1 = adjacency(network1);
            B = adjacency(network2);
            
            % Node status generation and save
            parfor T = 1:length(Timespan)
                [UAU_state_nodes,SIS_state_nodes]=UAU_SIS_state(A1,A2,lambda1,lambda2,delta,B,beta1,beta2,mu,Timespan(T),rhoa,rhob);
                filename = strcat(networkType, '_m', num2str(Timespan(T)), '_n', num2str(nNodes), '_kA', num2str(k1), '_kB', num2str(k2));
                disp(filename)
                fun_save(pathname, filename, A1, A2, B, UAU_state_nodes, SIS_state_nodes);
            end
        end
    end
end

function fun_save(pathname, filename, A1, A2, B, UAU_state_nodes, SIS_state_nodes)
    save(strcat(pathname, filename), 'A1', 'A2', 'B', 'UAU_state_nodes', 'SIS_state_nodes');
end

%if isequal(networkType, 'ER')
%    filename = strcat(networkType, '_m', num2str(T), '_n', num2str(n), '_pA0', num2str(Prob_A*100),'_pB0', num2str(Prob_B*100), '_tri', num2str(numTriangles));
%    %filename = fullfile(pathname, sprintf('ER_m%d_n%d.mat',T,n));
%elseif isequal(networkType, 'BA')
%    filename = strcat(networkType, '_m', num2str(T), '_n', num2str(n), '_dA', num2str(degree_A),'_dB', num2str(degree_B), '_tri', num2str(numTriangles));
%    %filename = fullfile(pathname, sprintf('BA_m%d_n%d.mat',T,n));
%elseif isequal(networkType, 'SW')
%    filename = strcat(networkType, '_m', num2str(T), '_n', num2str(n), '_kA', num2str(k_A),'_kB', num2str(k_B),'_b0', num2str(beta*100), '_tri', num2str(numTriangles));
%end
%filename = strcat(networkType, '_m', num2str()%d_n%d.mat',T,n)
%disp(filename)
%save(strcat(pathname, filename), 'A1', 'A2', 'B', 'UAU_state_nodes', 'SIS_state_nodes');