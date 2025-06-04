%% 第一步：生成网络和节点状态

clear, clc
% Set a fixed random seed for reproducibility
rng(12);

pathname = '..\data\';  % 单纯复形
% Parameters for the networks
nNodes_list = [100];  % Number of nodes
K1s = [16];  % Average degree of two-body interaction
K2s = [6];    % Average degree of three-body interaction

networkType = 'ER';
Timespan = 400000;

%dynamics parameters
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

for nNodes=nNodes_list
    for k1=K1s
        for k2 = K2s
            if (k1-2*k2)<=0  % avoid Prob_A1<=0 and mLinks<=0
                continue
            end
            filename = strcat(networkType, 'm', num2str(Timespan), 'n', num2str(nNodes), 'ka', num2str(k1), 'kb', num2str(k2));
            disp(filename)

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

            % Get adjacency matrices for both networks
            A1 = adjacency(network1);
            B = adjacency(network2);

            % Add and highlight second-order edges in the first layer
            A2 = [];
            while isempty(A2)
                [network1, A2] = addSecondOrderEdges(network1, Prob_A2);
            end
            
            
            % Node status generation and save
            [UAU_state_nodes,SIS_state_nodes]=UAU_SIS_state(A1,A2,lambda1,lambda2,delta,B,beta1,beta2,mu,Timespan,rhoa,rhob);
            fun_save(pathname, filename, A1, A2, B, UAU_state_nodes, SIS_state_nodes);
        
        end
    end
end

function fun_save(pathname, filename, A1, A2, B, UAU_state_nodes, SIS_state_nodes)
    save(strcat(pathname, filename), 'A1', 'A2', 'B', 'UAU_state_nodes', 'SIS_state_nodes');
end


%% Function to generate an Erdős-Rényi (ER) network
function G = erdos_renyi(nNodes, connectionProb)
    % Create an empty adjacency matrix
    adjMatrix = zeros(nNodes);

    % Fill the adjacency matrix
    for i = 1:nNodes
        for j = i+1:nNodes
            if rand < connectionProb
                adjMatrix(i, j) = 1;
                adjMatrix(j, i) = 1;
            end
        end
    end

    % Create graph from adjacency matrix
    G = graph(adjMatrix);
end


%% Function to generate a Barabási-Albert scale-free network
function G = barabasi_albert(nNodes, mLinks)
    % Initialize the adjacency matrix
    adjMatrix = zeros(nNodes);

    % Start with an initial connected network
    for i = 1:mLinks
        for j = i+1:mLinks
            adjMatrix(i, j) = 1;
            adjMatrix(j, i) = 1;
        end
    end

    % Add nodes and create links based on preferential attachment
    for i = mLinks+1:nNodes
        connected = sum(adjMatrix, 2);
        prob = connected / sum(connected);
        cumProb = cumsum(prob);
        for j = 1:mLinks
            r = rand;
            target = find(cumProb >= r, 1, 'first');
            adjMatrix(i, target) = 1;
            adjMatrix(target, i) = 1;
        end
    end

    % Create graph from adjacency matrix
    G = graph(adjMatrix);
end


%% Function to generate a Watts-Strogatz small world network
function G = watts_strogatz(nNodes, k, beta)
    % Create a ring of nodes
    s = repelem((1:nNodes)', 1, k);
    t = s + repmat(1:k, nNodes, 1);
    t = mod(t-1, nNodes) + 1;

    % Rewire the edges
    for i = 1:nNodes
        for j = 1:k
            if rand < beta
                u = t(i, j);
                v = randi(nNodes);
                while v == u || any(s(u,:) == v)
                    v = randi(nNodes);
                end
                t(i, j) = v;
            end
        end
    end

    G = graph(s, t, 1);      % 在 graph 函数中添加权重参数
    G = simplify(G); % Remove self-loops and duplicate edges
end


%% Function to add and highlight second-order edges (triangles)
function [G, triangles] = addSecondOrderEdges(G, Prob_A2)
    nodes = numnodes(G);
    triplets = nchoosek(1:nodes, 3);  % all possible triangles

    numTriangles = size(triplets, 1);
    randomNumbers = rand(numTriangles, 1);
    selectedIndices = randomNumbers < Prob_A2;
    triangles = triplets(selectedIndices, :);

    % According to triangles, add two-body interactions in G to satisfy the closure condition
    for i = 1:size(triangles,1)
        triangle = triangles(i, :);
        % Check if edges exist and add edges to form a triangle, with a weight of 1
        if ~findedge(G, triangle(1), triangle(2))
            G = addedge(G, triangle(1), triangle(2), 1);
        end
        if ~findedge(G, triangle(2), triangle(3))
            G = addedge(G, triangle(2), triangle(3), 1);
        end
        if ~findedge(G, triangle(3), triangle(1))
            G = addedge(G, triangle(3), triangle(1), 1);
        end
    end
end