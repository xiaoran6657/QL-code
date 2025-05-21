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

