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
