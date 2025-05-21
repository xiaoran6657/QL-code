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
