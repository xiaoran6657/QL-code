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