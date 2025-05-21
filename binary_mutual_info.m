function miList = binary_mutual_info(UAU_state_nodes, nod)
  % 利用互信息，求解和当前节点nod相关的节点集
  % 确保输入为列向量
  %UAU_state_nodes = gpuArray(UAU_state_nodes);
  [~, N] = size(UAU_state_nodes);

  y = UAU_state_nodes(:, nod);
  miList = zeros(1, N);
  for i = 1:N
      if i == nod
          continue
      end
      x = UAU_state_nodes(:, i);
      
      % 定义分箱边界（明确区分0和1）
      edges = [-0.5, 0.5, 1.5];
      
      % 计算联合概率分布
      joint_counts = histcounts2(x, y, edges, edges);
      joint_probs = joint_counts / numel(x);
      
      % 计算边际概率
      p_x = histcounts(x, edges) / numel(x);
      p_y = histcounts(y, edges) / numel(y);
      
      % 计算互信息
      mi = 0;
      for xi = 0:1
          for yi = 0:1
              p_xy = joint_probs(xi+1, yi+1);
              px_py = p_x(xi+1) * p_y(yi+1);
              if p_xy > 0 && px_py > 0
                  mi = mi + p_xy * log2(p_xy / px_py);
              end
          end
      end

      miList(i) = mi;
  end

  triplet_scores = zeros(N, N); % 存储得分（针对节点nod）

  % 遍历所有候选对(i,k)
  for i = 1:N
    if i == nod
      continue
    end

    for j = i:N
      if j == nod
        continue
      end
      
      % 提取i和k在t-1时刻的协同感染状态
      X_ij = UAU_state_nodes(1:end-1, i) & UAU_state_nodes(1:end-1, j); % t-1时刻i和k同时感染
      
      % 对目标节点nod计算协同效应
      % 计算条件概率
      p_j_given_ij = mean(UAU_state_nodes(2:end, nod) & X_ij); % P(j_t=1 | i_{t-1}=1且k_{t-1}=1)
      
      % 计算基线概率（i或k单独感染的较大值）
      p_j_given_i = mean(UAU_state_nodes(2:end, nod) & UAU_state_nodes(1:end-1, i));
      p_j_given_j = mean(UAU_state_nodes(2:end, nod) & UAU_state_nodes(1:end-1, j));
      baseline = max(p_j_given_i, p_j_given_j);
      
      % 协同效应得分：条件概率与基线的差值
      score = p_j_given_ij - baseline;
      triplet_scores(i, j) = score;
    end
  end

end