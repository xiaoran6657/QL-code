function [alpha_opt, S_opt, history] = solve_GA(A1, A2, Y, N, pop_size, max_gen)
  % 遗传算法参数
  crossover_prob = 0.8;    % 交叉概率
  mutation_prob = 0.1;     % 变异概率
  tournament_size = 3;     % 锦标赛选择的大小

  % 初始化种群（正确维度：pop_size × (N+1)）
  population = initialize_population(pop_size, N);
  history = zeros(max_gen, N+2);

  % 进化循环
  for gen = 1:max_gen
      % 计算适应度
      fitness = compute_fitness(population, A1, A2, Y, N);
      
      % 记录最佳个体
      [best_fitness, idx] = max(fitness);
      best_individual = population(idx, :);
      
      % 选择
      selected = tournament_selection(population, fitness, tournament_size);
      
      % 交叉
      offspring = crossover(selected, crossover_prob, N);
      
      % 变异
      offspring = mutate(offspring, mutation_prob, N);
      
      % 新一代种群 = 后代 + 精英保留
      population = [best_individual; offspring];
      population = population(1:pop_size, :); % 保持种群大小

      history(gen, :) = [best_fitness, best_individual];
  end

  % 提取最优解
  alpha_opt = best_individual(1);
  S_opt = best_individual(2:end) > 0.5; % 转换为二进制
  S_opt = double(S_opt)';
end

% 初始化种群（正确维度）
function population = initialize_population(pop_size, N)
  population = zeros(pop_size, N + 1);  % 关键：N+1列！
  for i = 1:pop_size
      alpha = rand(); % α ∈ [0,10]
      S = randi([0,1], 1, N);
      population(i, :) = [alpha, S];
  end
end

% 计算适应度（正确索引）
function fitness = compute_fitness(population, A1, A2, Y, N)
  pop_size = size(population, 1);
  fitness = zeros(pop_size, 1);
  for i = 1:pop_size
      alpha = population(i, 1);      % 第一列为α
      S = population(i, 2:end);      % 第2到N+1列为S
      residual = alpha * A1 * S' + A2 * (alpha^2 * kron(S,S)') - Y;
      fitness(i) = -0.5 * sum(norm(residual)^2);
  end
end

% 锦标赛选择
function selected = tournament_selection(population, fitness, tournament_size)
  pop_size = size(population, 1);
  selected = zeros(size(population));
  for i = 1:pop_size
      candidates = randperm(pop_size, tournament_size);
      [~, idx] = max(fitness(candidates));
      selected(i, :) = population(candidates(idx), :);
  end
end

% 交叉操作（正确处理α和S）
function offspring = crossover(parents, crossover_prob, N)
  pop_size = size(parents, 1);
  offspring = parents;

  combinations = nchoosek(1:pop_size, 2);
  for i = 1:length(combinations)
      if rand() < crossover_prob
        p1 = combinations(i,1);
        p2 = combinations(i,2);
        % 交叉α（模拟二进制交叉）
        alpha1 = parents(p1, 1);
        alpha2 = parents(p2, 1);
        beta = rand() * 2 - 1;
        offspring(p1, 1) = 0.5*((1+beta)*alpha1 + (1-beta)*alpha2);
        offspring(p2, 1) = 0.5*((1-beta)*alpha1 + (1+beta)*alpha2);
        
        % 交叉S（单点交叉）
        cross_point = randi([2, N+1]);
        temp = offspring(p1, cross_point:end);
        offspring(p1, cross_point:end) = offspring(p2, cross_point:end);
        offspring(p2, cross_point:end) = temp;
      end
  end
end

% 变异操作（分别处理α和S）
function offspring = mutate(offspring, mutation_prob, N)
  for i = 1:size(offspring, 1)
      % 变异α
      if rand() < mutation_prob
          offspring(i, 1) = offspring(i, 1) + randn() * 0.1;
      end
      % 变异S
      for j = 2:N+1
          if rand() < mutation_prob
              offspring(i, j) = 1 - offspring(i, j);
          end
      end
  end
end