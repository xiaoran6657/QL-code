function [ori_A_adj, P3_tensor] = Reconstruction_TStaylor7_GPU(UAU_state_nodes, SIS_state_nodes)
  % Reconstruction network by Two-step method with GPU acceleration, 
  %     using the second-order Taylor expansion to step two of the Two-step method
  % (https://doi.org/10.1038/s41467-022-30706-9|www.nature.com/naturecommunications)
  % Input:
      % UAU_state_nodes: the node state matrix of the virtual layer(UAU), [T, n]
      % SIS_state_nodes: the node state matrix of the physical layer(SIS), [T, n]
      % Lambda: the regularization parameter
  % Output:
      % ori_A_adj: the reconstructed network two-body interaction, [n, n]
      % P3_tensor: the reconstructed network three-body interaction, [n, n, n]
  tic;

  % Transfer input data to GPU if available
  if gpuDeviceCount > 0
    UAU_state_nodes = gpuArray(UAU_state_nodes);
    SIS_state_nodes = gpuArray(SIS_state_nodes);
  end

  [~, n]=size(UAU_state_nodes);  % 共m个时刻，n个节点
  ori_A_adj = zeros(n, n, 'gpuArray');
  P3_tensor = zeros(n, n, n, 'gpuArray');  % 创建一个n*n*n的三维张量，在确定阈值前需收集原始小数
  Lambda = 1e-3;  % lasso parameter
  % 初始化\alpha
  lambda = 0.1;       alpha = gpuArray(log(1-lambda));
  lambda_Delta = 0.9; alpha_Delta = gpuArray(log(1-lambda_Delta));
  
  options1 = optimoptions('fsolve', 'Display', 'none');
  options2 = statset('Display', 'off');

  % 循环求解所有节点的一阶边和二阶边
  for nod = 1:n
      fprintf("nod: %d \n", nod);
      tic; % 开始计时
      %%%step one --- Taylor1
      % solve 两个gamma（即向量x）
      [X, Y, A2]=Extract(UAU_state_nodes, SIS_state_nodes, nod);
      theta1=sum(A2,2);  % \={theta}^i(t_m), in Eq(4.22)

      x0 = 0.9999;
      x = fsolve(@(x) myfunTS(x,X,Y,theta1),x0,options1);   % solve TS.Eq(4.27), x=alpha^i
      
      % solve TS.Eq(4.37)
      M = x.^theta1;                                % instead a^(x_0) by TS.Eq(4.24)
      f = M./(1-M+eps) - M.*log(M)./((1-M+eps).^2); % \={W}^i(t_m), TS.Eq(4.31)
      g = M./((1-M+eps).^2);                        % \={V}^i(t_m), TS.Eq(4.31)
      
      C = zeros(n, n, 'gpuArray');                  % A_1, TS.Eq(4.37)
      D = zeros(n, 1, 'gpuArray');                  % Y_1, TS.Eq(4.37)
      
      % Calculate A_1 and Y_1
      for i = 1:n
          % lambda1, first n columns for j values
          C(i,1:n) = sum(Y.*A2(:,i).*g .* A2, 1);
          D(i) = sum((X-Y.*f).*A2(:,i));            % Y_1(i), TS.Eq(4.37)
      end
      clear theta1 f g;
      
      TS_X1 = lasso(gather(C), gather(D), 'Lambda', Lambda, 'RelTol', 1e-4, 'Options', options2);
      clear C D;
      
      TS_X1 = gpuArray(-TS_X1);
      tru = fun_cut(TS_X1);  % Find truncation value (on CPU)
      neig = find(TS_X1 >= tru);     % Get neighbor subset
      neig_subset = sort(union(nod, neig'));

      %%%step two --- Taylor2
      % 矩阵裁剪后X, Y并不改变，仅对A2进行裁切
      A2 = A2(:,union(nod, neig));
      theta1=sum(A2,2);  % \={theta}^i(t_m), in Eq(4.22)
      [m1,n2]=size(A2);   % m1:有效时刻数量，n2:TS2节点数量
      n2choose2 = nchoosek(n2,2); % X2维度

      theta2=gpuArray.zeros(m1,1,'double');
      A3 = gpuArray.zeros(m1,n2choose2,'double');       % 求A3，即[\={s}^j * \={s}^k]_{m1 \times C_N^2}
      for j=1:m1
        temp1=A2(j,:)'*A2(j,:);
        A3(j,:)=(temp1(logical(tril(ones(size(temp1)),-1))))';  % tril(*,-1)取严格下三角，从下三角矩阵到一行的转换通过列优先顺序实现
        theta2(j,1)=sum(sum(temp1-diag(diag(temp1))))/2;  % \={theta}^i_{Delta}(t_m), in Eq(4.23)
      end

      % 计算相关变量
      x0=[0.9999, 0.9999];
      x = fsolve(@(x) myfun(x,X,Y,theta1,theta2),x0, options1);  % solve Eq(4.29), x=[alpha^i, alpha^i_{Delta}]
      if min(x)<=0
          disp('fsolve alpha <= 0;\n')
          continue
      end

      M=x(1).^theta1.*x(2).^theta2;                         % instead a^(x_0)*b^(y_0) by Eq(4.24)
      Oi = M.*(1+M)./(2.*(1-M).^3);                         % \={O}^i(t_m), [m1,1], in TST2.Eq(4.31)
      Vi = M./((1-M).^2)-2.*Oi.*log(M);                     % \={V}^i(t_m), [m1,1], in TST2.Eq(4.31)
      Wi = M./(1-M)-M.*log(M)./((1-M).^2)+Oi.*(log(M).^2);  % \={W}^i(t_m), [m1,1], in TST2.Eq(4.31)
      clear M theta1 theta2;

      % 分块计算A6 G5 G5_Delta，避免内存限制
      G5 = gpuArray.zeros(n2, n2choose2^2, "single");       % n2 × n2*n2choose2
      G5_Delta = zeros(n2choose2, n2choose2^2, "single");   % n2choose2 × n2choose2^2

      block_length = 500;  % 每个块的大小，不是越大越好，过大的块会超出GPU专用内存大小，导致性能降低
      num_blocks = ceil(m1 / block_length);  % 计算块的数量
      
      % 一次性计算所有行的temp2矩阵（严格下三角元素乘积）
      mask = tril(true(n2, n2), -1);
      [k_indices, i_indices] = find(mask);
      temp2_matrix = A2(:, gpuArray(k_indices)) .* A2(:, gpuArray(i_indices)); % GPU数组，每行为temp2
      
      for k = 1:num_blocks  % 循环每个块
        start_index = (k-1) * block_length + 1;  % 块的起始索引
        end_index = min(k * block_length, m1);  % 块的结束索引
        current_block_size =  end_index - start_index + 1;  % 当前块的大小

        Y_part = Y(start_index:end_index, :);
        Oi_part = Oi(start_index:end_index, :);
        A6_part = gpuArray.zeros(current_block_size, n2choose2^2,'single');

        % 计算每行的Kronecker积
        for j = start_index:end_index
            temp2 = temp2_matrix(j, :); % 提取当前行的temp2
            % 使用矩阵乘法和reshape替代kron函数
            kron_product = reshape(temp2' * temp2, 1, []);
            A6_part(j-start_index+1, :) = single(kron_product); % 保持数据在GPU上
        end

        G5 = G5 + alpha_Delta^2*(Y_part .* A2(start_index:end_index, :))' * (Oi_part .* A6_part);          % n2 × n2choose2^2
        G5_Delta = G5_Delta + gather(alpha_Delta)^2*gather((Y_part.* A3(start_index:end_index, :))') * (gather(Oi_part) .* gather(A6_part));     % n2choose2 × n2choose2^2
      end
      clear Oi_part Y_part A6_part temp2 start_index end_index current_block_size mask k_indices i_indices kron_product;

      % 向量化计算A5 (kron(A2j, temp2))
      A5 = reshape(A2 .* permute(temp2_matrix, [1 3 2]), m1, n2 * n2choose2);
      clear temp2_matrix;
      % G上、D上，一阶边
      G4 = alpha*alpha_Delta*(Y .* A2)' * (Oi .* A5);          % n2 × n2*n2choose2
      G4_Delta = alpha*alpha_Delta*(Y .* A3)' * (Oi .* A5);    % n2choose2 × n2*n2choose2
      clear A5;

      % 向量化计算A4 (kron(A2j, A2j))
      A4 = reshape(A2 .* permute(A2, [1 3 2]), m1, n2^2);  % 避免显式循环
      G3 = alpha^2*(Y .* A2)' * (Oi .* A4);          % n2 × n2^2
      G3_Delta = alpha^2*(Y .* A3)' * (Oi .* A4);    % n2choose2 × n2^2
      clear A4;

      G1 = alpha*(Y .* A2)' * (Vi .* A2);          % n2 × n2
      G2 = alpha_Delta*(Y .* A2)' * (Vi .* A3);          % n2 × n2choose2
      G1_Delta = alpha*(Y .* A3)' * (Vi .* A2);    % n2choose2 × n2
      
      D1 = sum((X - Y .* Wi) .* A2, 1)';     % n2 × 1      
      clear A2;

      G2_Delta = alpha_Delta*(Y .* A3)' * (Vi .* A3);    % n2choose2 × n2choose2
      clear Vi;
      
      D2 = sum((X - Y .* Wi) .* A3, 1)';     % n2choose2 × n2choose2^2
      clear X Y Wi A3;
      
      % 混合LM-牛顿法优化
      %theta0 = lasso(gather([G1, G2;G1_Delta, G2_Delta]),gather([D1;D2]),'Lambda', Lambda,'RelTol',10^-4, Options=options2); % 初始值（GPU）
      
      G = [G1, G2; G1_Delta, G2_Delta];
      D = [D1; D2];
      M_poly = @(x) x - 0.5*G*x + 0.25*G^2*x;  % 最小二乘多项式

      [X0, flag] = gmres(G, D, 3, 1e-3, 20, M_poly);  % 使用GMRES求解
      clear G D; 
      
      % 解GX=D，得P0=[P1 P2]'=[eta1 eta2]'是一个列向量
      %residual = @(theta) compute_residual(theta,G1,G2,G3,G4,G5,G1_Delta,G2_Delta,G3_Delta,G4_Delta,G5_Delta,D1,D2);
      %options = optimoptions('lsqnonlin','Algorithm','levenberg-marquardt','MaxFunctionEvaluations',1e6,'MaxIterations',500,'FunctionTolerance',1e-3,  'StepTolerance',1e-3,'SpecifyObjectiveGradient',true, 'UseParallel',true);
      
      %P0 = lsqnonlin(residual, gather(X0), [], [], options);
      % 将非线性方程组求解问题转化为非线性最小二乘的优化问题，速度太慢
      X0 = -X0;
      S0 = X0 >= fun_cut(X0);     % Get neighbor subset
      P0 = GALMoptimizer_GPU(S0,G1,G2,G3,G4,G5,G1_Delta,G2_Delta,G3_Delta,G4_Delta,G5_Delta,D1,D2);

      P1=P0(1:n2);
      % ori_A_adj:重构的网络邻接矩阵，元素为小数，不一定对称
      P5 = zeros(n,1);
      P5(neig_subset) = P1;
      ori_A_adj(:,nod) = P5;

      % P2=eta2:该nod二阶边（C_n^2个元素组成的列向量）
      %P2=zeros(n*(n-1)/2,1);
      P2=P0((n2+1):n2*(n2+1)/2);
      % P3:将列向量P2转换为矩阵存到严格下三角阵P3里
      P3=zeros(n2,n2);
      temp2 = find(tril(ones(n2, n2), -1));  %获取P2的严格下三角部分的线性索引
      P3(temp2) = P2;  %将P2中的元素按照列优先顺序填充到P3的严格下三角部分
      P4 = zeros(n,n);
      P4(neig_subset, neig_subset) = P3;
      P3_tensor(:, :, nod) = P4;  %将计算出的P3存储到三维张量中的对应切片
      
      toc; % 结束计时并输出所用时间
  end
  toc;
end

%% LM优化器
function theta = GALMoptimizer_GPU(theta,G1,G2,G3,G4,G5,G1_Delta,G2_Delta,G3_Delta,G4_Delta,G5_Delta,Y1_gpu,Y2_gpu)
    nvars = sum(size(G2));        % 待求解变量维度（根据实际问题调整）
%{
  % ------------------- GA全局搜索 -------------------
  popSize = 50;                 % GA种群大小
  numElite = 5;                 % GA中保留的较优解数量

  % 定义适应度函数（残差平方和）
  fitnessFunc = @(x) fitnessComputer(x,G1,G2,G3,G4,G5,G1_Delta,G2_Delta,G3_Delta,G4_Delta,G5_Delta,Y1_gpu,Y2_gpu);

  % 配置GA参数
  optionsGA = optimoptions('ga', ...
    'PopulationType', 'bitstring', ...    % 01变量
    'PopulationSize', 10, ...             % GA种群大小
    'MaxGenerations', 3, ...              % 最大迭代次数
    'CrossoverFraction', 0.8, ...
    'MutationFcn', @mutationuniform, ...  % 均匀变异（可自定义）
    'Display', 'iter');

  % 运行GA
  [~, ~, ~, ~, population, scores] = ga(fitnessFunc, nvars, [], [], [], [], [], [], [], optionsGA);

  % 从GA种群中选择前numElite个较优解
  [sortedScores, idx] = sort(scores);
  eliteSolutions = population(idx(1:numElite), :);
  %theta = population(find(scores == min(scores), 1),:);
  %theta = gpuArray(theta(:));
  %}

  % ------------------- 最小二乘 -------------------
  % 定义参数传递
  fun = @(x) compute_residual(x,G1,G2,G3,G4,G5,G1_Delta,G2_Delta,G3_Delta,G4_Delta,G5_Delta,Y1_gpu,Y2_gpu);

  % 配置优化选项
  options_lsq = optimoptions('lsqnonlin', ...
    'Algorithm', 'trust-region-reflective', ...
    'SpecifyObjectiveGradient', true, ...  % 使用自定义雅可比
    'StepTolerance', 1e-4, ...                    % 放宽步长容差
    'FunctionTolerance', 1e-4, ...             % 放宽函数值容差
    'Display', 'iter', ...
    'MaxIterations', 100);

  % 变量边界（强制变量在[0,1]区间）
  lb = zeros(nvars, 1);
  ub = ones(nvars, 1);

  % 调用求解器
  %{
  bestResnorm = inf;
  for i = 1:size(eliteSolutions,1)
    [x_opt, resnorm] = lsqnonlin(fun, eliteSolutions(i,:), lb, ub, options_lsq);
    if resnorm < bestResnorm
        bestResnorm = resnorm;
        theta = x_opt;
        fprintf("最优解更新，resnorm: %.3e", resnorm);
    end
  end
  %}
  [theta, resnorm] = lsqnonlin(fun, gather(double(theta)), lb, ub, options_lsq);

end


%% 计算残差、函数值和雅可比矩阵
function [R, J]= compute_residual(theta,G1,G2,G3,G4,G5,G1_Delta,G2_Delta,G3_Delta,G4_Delta,G5_Delta,Y1_gpu,Y2_gpu)
  [n2, n2choose2]=size(G2);
  S1 = gpuArray(theta(1:n2));
  S2 = gpuArray(theta(n2+1:end));
  S1 = S1(:);  S2 = S2(:);  % 强制列向量

  % 残差计算
  kron_S1S1 = reshape(S1 * S1', 1, n2, n2);
  kron_S1S2 = reshape(S1 * S2', 1, n2, n2choose2);
  kron_S2S2 = reshape(S2 * S2', 1, n2choose2, n2choose2);

  term1 = sum(reshape(G3, size(G3,1), n2, n2) .* kron_S1S1, [2,3]);
  term2 = 2 * sum(reshape(G4, size(G4,1), n2, n2choose2) .* kron_S1S2, [2,3]);
  term3 = double(sum(reshape(G5, size(G5,1), n2choose2, n2choose2) .* kron_S2S2, [2,3]));

  term1D = sum(reshape(G3_Delta, size(G3_Delta,1), n2, n2) .* kron_S1S1, [2,3]);
  term2D = 2 * sum(reshape(G4_Delta, size(G4_Delta,1), n2, n2choose2) .* kron_S1S2, [2,3]);
  term3D = double(gpuArray(sum(reshape(G5_Delta, size(G5_Delta,1), n2choose2, n2choose2) .* gather(kron_S2S2), [2,3])));
  % 残差
  R = [G1*S1 + G2*S2 + term1 + term2 + term3 - Y1_gpu; G1_Delta*S1 + G2_Delta*S2 + term1D + term2D + term3D - Y2_gpu];
  R = gather(R);  % lsqnonlin不支持gpuArray
  % 添加松弛项（鼓励变量接近 0/1）
  %lambda = 0.1;  % 松弛系数
  %penalty = (lambda * (sum(S1.*(1-S1))) + lambda * (sum(S2.*(1-S2)))) / (n2 + n2choose2);
  %R = R + double(gather(penalty));

  % 雅可比计算
  I_N = eye(n2, 'gpuArray');
  I_C = eye(n2choose2, 'gpuArray');
  %I_C = eye(n2choose2, 'double');

  % 计算非线性项的导数
  D_S1 = kron(S1, I_N) + kron(I_N, S1);
  D_S2 = kron(S2, I_C) + kron(I_C, S2);
  %D_S2 = kron(gather(S2), I_C) + kron(I_C, gather(S2));
  
  % 雅可比矩阵
  J = [G1 + G3 * D_S1 + 2 * G4 * kron(I_N, S2), G2 + 2 * G4 * kron(S1, I_C) + double(G5 * D_S2); ...
      G1_Delta + G3_Delta * D_S1 + 2 * G4_Delta * kron(I_N, S2), G2_Delta + 2 * G4_Delta * kron(S1, I_C) + double(gpuArray(G5_Delta * gather(D_S2)))];
  J = gather(J);
end


%% 计算 GA 适应度函数
function error = fitnessComputer(theta,G1,G2,G3,G4,G5,G1_Delta,G2_Delta,G3_Delta,G4_Delta,G5_Delta,Y1_gpu,Y2_gpu)
  [n2, n2choose2]=size(G2);
  S1 = gpuArray(theta(1:n2));
  S2 = gpuArray(theta(n2+1:end));
  S1 = S1(:);  S2 = S2(:);

  % 转换方程组系数计算
  kron_S1S1 = reshape(S1 * S1', 1, n2, n2);
  kron_S1S2 = reshape(S1 * S2', 1, n2, n2choose2);
  kron_S2S2 = reshape(S2 * S2', 1, n2choose2, n2choose2);

  c1 = G1*S1; 
  c2 = G2*S2;
  c3 = sum(reshape(G3, size(G3,1), n2, n2) .* kron_S1S1, [2,3]);
  c4 = 2 * sum(reshape(G4, size(G4,1), n2, n2choose2) .* kron_S1S2, [2,3]);
  c5 = double(sum(reshape(G5, size(G5,1), n2choose2, n2choose2) .* kron_S2S2, [2,3]));

  d1 = G1_Delta*S1; 
  d2 = G2_Delta*S2;
  d3 = sum(reshape(G3_Delta, size(G3_Delta,1), n2, n2) .* kron_S1S1, [2,3]);
  d4 = 2 * sum(reshape(G4_Delta, size(G4_Delta,1), n2, n2choose2) .* kron_S1S2, [2,3]);
  d5 = double(gpuArray(sum(reshape(G5_Delta, size(G5_Delta,1), n2choose2, n2choose2) .* gather(kron_S2S2), [2,3])));

  F1 = c1 + c2 + c3 + c4 + c5 - Y1_gpu;
  F2 = d1 + d2 + d3 + d4 + d5 - Y2_gpu;
  error = sum(F1.^2) + sum(F2.^2);

  % ------------------- LM  算法 -------------------
  %{
  tolerence = 1e-3;
  max_iter = 100;
  error_list = gpuArray.zeros(max_iter,1);

  % 计算残差和雅可比矩阵
  F1 = alpha*c1 + alpha_Delta*c2 + alpha^2*c3 + alpha*alpha_Delta*c4 + alpha_Delta^2*c5 - Y1_gpu;
  F2 = alpha*d1 + alpha_Delta*d2 + alpha^2*d3 + alpha*alpha_Delta*d4 + alpha_Delta^2*d5 - Y2_gpu;
  J = [2*alpha*c3+alpha_Delta*c4+c1, alpha*c4+2*alpha_Delta*c5+c2;...
      2*alpha*d3+alpha_Delta*d4+d1, alpha*d4+2*alpha_Delta*d5+d2];
  current_error = sum(F1.^2) + sum(F2.^2);
  % 计算Hessian和梯度
  H = J' * J;
  grad = J' * [F1; F2];
  % 阈值归一化
  if norm(grad) > 1e-3
    grad = grad / (norm(grad) + 1e-12);
  end

  % 计算阻尼因子
  norm_H = H / (norm(H) + 1e-12);  % 添加归一化
  mu = 1e-4 * max(diag(norm_H));
  v = gpuArray(4);

  for iter = 1:max_iter
    % 添加最小正则化保证正定性
    H_reg = H + mu*eye(size(H));

    % 改进方案：Cholesky分解 + 条件数检查
    [HR, p] = chol(H_reg);
    if p == 0
      delta = -HR \ (HR' \ grad); % Cholesky分解求解
    else
      % 退化到伪逆
      %warning('矩阵不正定，使用SVD求解');
      [U, S, V] = svd(H_reg);
      s = diag(S);
      tol = max(size(H_reg)) * eps(max(s));  % 基于机器精度与维度
      s_inv = 1./s;
      s_inv(s < tol) = 0;  % 截断小奇异值

      % 处理全零情况（可选）
      if all(s_inv == 0)
        warning('所有奇异值被截断，退化为零方向');
        delta = zeros(size(grad));
      else
        delta = -V * (s_inv .* (U' * grad));
      end
    end
    
    % 更新参数
    alpha_new = alpha + delta(1);
    alpha_Delta_new = alpha_Delta + delta(2);
    % 计算残差和雅可比矩阵
    F1_new = alpha_new*c1 + alpha_Delta_new*c2 + alpha_new^2*c3 + alpha_new*alpha_Delta_new*c4 + alpha_Delta_new^2*c5 - Y1_gpu;
    F2_new = alpha_new*d1 + alpha_Delta_new*d2 + alpha_new^2*d3 + alpha_new*alpha_Delta_new*d4 + alpha_Delta_new^2*d5 - Y2_gpu;
    J_new = [2*alpha_new*c3+alpha_Delta_new*c4+c1, alpha_new*c4+2*alpha_Delta_new*c5+c2;...
      2*alpha_new*d3+alpha_Delta_new*d4+d1, alpha_new*d4+2*alpha_Delta_new*d5+d2];
    new_error = sum(F1_new.^2) + sum(F2_new.^2);

    Delta_L = -delta'*J_new'*[F1_new;F2_new] - 0.5*(delta'*J_new'*J_new*delta);
    rho = 2*(current_error - new_error) / Delta_L;
    if rho > 0
      if norm(J_new'*[F1_new;F2_new])^2 < tolerence
        break;
      end
      
      alpha = alpha_new;
      alpha_Delta = alpha_Delta_new;
      error_list(iter) = new_error;
      current_error = new_error;

      % 计算Hessian和梯度
      H = J_new' * J_new;
      grad = J_new' * [F1_new;F2_new];
      % 阈值归一化
      if norm(grad) > 1e-1
        grad = grad / (norm(grad) + 1e-12);
      end

      % 更新阻尼因子
      mu = max(1e-10, mu*max(1/4, 1-(2*rho-1)^3));
      %mu = mu*max(1/10, 1-(2*rho-1)^3);
      v = 4;

    else
      mu = mu*v;
      v = 4*v;
    end
    
    %fprintf('Iter %d: Error = %.6f, rho = %.3f, mu = %.3e, v = %d\n', iter, current_error, rho, mu, v);
  end
  %fprintf("\n------------------LM algorithm-----------------\n");
  error = min(error_list);  % 返回迭代最小值作为当前种群适应度

  %fprintf("Decrease : %.3e\n", (error_list(1) - error) / error_list(1))
  %}
end


%% Find the two-body truncation value
function c_cut = fun_cut(a)
    %Find the two-body truncation value
    %Input: Connection probability between reconstructed node and other nodes
    %Output: Get the truncation value in the vector
    
    aveave=mean(a(a<=mean(a))); 
    a_c=min(aveave, 1/length(a));  
    if a_c>0
        a_n=a(a>a_c);
        if isrow(a_n)  % 根据a_n的格式调整
            b=[-sort(-a_n),a_c];   %Descending order
        else
            b=[-sort(-a_n);a_c];   %Descending order
        end
        [~,id1]=max(b(1:end-1).*(b(1:end-1)-b(2:end))./b(2:end)); %Truncation method, the first
        bb=b;
        b(1:id1)=[];
        if isscalar(b)
            c_cut=bb(id1);  
        else
            [~,id1]=max(b(1:end-1).*(b(1:end-1)-b(2:end))./b(2:end)); %Truncation method, the second
            c_cut=b(id1+1);  
        end
    else
        c_cut=0.000001;
    end
end


%% Function myfun 用来求解两个alpha（即向量x） Eq(4.29)
function F=myfun(x,X,Y,theta1,theta2)
  F(1)=sum(Y.*theta1./(1-x(1).^theta1.*x(2).^theta2+eps)-(X+Y).*theta1);
  F(2)=sum(Y.*theta2./(1-x(1).^theta1.*x(2).^theta2+eps)-(X+Y).*theta2);
  F = gather(F);
end


%% Function myfunTS 两步法第一步中用来求解一个alpha（即向量x） TS.Eq(4.27)
function F=myfunTS(x,X,Y,theta1)
  F = gather(sum(Y.*theta1./(1-x.^theta1)-(X+Y).*theta1));
end


%% Function Extract
function [X,Y,A2]=Extract(SA,SB,nod)
  % Extract the special data of the node i from the node state matrix SA and SB
  % Input:
      % SA: the node state matrix of the virtual layer(UAU), UAU_state_nodes
      % SB: the node state matrix of the physical layer(SIS), SIS_state_nodes
      % nod: the node i
  % Output:
      % X: \={Q}^i(t_m), [m1,1], in Eq(4.8)
      % Y: \={R}^i(t_m), [m1,1], in Eq(4.8)
      % theta1: \={theta}^i(t_m), in Eq(4.22)
      % theta2: \={theta}^i_{Delta}(t_m), in Eq(4.23)
      % A2: Filtered node state matrix, \={s}^i(t), in Eq(4.38)
      % A3: \={s}^j * \={s}^k, in Eq(4.39)

  [m,n]=size(SA); % [T, n]
  A1=SA(:,nod);   % Extract Virtual layer's column i, [1,n]
  A2=SA;
  t=find(A1==0);  % 寻找A1中0元素的位置，将位置记录至t，如t=[2 3 8]
  t(t==m)=[];     % 只记录前m-1个时刻里，A1的0元素的位置，保证t+1时刻还有数据
  A2=A2(t,:);     % 这些有效时刻所有节点的状态
  A1=A1(t+1,:);   % 下一时刻nod虚拟层的状态

  B1=SB(:,nod);   % Extract Physical layer's column i    
  B1=B1(t+1,:);   % 下一时刻nod实际层的状态

  Y=A1.*(1-B1);   % \={Q}^i(t_m)表达式第一项通过t的筛选而导致必等于1==>
  if nnz(Y) == 0  % 如果Y全为0，即没有1，随机选取0.1%的位置赋值为1
    for i=1:ceil(length(Y)*0.001)
    Y(randi(length(Y))) = 1;
    end
  end
  X=1-A1;         % 1-\={s}^i(t_m)===1, in Eq(4.8)
  if nnz(X) == 0  % 如果X全为0，即没有1，随机选取0.1%的位置赋值为1
    for i=1:ceil(length(X)*0.001)
    X(randi(length(X))) = 1;
    end
  end
end
