function [X1_opt, X2_opt, history] = ADMM_GPU(A1, A2, Y, max_iter, tol)
  A1 = gpuArray(A1);
  A2 = gpuArray(A2);
  A2T_gpu = A2';
  Y = gpuArray(Y);
  
  % 参数初始化并转移到GPU
  n = size(A1, 2);
  X1 = gpuArray.rand(n, 1);       % 直接生成GPU随机数
  X2 = X1 * X1';
  X2_vec = reshape(X2.', [], 1);  % GPU自动支持
  Lambda = gpuArray.zeros(n);     % GPU初始化

  rho = gpuArray(10);             % 标量也需转为GPU类型
  eps_primal = tol;
  eps_dual = tol;
  eps_f = tol;
  f_prev = 0.5 * norm(A1*X1 + A2*X2_vec - Y)^2;

  tau = 2;
  mu = 10;

  history = zeros(max_iter, 4, 'gpuArray');  % GPU数组记录历史

  % ADMM主循环
  for iter = 1:max_iter
    % 1. L-BFGS求解X1子问题（需将数据传回CPU，因fminunc不支持GPU）
    X1_cpu = gather(X1);  % 临时转回CPU
    options = optimoptions('fminunc', 'Algorithm', 'quasi-newton', 'HessianApproximation', 'lbfgs', 'GradObj', 'on', 'Display', 'off', 'MaxIterations', 500, 'OptimalityTolerance', tol);
    X1_new_cpu = fminunc(@(x) compute_cost_grad_gpu(x, A1, A2, Y, X2, Lambda, rho), X1_cpu, options);
    X1_new = gpuArray(X1_new_cpu);  % 结果转回GPU

    % 2. 共轭梯度法（GPU加速）
    Z = X1_new * X1_new';
    b = A1 * X1_new - Y;
    rhs = -A2T_gpu * b + rho * reshape(Z.', [], 1) - reshape(Lambda.', [], 1);
    
    Afun = @(x) A2T_gpu * (A2 * x) + rho * x;
    M_gpu = diag(diag(A2T_gpu*A2 + rho*eye(size(A2,2), 'like', gpuArray(1))));
    [X2_new_vec, ~] = pcg(Afun, rhs, tol, 500, M_gpu, [], X2_vec); % MATLAB自动支持GPU输入
    X2_new = reshape(X2_new_vec, [n, n]);
    X2_new = (X2_new + X2_new') / 2;

    % 3. 计算残差和目标函数
    r = X2_new - Z;
    s = rho * (X2_new - X2);
    r_norm = gather(norm(r, 'fro'));  % 取回CPU进行判断
    s_norm = gather(norm(s, 'fro'));
    f_current = 0.5 * norm(A1*X1_new + A2*X2_new_vec - Y, 'fro')^2;
    history(iter, :) = [f_current, r_norm, s_norm, rho];

    if r_norm < eps_primal && s_norm < eps_dual && abs(gather(f_current - f_prev)) < eps_f
      X1 = X1_new;
      X2 = X2_new;
      fprintf('Converged at iteration %d\n', iter);
      break;
    end

    % 4. 更新Lambda和rho
    Lambda = Lambda + rho * r;

    %if gather(f_current / mean(history(1:3, 1))) > 1e-2
    if iter < max_iter*0.4
      if r_norm > mu * s_norm
        rho_new = rho * tau;
      elseif s_norm > mu * r_norm
        rho_new = rho / tau;
      else
        rho_new = rho;
      end
      
      rho = rho_new;
    else
      rho = gpuArray(2);
    end

    % 5. 更新变量
    X1 = X1_new;
    X2 = X2_new;
    X2_vec = X2_new_vec;
    f_prev = f_current;
  end

  % 输出结果转回CPU（可选）
  X1_opt = gather(X1);
  X2_opt = gather(X2);
  history = gather(history(1:iter, :));
end

function [f, grad] = compute_cost_grad_gpu(X1_cpu, A1, A2, Y, X2_gpu, Lambda_gpu, rho_gpu)
  % 将输入数据转移到GPU
  X1_gpu = gpuArray(X1_cpu);

  X2_vec = reshape(X2_gpu.', [], 1);
  Z = X2_gpu - X1_gpu * X1_gpu';
  f_current = A1*X1_gpu + A2*X2_vec - Y;

  data_term = 0.5 * norm(f_current)^2;
  trace_term = trace(Lambda_gpu' * Z);
  frob_term = 0.5 * rho_gpu * norm(Z, 'fro')^2;
  f = gather(data_term + trace_term + frob_term);  % f需要返回给CPU的fminunc

  grad_data = A1' * f_current;
  grad_trace = -2 * Lambda_gpu * X1_gpu;
  grad_frob = -2 * rho_gpu * Z * X1_gpu;
  grad = gather(grad_data + grad_trace + grad_frob);  % 梯度返回给CPU
end