function [X1_opt, X2_opt, history] = admm_bfgs(A1, A2, Y, max_iter, tol)
  % 参数初始化
  n = size(A1, 2);     % X1为n×1向量
  X1 =  rand(n,1);    % 初始化X1
  X2 = X1 * X1';       % 初始X2满足约束
  X2_vec = reshape(X2.', [], 1);
  Lambda = zeros(n);   % 拉格朗日乘子矩阵

  rho = 20; % initial value of penalty parameter
  eps_primal = tol; % primal tolerance
  eps_dual = tol; % dual tolerance
  eps_f = tol; % function tolerance
  f_prev = 0.5 * norm(A1 * X1 + A2 * X2_vec - Y)^2;  % previous objective value

  tau = 2;
  mu = 10;
  adaptive_phase = true;

  history = zeros(max_iter,3);
  
  % ADMM主循环
  for iter = 1:max_iter
    % 1. L-BFGS算法 求解X1子问题
    %options = optimoptions('fminunc', 'Algorithm', 'quasi-newton', 'GradObj', 'on', 'Display', 'off', 'MaxIterations', 100);
    options = optimoptions('fminunc', 'Algorithm', 'quasi-newton', 'HessianApproximation', 'lbfgs', 'GradObj', 'on', 'Display', 'off', 'MaxIterations', 300, 'OptimalityTolerance', tol);
    X1_new = fminunc(@(x) compute_cost_grad(x, A1, A2, Y, X2, Lambda, rho), X1, options);
  
    % 2. 共轭梯度法 更新X2子问题
    Z = X1_new*X1_new';
    b = A1*X1_new - Y;
    rhs = -A2'*b + rho*reshape(Z.', [], 1) - reshape(Lambda.', [], 1);
    
    % 定义隐式矩阵乘法函数
    Afun = @(x) A2'*(A2*x) + rho*x;
    M = diag(diag(A2'*A2 + rho*eye(size(A2,2))));
    [X2_new_vec, ~] = pcg(Afun, rhs, tol, 300, M, [], X2_vec);  % 共轭梯度求解
    X2_new = reshape(X2_new_vec, [n, n]);
    X2_new = (X2_new + X2_new')/2;  % 强制对称


    % 3. Calculate residuals
    r = X2_new - Z;           % primal residual
    s = rho * (X2_new - X2);  % dual residual
    r_norm = norm(r,'fro');   % primal residual norms
    s_norm = norm(s,'fro');   % dual residual norms
    f_current = 0.5 * norm(A1 * X1_new + A2 * X2_new_vec - Y, 'fro')^2;  % current objective value
    history(iter,:) = [f_current, r_norm, s_norm];

    if r_norm < eps_primal && s_norm < eps_dual && abs(f_current - f_prev) < eps_f
      X1 = X1_new;  % update X1
      X2 = X2_new;  % update X2

      fprintf('Converged at iteration %d\n', iter);
      break;
    end

    % 4. Lambda-update
    Lambda = Lambda + rho * r;

    % 5. rho-update
    if adaptive_phase && (f_current < 1000  || (r_norm < 100 && s_norm < 100))
      adaptive_phase = false;  % 进入固定rho阶段
    end

    if adaptive_phase
      if r_norm > mu * s_norm
        rho_new = rho * tau; % increase rho
      elseif s_norm > mu * r_norm
        rho_new = rho / tau; % decrease rho
      else
        rho_new = rho;       % keep rho unchanged
      end
    end  

    % 6. update variables
    X1 = X1_new;
    X2 = X2_new;
    X2_vec = X2_new_vec;
    rho = rho_new;
    f_prev = f_current;
  end
  
  % 输出结果
  X1_opt = X1;
  X2_opt = X2;
end
  
function [f, grad] = compute_cost_grad(X1, A1, A2, Y, X2, Lambda, rho)
  X2_vec = reshape(X2.', [], 1);  
  X1X1T = X1 * X1';
  Z = X2 - X1X1T;

  % 计算目标函数值
  data_term = 0.5 * norm(A1*X1 + A2*X2_vec - Y)^2;
  trace_term = trace(Lambda' * Z);
  frob_term = 0.5 * rho * norm(Z, 'fro')^2;
  f = data_term + trace_term + frob_term; 
  
  % 计算梯度
  grad_data = A1' * (A1*X1 + A2*X2_vec - Y);
  grad_trace = -2 * Lambda * X1;
  grad_frob = -2 * rho * Z * X1;
  grad = grad_data + grad_trace + grad_frob;
end