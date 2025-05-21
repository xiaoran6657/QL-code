function [X1_opt, history] = ADMM_GPU2(A1, A2, Y, max_iter, tol)
    % 参数设置
    N = size(A1, 2);        % 问题维度
    lambda = 0.2;           % L1正则化系数
    rho = gpuArray(3);     % 标量也需转为GPU类型
    tau = 2;
    mu = 10;

    % 初始化变量（GPU数组）
    X1 = gpuArray(lasso(A1, Y, 'Lambda',lambda,'RelTol',10^-6,Options=statset('Display', 'off')));
    Z = kron(X1, X1);
    U = zeros(N^2, 1, 'gpuArray');             % 对偶变量
    
    A1 = gpuArray(A1);
    A2 = gpuArray(A2);
    A2T_gpu = A2';
    Y = gpuArray(Y);    
    
    history = zeros(max_iter, 4, 'gpuArray');  % GPU数组记录历史

    % ADMM主循环
    for iter = 1:max_iter
        Z_prev = Z;
        % 1. 更新X1：使用L-BFGS求解非线性子问题
        options = optimoptions('fminunc', 'Algorithm', 'quasi-newton', 'GradObj', 'on', 'HessianApproximation', 'lbfgs', 'Display', 'off', 'MaxIterations', 500, 'OptimalityTolerance', tol);
        X1_cpu = gather(X1);
        X1_cpu = fminunc(@(x) X1_objective(x, A1, A2, Y, Z, U, lambda, rho), X1_cpu, options);
        X1 = gpuArray(X1_cpu);
        X1_kron = kron(X1, X1);
        
        % 2. 更新Z：使用共轭梯度法求解线性方程组
        b = A2T_gpu * (Y - A1*X1) + rho*X1_kron - U;  % 右端项
        A = A2T_gpu*A2 + rho*eye(N^2, 'gpuArray');    % 系数矩阵
        %L = ichol(A);
        [Z, flag] = pcg(A, b, tol, 500, [], [], Z_prev);         % 共轭梯度法
        
        % 3. 更新对偶变量U
        U = U + Z - X1_kron;
        
        % 收敛性检查
        f_current = 0.5 * norm(A1*X1 + A2*X1_kron - Y)^2;
        primal_res = norm(Z - X1_kron, 'fro');
        dual_res = rho * norm(Z - Z_prev, 'fro');
        if primal_res < tol && dual_res < tol && f_current < 1e-3
            break;
        end

        history(iter, :) = [f_current, primal_res, dual_res, rho];
    end

    % 输出结果转回CPU（可选）
    X1_opt = gather(X1);
    history = gather(history(1:iter, :));
end

% 目标函数与梯度（用于L-BFGS更新X1）
function [f, grad] = X1_objective(X1, A1, A2, Y, Z, U, lambda, rho)
    f_current = A1*X1 + A2*Z - Y;
    dual_current = Z - kron(X1, X1) + U/rho;
    %dual_current = Z - kron(X1, X1) + U;
    % 计算目标函数值
    term1 = 0.5 * norm(f_current)^2;
    term2 = lambda * norm(X1, 1);
    term3 = 0.5*rho * norm(dual_current)^2;
    f = gather(term1 + term2 + term3);
    
    % 计算梯度（显式导数）
    grad_term1 = A1' * f_current;
    grad_term2 = lambda * sign(X1);
    kron_grad = kron(X1', eye(length(X1))) + kron(eye(length(X1)), X1');
    grad_term3 = -rho * kron_grad * dual_current;
    grad = gather(grad_term1 + grad_term2 + grad_term3);
end