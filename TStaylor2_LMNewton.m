function [X1_opt, X2_opt] = TStaylor2_LMNewton(A, A_Delta, Y, N, max_iter, tol)
    % MATLAB 代码：混合 LM-牛顿法 + L1 正则化 + GPU 加速
    % 需要 Parallel Computing Toolbox 和 Optimization Toolbox

    % 1. 数据准备与 GPU 加载
    Nchoose2 = nchoosek(N,2); % X2 维度 C×1

    A1_gpu = gpuArray(A(:, 1:N)); % 示例数据
    A2_gpu = gpuArray(A(:, N+1:N+Nchoose2));
    A3_gpu = gpuArray(A(:, N+Nchoose2+1:N+Nchoose2+N^2));
    A4_gpu = gpuArray(A(:, N+Nchoose2+N^2+1:N+Nchoose2+N^2+N*Nchoose2));
    A5_gpu = gpuArray(A(:, N+Nchoose2+N^2+N*Nchoose2+1:N+Nchoose2+N^2+N*Nchoose2+Nchoose2^2));
    Y1_gpu = gpuArray(Y(1:N, 1));

    A1Delta_gpu = gpuArray(A_Delta(:, 1:N)); % 示例数据
    A2Delta_gpu = gpuArray(A_Delta(:, N+1:N+Nchoose2));
    A3Delta_gpu = gpuArray(A_Delta(:, N+Nchoose2+1:N+Nchoose2+N^2));
    A4Delta_gpu = gpuArray(A_Delta(:, N+Nchoose2+N^2+1:N+Nchoose2+N^2+N*Nchoose2));
    A5Delta_gpu = gpuArray(A_Delta(:, N+Nchoose2+N^2+N*Nchoose2+1:N+Nchoose2+N^2+N*Nchoose2+Nchoose2^2));
    Y2_gpu = gpuArray(Y(N+1:N+Nchoose2, 1));

    % 2. 定义残差函数与正则化
    residual = @(theta) compute_residual(theta, A1_gpu, A2_gpu, A3_gpu, A4_gpu, A5_gpu, A1Delta_gpu, A2Delta_gpu, A3Delta_gpu, A4Delta_gpu, A5Delta_gpu, Y1_gpu, Y2_gpu);

    lambda1 = 0.1; % L1 正则化系数
    lambda2 = 0.01; % L2 正则化系数

    % 3. 混合 LM-牛顿法优化
    theta0 = rand(N + Nchoose2, 1, 'gpuArray'); % 初始值（GPU）
    options = optimoptions('lsqnonlin', ...
        'Algorithm', 'levenberg-marquardt', ...
        'MaxIterations', 1000, ...
        'MaxFunctionEvaluations', 5000, ...
        'UseParallel', true); % 启用 GPU 加速 [[3]][[8]]

    % 添加 L1 正则化（通过软阈值操作）
    theta_opt = lsqnonlin(residual, theta0, [], [], options);
    %theta_opt = soft_threshold(theta_opt, lambda1); % L1 正则化 [[6]][[7]]

    % 4. 结果提取与验证
    X1_opt = gather(theta_opt(1:N)); % GPU 转 CPU
    X2_opt = gather(theta_opt(N+1:end));
end

% 辅助函数：残差计算
function F = compute_residual(theta, A1_gpu, A2_gpu, A3_gpu, A4_gpu, A5_gpu, A1Delta_gpu, A2Delta_gpu, A3Delta_gpu, A4Delta_gpu, A5Delta_gpu, Y1_gpu, Y2_gpu)
    N = size(A1_gpu, 2);
    X1 = gpuArray(theta(1:N));
    X2 = gpuArray(theta(N+1:end));
    
    % 方程1残差
    F1 = A1_gpu*X1 + A2_gpu*X2 + A3_gpu*kron(X1,X1) + 2*A4_gpu*kron(X1,X2) + A5_gpu*kron(X2,X2) - Y1_gpu;
    % 方程2残差（带Δ的项）
    F2 = A1Delta_gpu*X1 + A2Delta_gpu*X2 + A3Delta_gpu*kron(X1,X1) + 2*A4Delta_gpu*kron(X1,X2) + A5Delta_gpu*kron(X2,X2) - Y2_gpu;
    
    F = [F1; F2];
end

% 辅助函数：软阈值操作（L1 正则化）
function theta = soft_threshold(theta, lambda1)
    theta(abs(theta) <= lambda1) = 0;
    theta(theta > lambda1) = theta(theta > lambda1) - lambda1;
    theta(theta < -lambda1) = theta(theta < -lambda1) + lambda1;
end