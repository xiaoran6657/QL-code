%% LM优化器
function theta = LMoptimizer_GPU(max_iter,lambda,tol,theta,G1,G2,G3,G4,G5,G1_Delta,G2_Delta,G3_Delta,G4_Delta,G5_Delta,Y1_gpu,Y2_gpu)

  best_error = inf;
  alpha = 4;
  error_list = zeros(max_iter,1,'gpuArray');

  % 计算残差
  [R,J] = compute_residual(theta,G1,G2,G3,G4,G5,G1_Delta,G2_Delta,G3_Delta,G4_Delta,G5_Delta,Y1_gpu,Y2_gpu);
  current_error = norm(R)^2;
  % 计算阻尼因子
  lambda = 0.01 * max(diag(J'*J));
  
  for iter = 1:max_iter
    % 计算Hessian和梯度
    H = J' * J;
    grad = J' * R;
    grad = grad / (norm(grad) + 1e-12); % 添加归一化，防止零梯度
    
    % 添加最小正则化保证正定性
    %H_reg = H + lambda*diag(diag(H)) + 1e-12*eye(size(H));
    H_reg = H + lambda*eye(size(H));
    
    % 原代码
    % delta = -H_reg \ grad;
    % 改进方案：Cholesky分解 + 条件数检查
    [HR, p] = chol(H_reg);
    if p == 0
        delta = -HR \ (HR' \ grad); % Cholesky分解求解
    else
        % 退化到伪逆
        warning('矩阵不正定，使用SVD求解');
        [U, S, V] = svd(H_reg);
        s = diag(S);
        s_inv = 1./s;
        s_inv(s < 1e-12*max(s)) = 0; % 截断小奇异值
        delta = -V*(s_inv.*(U'*grad));
    end
    
    % 更新参数
    delta = alpha*delta;
    theta_new = theta + delta;
    [R_new, J_new] = compute_residual(theta_new, G1,G2,G3,G4,G5,G1_Delta,G2_Delta,G3_Delta,G4_Delta,G5_Delta,Y1_gpu,Y2_gpu);
    new_error = norm(R_new)^2;
    
    error_list(iter) = new_error;
    % 在迭代循环中添加：
    %rho = (current_error - new_error) / (grad' *delta + 0.5 * delta' * H_reg * delta);
    rho = (current_error - new_error) / (0.5*delta'*(lambda*delta + grad));
    if new_error < current_error 
        if abs(new_error - best_error) < tol
          break;
        end

        if rho > 0.75
            lambda = min(lambda*2, 2^7);  % 近似良好
            alpha = max(alpha * 0.5, 1);
        elseif rho < 0.01
            lambda = max(lambda/16, 1e-9);
            alpha = min(alpha * 16, 2^10);
        elseif rho < 0.1
            lambda = max(lambda/4, 1e-9);
            alpha = min(alpha * 4, 2^10);
        elseif rho < 0.25
            lambda = max(lambda/2, 1e-9);
            alpha = min(alpha * 2, 2^10);
        end
        
        theta = theta_new;
        best_error = new_error;
        current_error = new_error;
        R = R_new;
        J = J_new;

    else
        lambda = min(lambda * 4, 2^7);  % 保守增加
    end
    
    fprintf('Iter %d: Error = %.6f, rho = %.3f, Lambda = %.2e, alpha = %.3f\n', iter, new_error, rho, lambda, alpha);
  end
end


%%
function [R, J]= compute_residual(theta,G1,G2,G3,G4,G5,G1_Delta,G2_Delta,G3_Delta,G4_Delta,G5_Delta,Y1_gpu,Y2_gpu)
  [n2, n2choose2]=size(G2);
  X1 = gpuArray(theta(1:n2));
  X2 = gpuArray(theta(n2+1:end));
  X1 = X1(:);  X2 = X2(:);

  % 残差计算
  kron_X1X1 = reshape(X1 * X1', 1, n2, n2);
  kron_X1X2 = reshape(X1 * X2', 1, n2, n2choose2);
  kron_X2X2 = reshape(X2 * X2', 1, n2choose2, n2choose2);

  term1 = sum(reshape(G3, size(G3,1), n2, n2) .* kron_X1X1, [2,3]);
  term2 = 2 * sum(reshape(G4, size(G4,1), n2, n2choose2) .* kron_X1X2, [2,3]);
  term3 = double(sum(reshape(G5, size(G5,1), n2choose2, n2choose2) .* kron_X2X2, [2,3]));

  term1D = sum(reshape(G3_Delta, size(G3_Delta,1), n2, n2) .* kron_X1X1, [2,3]);
  term2D = 2 * sum(reshape(G4_Delta, size(G4_Delta,1), n2, n2choose2) .* kron_X1X2, [2,3]);
  term3D = double(gpuArray(sum(reshape(G5_Delta, size(G5_Delta,1), n2choose2, n2choose2) .* gather(kron_X2X2), [2,3])));
  
  R = [G1*X1 + G2*X2 + term1 + term2 + term3 - Y1_gpu; G1_Delta*X1 + G2_Delta*X2 + term1D + term2D + term3D - Y2_gpu];

  % 雅可比计算
  I_N = eye(n2, 'gpuArray');
  I_C = eye(n2choose2, 'double');
  
  % 计算非线性项的导数
  D_X1 = kron(X1, I_N) + kron(I_N, X1);
  D_X2 = kron(gather(X2), I_C) + kron(I_C, gather(X2));
  
  % 雅可比矩阵
  J = [G1 + G3 * D_X1 + 2 * G4 * kron(I_N, X2), G2 + 2 * G4 * kron(X1, I_C) + double(gpuArray(gather(G5) * D_X2)); G1_Delta + G3_Delta * D_X1 + 2 * G4_Delta * kron(I_N, X2), G2_Delta + 2 * G4_Delta * kron(X1, I_C) + double(gpuArray(G5_Delta * D_X2))];
end