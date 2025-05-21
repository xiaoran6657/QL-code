function [ori_A_adj, P3_tensor] = Reconstruction_TS_GPU(UAU_state_nodes, SIS_state_nodes)
    % Reconstruction network by Two-step method with GPU acceleration
    % Input:
        % UAU_state_nodes: the node state matrix of the virtual layer(UAU), [T, n]
        % SIS_state_nodes: the node state matrix of the physical layer(SIS), [T, n]
        % Lambda: the regularization parameter
    % Output:
        % ori_A_adj: the reconstructed network two-body interaction, [n, n]
        % P3_tensor: the reconstructed network three-body interaction, [n, n, n]
    
    tic; % Start timer
    
    % Transfer input data to GPU if available
    if gpuDeviceCount > 0
        UAU_state_nodes = gpuArray(UAU_state_nodes);
        SIS_state_nodes = gpuArray(SIS_state_nodes);
    end
    
    [~, n] = size(UAU_state_nodes);  % m time points, n nodes
    ori_A_adj = zeros(n, n, 'gpuArray');
    P3_tensor = zeros(n, n, n, 'gpuArray');  % n*n*n 3D tensor for raw values
    Lambda = 1e-3;  % lasso parameter
    
    options1 = optimoptions('fsolve', 'Display', 'none');
    options2 = statset('Display', 'off');

    % Loop through all nodes to solve first and second order edges
    for nod = 1:n
        fprintf("nod: %d \n", nod);
        
        %%%step one
        % solve two gamma (vector x)
        [X, Y, theta1, ~, A2, ~] = Extract(UAU_state_nodes, SIS_state_nodes, nod);
        x0 = 0.9999;
        x = fsolve(@(x) myfunTS(x, X, Y, theta1), x0, options1);  % solve TS.Eq(4.27)
        
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
        
        % Solve CX=D, get P0=X1 (column vector)
        % Transfer to CPU for lasso if needed (lasso doesn't support GPU)
        %C_cpu = gather(C); D_cpu = gather(D);
        %M_poly = @(x) x - 0.5*C*x + 0.25*C^2*x;  % 最小二乘多项式
        %C_normalized = (2 / max(abs(eig(C)))) * C - eye(size(C));  % 归一化矩阵
        %M_poly = @(x) chebyshev_preconditioner(C_normalized, x, 3);  % 构造四阶切比雪夫多项式预处理器

        %[TS_X1, flag] = gmres(C, D, 3, 1e-4, 10, M_poly);  % 使用GMRES求解
        TS_X1 = lasso(gather(C), gather(D), 'Lambda', Lambda, 'RelTol', 1e-4, 'Options', options2);

        TS_X1 = gpuArray(-TS_X1);
        tru = fun_cut(TS_X1);  % Find truncation value (on CPU)
        neig = find(TS_X1 >= tru);     % Get neighbor subset
        UAU_state_nodes_neig = UAU_state_nodes(:, sort([nod, neig']));
        SIS_state_nodes_neig = SIS_state_nodes(:, sort([nod, neig']));
        nod_index = find(sort([nod, neig']) == nod);  % index in compressed matrix

        %%%step two
        % solve two gamma (vector x)
        [X, Y, theta1, theta2, A2, A3] = Extract(UAU_state_nodes_neig, SIS_state_nodes_neig, nod_index);
        x0 = [0.9999, 0.9999];
        x = fsolve(@(x) myfun(x, X, Y, theta1, theta2), x0, options1);  % Eq(4.29)
        if min(x) <= 0
            disp('fsolve alpha <= 0;\n')
            continue
        end

        % solve equation CX=D
        M = x(1).^theta1 .* x(2).^theta2;            % a^(x_0)*b^(y_0), Eq(4.24)
        f = M./(1-M+eps) - M.*log(M)./((1-M+eps).^2); % \={W}^i(t_m), Eq(4.31)
        g = M./((1-M+eps).^2);                       % \={V}^i(t_m), Eq(4.31)

        n2 = length(neig) + 1;
        C = zeros(n2*(n2+1)/2, n2*(n2+1)/2, 'gpuArray'); % [A_1, A_2; A_1^delta, A_2^delta]
        D = zeros(n2*(n2+1)/2, 1, 'gpuArray');           % [Y_1, Y_2]
        
        % Upper C and D (first order edges)
        for i = 1:n2
            C(i,1:n2) = sum(Y.*A2(:,i).*g .* A2, 1);      % A_1(i,:), Eq(4.40)
            C(i,(n2+1):n2*(n2+1)/2) = sum(Y.*A2(:,i).*g .* A3, 1); % A_2(i,:), Eq(4.41)
            D(i) = sum((X-Y.*f).*A2(:,i));                % Y_1(i), Eq(4.38)
        end
        % Lower C and D (second order edges)
        for i = (n2+1):n2*(n2+1)/2
            C(i,1:n2) = sum(Y.*A3(:,i-n2).*g.*A2, 1);   % A_1^delta(i,:), Eq(4.42)
            C(i,(n2+1):n2*(n2+1)/2) = sum(Y.*A3(:,i-n2).*g.*A3, 1); % A_2^delta(i,:), Eq(4.43)
            D(i) = sum((X-Y.*f).*A3(:,i-n2));             % Y_2(i), Eq(4.39)
        end

        % Solve CX=D, get P0=[P1 P2]'=[eta1 eta2]' (column vector)
        %C_cpu = gather(C);D_cpu = gather(D);
        M_poly = @(x) x - 0.5*C*x + 0.25*C^2*x;  % 最小二乘多项式
        %C_normalized = (2 / max(abs(eig(C)))) * C - eye(size(C));  % 归一化矩阵
        %M_poly = @(x) chebyshev_preconditioner(C_normalized, x, 3);  % 构造四阶切比雪夫多项式预处理器

        [P0, flag] = gmres(C, D, 10, 1e-6, 100, M_poly);  % 使用GMRES求解
        %P0 = lasso(gather(C), gather(D), 'Lambda', Lambda, 'RelTol', 1e-4, 'Options', options2);
        P0 = gpuArray(P0);
        
        % P1=eta1: first order edges for nod (n2 elements)
        P1 = P0(1:n2);
        P5 = zeros(n, 1, 'gpuArray');
        P5(sort([nod; neig])) = P1;
        ori_A_adj(:,nod) = P5;

        % P2=eta2: second order edges for nod (C_n^2 elements)
        P2 = P0((n2+1):n2*(n2+1)/2);
        P3 = zeros(n2, n2, 'gpuArray');
        temp2 = find(tril(ones(n2, n2, 'gpuArray'), -1));
        P3(temp2) = P2;
        P4 = zeros(n, n, 'gpuArray');
        P4(sort([nod; neig]), sort([nod; neig])) = P3;
        P3_tensor(:, :, nod) = P4;
    end
  
    % Gather results back to CPU if they're on GPU
    ori_A_adj = gather(ori_A_adj);
    P3_tensor = gather(P3_tensor);
    toc; % End timer
end

%% Find the two-body truncation value
function c_cut = fun_cut(a)
    %Find the two-body truncation value
    %Input: Connection probability between reconstructed node and other nodes
    %Output: Get the truncation value in the vector

    aveave = mean(a(a <= mean(a))); 
    a_c = min(aveave, 1/length(a));  
    if a_c > 0
        a_n = a(a > a_c);
        if isrow(a_n)
            b = [-sort(-a_n), a_c];   % Descending order
        else
            b = [-sort(-a_n); a_c];   % Descending order
        end
        [~, id1] = max(b(1:end-1).*(b(1:end-1)-b(2:end))./b(2:end));
        bb = b;
        b(1:id1) = [];
        if isscalar(b)
            c_cut = bb(id1);  
        else
            [~, id1] = max(b(1:end-1).*(b(1:end-1)-b(2:end))./b(2:end));
            c_cut = b(id1+1);  
        end
    else
        c_cut = 0.000001;
    end

    % Keep data on GPU if input was on GPU
    if isa(a, 'gpuArray')
        c_cut = gpuArray(c_cut);
    end
end


%% 切比雪夫多项式构造
function T = chebyshev_preconditioner(C_normalized, x, order)
    if order == 0
        T = x;
    elseif order == 1
        T = C_normalized * x;
    else
        T0 = x;
        T1 = C_normalized * x;
        for k = 2:order
            T = 2*C_normalized*T1 - T0;
            T0 = T1;
            T1 = T;
        end
    end
end


%% Function myfun to solve two alpha (vector x) Eq(4.29)
function F = myfun(x, X, Y, theta1, theta2)
    F(1) = sum(Y.*theta1./(1-x(1).^theta1.*x(2).^theta2+eps) - (X+Y).*theta1);
    F(2) = sum(Y.*theta2./(1-x(1).^theta1.*x(2).^theta2+eps) - (X+Y).*theta2);
    F = gather(F);
end


%% Function myfunTS for two-step method first step (solve one alpha) TS.Eq(4.27)
function F = myfunTS(x, X, Y, theta1)
    F = gather(sum(Y.*theta1./(1-x.^theta1) - (X+Y).*theta1));
end


%% Function Extract (modified for GPU)
function [X, Y, theta1, theta2, A2, A3] = Extract(SA, SB, nod)
    % Extract the special data of the node i from the node state matrix SA and SB
    % Input:
        % SA: the node state matrix of the virtual layer(UAU), UAU_state_nodes
        % SB: the node state matrix of the physical layer(SIS), SIS_state_nodes
        % nod: the node i
    % Output:
        % X: \={Q}^i(t_m), in Eq(4.8)
        % Y: \={R}^i(t_m), in Eq(4.8)
        % theta1: \={theta}^i(t_m), in Eq(4.22)
        % theta2: \={theta}^i_{Delta}(t_m), in Eq(4.23)
        % A2: Filtered node state matrix, \={s}^i(t), in Eq(4.38)
        % A3: \={s}^j * \={s}^k, in Eq(4.39)

    [m, n] = size(SA); % [T, n]
    A1 = SA(:, nod);   % Extract column i
    A2 = SA;
    t = find(A1 == 0); % Find positions of 0 in A1
    t(t == m) = [];    % Only record first m-1 time points

    A2 = A2(t, :);     % States at effective time points
    A1 = A1(t+1, :);   % Next time point state for nod

    B1 = SB(:, nod);    
    B1 = B1(t+1, :);   % Next time point physical state

    Y = A1 .* (1-B1);  % \={Q}^i(t_m)
    if nnz(Y) == 0     % If Y is all zeros
        rand_indices = randi(length(Y), ceil(length(Y)*0.001), 1);
        Y(rand_indices) = 1;
    end

    X = 1 - A1;        % 1-\={s}^i(t_m)
    if nnz(X) == 0     % If X is all zeros
        rand_indices = randi(length(X), ceil(length(X)*0.001), 1);
        X(rand_indices) = 1;
    end

    theta1 = sum(A2, 2);  % \={theta}^i(t_m)
    [m1, ~] = size(A2);   % Number of effective time points
    theta2 = zeros(m1, 1, 'like', theta1);
    for i = 1:m1
        b = A2(i,:)' * A2(i,:);
        theta2(i,1) = sum(sum(b - diag(diag(b)))) / 2;  % \={theta}^i_{Delta}(t_m)
    end

    A3 = zeros(m1, n*(n-1)/2, 'like', A2); % (\={s}^j * \={s}^k)
    for j = 1:m1
        temp1 = A2(j,:)' * A2(j,:);  % [n,n]
        A3(j,:) = (temp1(logical(tril(ones(size(temp1)), -1))))';
    end
end