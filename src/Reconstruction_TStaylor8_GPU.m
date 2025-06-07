function [ori_A_adj, P3_tensor] = Reconstruction_TStaylor8_GPU(UAU_state_nodes, SIS_state_nodes)
  % Reconstruction network by Two-step method with GPU acceleration, 
  %     using the second-order Taylor expansion to step one and step two of the Two-step method
  % (https://doi.org/10.1038/s41467-022-30706-9|www.nature.com/naturecommunications)
  % Input:
      % UAU_state_nodes: the node state matrix of the virtual layer(UAU), [T, n]
      % SIS_state_nodes: the node state matrix of the physical layer(SIS), [T, n]
  % Output:
      % ori_A_adj: the reconstructed network two-body interaction, [n, n]
      % P3_tensor: the reconstructed network three-body interaction, [n, n, n]
  
  % // TODO
    % 1.替换TS1 pinv √
    % 2.超参调节lambda, lambda_Delta
    % 3.TS1 + taylor2 √
    % 4.Hessian矩阵
    % 5.物理层识别
  
  % Transfer input data to GPU if available
  %if gpuDeviceCount > 0
  %  UAU_state_nodes = gpuArray(UAU_state_nodes);
  %  SIS_state_nodes = gpuArray(SIS_state_nodes);
  %end

  [~, n]=size(UAU_state_nodes);  % 共m个时刻，n个节点
  ori_A_adj = zeros(n, n);
  P3_tensor = zeros(n, n, n);  % 创建一个n*n*n的三维张量，在确定阈值前需收集原始小数
  
  UAU_state_nodes_global = parallel.pool.Constant(UAU_state_nodes);
  SIS_state_nodes_global = parallel.pool.Constant(SIS_state_nodes);

  % 循环求解所有节点的一阶边和二阶边
  parfor nod = 1:n
    tic; % 开始计时

    %% step one --- Taylor1
    [X,Y,A2] = extract(UAU_state_nodes_global.Value, SIS_state_nodes_global.Value, nod);

    [A2,neig_subset] = solveTS1(X,Y,A2,nod);

    %% step two --- Taylor2
    P0 = solveTS2(A2, X, Y);

    % ------------------- 数据写入 -------------------
    n2=size(A2,2);   % m1:有效时刻数量，n2:TS2节点数量

    P1=P0(1:n2);
    % ori_A_adj:重构的网络邻接矩阵，元素为小数，不一定对称
    P5 = zeros(n,1);
    P5(neig_subset) = P1;
    ori_A_adj(:,nod) = gather(P5);

    % P2=eta2:该nod二阶边（C_n^2个元素组成的列向量）
    %P2=zeros(n*(n-1)/2,1);
    P2=P0((n2+1):n2*(n2+1)/2);
    % P3:将列向量P2转换为矩阵存到严格下三角阵P3里
    P3=zeros(n2,n2);
    temp2 = find(tril(ones(n2, n2), -1));  %获取P2的严格下三角部分的线性索引
    P3(temp2) = P2;  %将P2中的元素按照列优先顺序填充到P3的严格下三角部分
    P4 = zeros(n,n);
    P4(neig_subset, neig_subset) = P3;
    P3_tensor(:, :, nod) = gather(P4);  %将计算出的P3存储到三维张量中的对应切片

    time = toc; % 结束计时并输出所用时间
    fprintf("nod: %d/%d, time: %.4f(s)\n", nod, n, time);

    % ------------------- 变量清理 -------------------
    A1=[]; A2=[]; A2j=[]; A3=[]; B1=[]; C=[]; D=[]; M=[]; Oi=[]; P1=[]; 
    P2=[]; P3=[]; P4=[]; P5=[]; SIS_state_nodes_copy=[]; UAU_state_nodes_copy=[];
    TS_X1=[]; Vi=[]; Wi=[]; X=[]; X0=[]; Y=[]; f=[]; g=[]; lb=[]; neig=[];
    neig_subset=[]; t=[]; temp1=[]; temp2=[]; theta1=[]; theta2=[]; ub=[]; 

  end
end


%% Extract提取X Y A2
function [X, Y, A2] = extract(UAU_state_nodes, SIS_state_nodes, nod)
% Extract the special data of the node i from the node state matrix SA and SB
% Input:
    % UAU_state_nodes: the node state matrix of the virtual layer(UAU), UAU_state_nodes
    % SIS_state_nodes: the node state matrix of the physical layer(SIS), SIS_state_nodes
    % nod: the node i
% Output:
    % X: \={Q}^i(t_m), in Eq(4.8)
    % Y: \={R}^i(t_m), in Eq(4.8)
    % A2: Filtered node state matrix, \={s}^i(t), in Eq(4.38)
    m=size(UAU_state_nodes,1); % [T, n]
    
    A1=UAU_state_nodes(:,nod);   % Extract Virtual layer's column i, [1,n]
    A2=UAU_state_nodes;
    t=find(A1==0);  % 寻找A1中0元素的位置，将位置记录至t，如t=[2 3 8]
    t(t==m)=[];     % 只记录前m-1个时刻里，A1的0元素的位置，保证t+1时刻还有数据
    A2=A2(t,:);     % 这些有效时刻所有节点的状态
    A1=A1(t+1,:);   % 下一时刻nod虚拟层的状态
    
    B1=SIS_state_nodes(:,nod);   % Extract Physical layer's column i    
    B1=B1(t+1,:);   % 下一时刻nod实际层的状态
    
    Y=A1.*(1-B1);   % \={Q}^i(t_m)表达式第一项通过t的筛选而导致必等于1==>
    if nnz(Y)/m < 0.01  % 如果Y全为0，即没有1，随机选取0.1%的位置赋值为1
        for i=1:ceil(length(Y)*0.001)
            Y(randi(length(Y))) = 1;
        end
    end
    
    X=1-A1;         % 1-\={s}^i(t_m)===1, in Eq(4.8)
    if nnz(X)/m < 0.01  % 如果X全为0，即没有1，随机选取0.1%的位置赋值为1
        for i=1:ceil(length(X)*0.001)
            X(randi(length(X))) = 1;
        end
    end

    clear A1 t B1 m temp1;
end


%% TS1求解
function [A2,neig_subset] = solveTS1(X,Y,A2,nod)
    theta1=sum(A2,2);  % \={theta}^i(t_m), in Eq(4.22)

    x0 = 0.9999;
    x = fsolve(@(x) myfunTS(x,X,Y,theta1),x0, optimoptions('fsolve', 'Display', 'none'));   % solve TS.Eq(4.27), x=alpha^i

    % solve TS.Eq(4.37)
    M = x.^theta1;                                             % instead a^(x_0) by TST2.Eq(4.24)
    Wi = M./(1-M+eps) - M.*log(M)./((1-M+eps).^2) + ...
         M.*(1+M).*(log(M)).^2/(2*(1-M+eps).^3);               % \={W}^i(t_m), TST2.Eq(4.31)
    Vi = M./((1-M+eps).^2) - M.*(1+M).*log(M)/((1-M+eps).^3);  % \={V}^i(t_m), TST2.Eq(4.31)
    Oi = M.*(1+M) / (2.*(1-M+eps).^3);                         % \={O}^i(t_m), TST2.Eq(4.31)

    %TS_X1 = pinv(C) * D;  % 伪逆求解，避免矩阵奇异
    TS_X1 = lsqnonlin_Opt_TS1(A2,Oi,Vi,Wi,X,Y);

    TS_X1 = gpuArray(TS_X1);
    %tru = fun_cut(TS_X1);  % Find truncation value (on CPU)
    tru = graythresh(TS_X1);
    %tru = prctile(TS_X1, 70);  % 选择50%中位数为阈值
    neig = find(TS_X1 >= tru);     % Get neighbor subset
    neig_subset = sort(union(nod, neig'));
    % 矩阵裁剪后X, Y并不改变，仅对A2进行裁切
    A2 = A2(:,union(nod, neig));
end


%% 
function P0 = lsqnonlin_Opt_TS1(A2,Oi,Vi,Wi,X,Y)
% ------------------- 最小二乘 -------------------
    fun = @(x) compute_residual_torch_TS1(x,A2,Oi,Vi,Wi,X,Y);  % 注意参数传递，尤其是x
    
    % 配置优化选项
    options_lsq = optimoptions('lsqnonlin', ...
    'Algorithm', 'trust-region-reflective', ...
    'SpecifyObjectiveGradient', true, ...  % 使用自定义雅可比
    'StepTolerance', 1e-8, ...              % 放宽步长容差
    'FunctionTolerance', 1e-6, ...          % 放宽函数值容差
    'MaxIterations', 30,...
    'Display', 'iter');
    %'Display', 'iter', ...
    
    % 变量边界（强制变量在[0,1]区间）
    n = size(A2,2);
    
    lb = zeros(n, 1);
    ub = ones(n, 1);
    X0 = zeros(n,1);
    
    fprintf("solve TS1:\n");
    [P0, resnorm] = lsqnonlin(fun, X0, lb, ub, options_lsq);

end


%% 计算TS1残差和雅可比矩阵（直接计算版本）
function [R,J] = compute_residual_torch_TS1(theta,A2,Oi,Vi,Wi,X,Y)
    % 初始化\alpha
    lambda = 0.1;       alpha = gpuArray(log(1-lambda));
    n = size(A2,2);    
    S1 = gpuArray(theta(1:n));
    S1 = S1(:);
    
    % 计算投影（关键优化）
    proj_A2_S1 = A2 * S1;      % [m1, 1]
    
    % 完整残差
    R = gather(alpha * (Y .* A2)' * (Vi .* proj_A2_S1)...
               + alpha^2 * (Y .* A2)' * (Oi .* (proj_A2_S1.^2))...
               - sum((X - Y .* Wi) .* A2, 1)');
    
    % 雅可比矩阵计算 - 避免显式克罗内克积
    J = gather(alpha * (Y .* A2)' * (Vi .* A2)...
          + alpha^2 * (Y .* A2)' * (Oi .* (A2 .* proj_A2_S1)));
    
    clear lambda alpha n S1 proj_A2_S1;
end


%% Function myfunTS 两步法第一步中用来求解一个alpha（即向量x） TS.Eq(4.27)
function F=myfunTS(x,X,Y,theta1)
  F = gather(sum(Y.*theta1./(1-x.^theta1)-(X+Y).*theta1));
end


%% TS2
function P0 = solveTS2(A2, X, Y)
    theta1=sum(A2,2);  % \={theta}^i(t_m), in Eq(4.22)
    [m1,n2]=size(A2);   % m1:有效时刻数量，n2:TS2节点数量
    n2choose2 = nchoosek(n2,2); % X2维度
    
    theta2 = gpuArray.zeros(m1,1,'double');  % \={theta}^i_{Delta}(t_m), in Eq(4.23)
    A3 = gpuArray.zeros(m1,n2choose2,'double');       % 求A3，即[\={s}^j * \={s}^k]_{m1 \times C_N^2}
    for j=1:m1
        A2j = A2(j,:);
        temp1=A2j'*A2j;
        A3(j,:)=(temp1(logical(tril(ones(size(temp1)),-1))))';  % tril(*,-1)取严格下三角，从下三角矩阵到一行的转换通过列优先顺序实现
        theta2(j,1)=sum(sum(temp1-diag(diag(temp1))))/2;  % \={theta}^i_{Delta}(t_m), in Eq(4.23)
    end
    
    % 计算相关变量
    x0=[0.9999, 0.9999];
    x = fsolve(@(x) myfun(x,X,Y,theta1,theta2),x0, optimoptions('fsolve', 'Display', 'none'));  % solve Eq(4.29), x=[alpha^i, alpha^i_{Delta}]
    
    M=x(1).^theta1.*x(2).^theta2;                         % instead a^(x_0)*b^(y_0) by Eq(4.24)
    Oi = M.*(1+M)./(2.*(1-M+eps).^3);                         % \={O}^i(t_m), [m1,1], in TST2.Eq(4.31)
    Vi = M./((1-M+eps).^2)-2.*Oi.*log(M);                     % \={V}^i(t_m), [m1,1], in TST2.Eq(4.31)
    Wi = M./(1-M+eps)-M.*log(M)./((1-M+eps).^2)+Oi.*(log(M).^2);  % \={W}^i(t_m), [m1,1], in TST2.Eq(4.31)

    fprintf("solve TS2:\n");
    P0 = lsqnonlin_Opt_TS2(A2,A3,Oi,Vi,Wi,X,Y);
end


%% Function myfun 用来求解两个alpha（即向量x） Eq(4.29)
function F=myfun(x,X,Y,theta1,theta2)
  F(1)=sum(Y.*theta1./(1-x(1).^theta1.*x(2).^theta2+eps)-(X+Y).*theta1);
  F(2)=sum(Y.*theta2./(1-x(1).^theta1.*x(2).^theta2+eps)-(X+Y).*theta2);
  F = gather(F);
end


function P0 = lsqnonlin_Opt_TS2(A2,A3,Oi,Vi,Wi,X,Y)
% ------------------- 最小二乘 -------------------
    fun = @(x) compute_residual_torch_TS2(x,A2,A3,Oi,Vi,Wi,X,Y);  % 注意参数传递，尤其是x
    
    % 配置优化选项
    options_lsq = optimoptions('lsqnonlin', ...
    'Algorithm', 'trust-region-reflective', ...
    'SpecifyObjectiveGradient', true, ...  % 使用自定义雅可比
    'StepTolerance', 1e-8, ...              % 放宽步长容差
    'FunctionTolerance', 1e-6, ...          % 放宽函数值容差
    'MaxIterations', 100,...
    'Display', 'iter');
    %'Display', 'iter', ...
    
    % 变量边界（强制变量在[0,1]区间）
    n2=size(A2,2);   % m1:有效时刻数量，n2:TS2节点数量
    n2choose2 = nchoosek(n2,2); % X2维度
    
    nvars = n2+n2choose2;        % 待求解变量维度（根据实际问题调整）
    lb = zeros(nvars, 1);
    ub = ones(nvars, 1);
    X0 = zeros(n2+n2choose2,1);
    
    [P0, resnorm] = lsqnonlin(fun, X0, lb, ub, options_lsq);

end


%% 计算TS2残差和雅可比矩阵（直接计算版本）
function [R,J] = compute_residual_torch_TS2(theta,A2,A3,Oi,Vi,Wi,X,Y)
    % 初始化\alpha
    lambda = 0.1;       alpha = gpuArray(log(1-lambda));
    lambda_Delta = 0.9; alpha_Delta = gpuArray(log(1-lambda_Delta));

    n2 = size(A2,2);
    % 预计算严格下三角部分的索引
    mask = tril(true(n2, n2), -1);
    [k_indices, i_indices] = find(mask);
    temp2_matrix = A2(:, k_indices) .* A2(:, i_indices); % m1 × n2choose2

    D1 = sum((X - Y .* Wi) .* A2, 1)';     % n2 × 1      
    D2 = sum((X - Y .* Wi) .* A3, 1)';     % n2choose2 × n2choose2^2
    
    S1 = gpuArray(theta(1:n2));
    S2 = gpuArray(theta(n2+1:end));
    S1 = S1(:);  S2 = S2(:);
    
    % 计算投影（关键优化）
    proj_A2_S1 = A2 * S1;      % [m1, 1]
    proj_A3_S2 = A3 * S2;      % [m1, 1]
    temp2S2 = temp2_matrix * S2;  % [m1, 1]
    
    % 直接计算残差
    Y1_pred = alpha * (Y .* A2)' * (Vi .* proj_A2_S1) + alpha_Delta * (Y .* A2)' * (Vi .* proj_A3_S2) + ...
              alpha^2 * (Y .* A2)' * (Oi .* (proj_A2_S1.^2)) + ...
              2 * alpha * alpha_Delta * (Y .* A2)' * (Oi .* (proj_A2_S1 .* temp2S2)) + ...  % n2 x n2
              alpha_Delta^2 * (Y .* A2)' * (Oi .* (temp2S2.^2));
    Y2_pred = alpha * (Y .* A3)' * (Vi .* proj_A2_S1) + alpha_Delta * (Y .* A3)' * (Vi .* proj_A3_S2) + ...
              alpha^2 * (Y .* A3)' * (Oi .* (proj_A2_S1.^2)) + ...
              2 * alpha * alpha_Delta * (Y .* A3)' * (Oi .* (proj_A2_S1 .* temp2S2)) + ... % n2choose2 x n2
              alpha_Delta^2 * (Y .* A3)' * (Oi .* (temp2S2.^2));
    
    % 完整残差
    R = gather([Y1_pred - D1; Y2_pred - D2]);
    
    
    % 雅可比矩阵计算 - 避免显式克罗内克积
    % J11: n2 × n2
    J11 = alpha * (Y .* A2)' * (Vi .* A2) +...
          alpha^2 * (Y .* A2)' * (Oi .* (A2 .* proj_A2_S1)) +...
          4 * alpha * alpha_Delta * (Y .* A2)' * (Oi .* (A2 .* temp2S2));  

    % J12: n2 × n2choose2
    J12 = alpha_Delta * (Y .* A2)' * (Vi .* A3) +...
          4 * alpha * alpha_Delta * (Y .* A2)' * (Oi .* (A3 .* proj_A2_S1)) +...  
          alpha_Delta^2 * (Y .* A2)' * (Oi .* (temp2_matrix .* temp2S2));

    % J21: n2choose2 × n2
    J21 = alpha * (Y .* A3)' * (Vi .* A2) +...
          alpha^2 * (Y .* A3)' * (Oi .* (A2 .* proj_A2_S1)) +...
          4 * alpha * alpha_Delta * (Y .* A3)' * (Oi .* (A2 .* temp2S2));

    % J22: n2choose2 × n2choose2
    J22 = alpha_Delta * (Y .* A3)' * (Vi .* A3) +...
          4 * alpha * alpha_Delta * (Y .* A3)' * (Oi .* (A3 .* proj_A2_S1)) +...
          alpha_Delta^2 * (Y .* A3)' * (Oi .* (temp2_matrix .* temp2S2));

    J = gather([J11, J12; J21, J22]);
    
    clear mask k_indices i_indices temp2_matrix D1 D2 S1 S2 proj_A2_S1...
        proj_A3_S2 temp2S2 Y1_pred Y2_pred J11 J12 J21 J22;
end
