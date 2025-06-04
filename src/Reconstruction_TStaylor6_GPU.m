function [ori_A_adj, P3_tensor] = Reconstruction_TStaylor6_GPU(UAU_state_nodes, SIS_state_nodes)
  % Reconstruction network by Two-step method with GPU acceleration, 
  %     using the second-order Taylor expansion to step two of the Two-step method
  % (https://doi.org/10.1038/s41467-022-30706-9|www.nature.com/naturecommunications)
  % Input:
      % UAU_state_nodes: the node state matrix of the virtual layer(UAU), [T, n]
      % SIS_state_nodes: the node state matrix of the physical layer(SIS), [T, n]
  % Output:
      % ori_A_adj: the reconstructed network two-body interaction, [n, n]
      % P3_tensor: the reconstructed network three-body interaction, [n, n, n]
  
  % TODO
    % 1.替换TS1 lasso
    % 2.超参调节lambda, lambda_Delta, prctile()
    % 3.
  tic;

  % Transfer input data to GPU if available
  if gpuDeviceCount > 0
    UAU_state_nodes = gpuArray(UAU_state_nodes);
    SIS_state_nodes = gpuArray(SIS_state_nodes);
  end

  [~, n]=size(UAU_state_nodes);  % 共m个时刻，n个节点
  ori_A_adj = zeros(n, n, 'gpuArray');
  P3_tensor = zeros(n, n, n, 'gpuArray');  % 创建一个n*n*n的三维张量，在确定阈值前需收集原始小数
  
  UAU_state_nodes_global = parallel.pool.Constant(UAU_state_nodes);
  SIS_state_nodes_global = parallel.pool.Constant(SIS_state_nodes);

  % 循环求解所有节点的一阶边和二阶边
  for nod = 1:n
      tic; % 开始计时

      % 全局数据加载
      UAU_state_nodes_copy = UAU_state_nodes_global.Value;
      SIS_state_nodes_copy = SIS_state_nodes_global.Value;

      Lambda = 1e-3;  % lasso parameter
      % 初始化\alpha
      lambda = 0.1;       alpha = gpuArray(log(1-lambda));
      lambda_Delta = 0.9; alpha_Delta = gpuArray(log(1-lambda_Delta));
      
      options1 = optimoptions('fsolve', 'Display', 'none');
      options2 = statset('Display', 'off');

      %%%step one --- Taylor1
      % solve 两个gamma（即向量x）
      [X, Y, A2]=Extract(UAU_state_nodes_copy, SIS_state_nodes_copy, nod);
      theta1=sum(A2,2);  % \={theta}^i(t_m), in Eq(4.22)
      %theta1 = sum(A2, 2) - A2(:,nod);  % 对每一行，去除nod节点贡献
      %A2_nodRemove = A2;
      %A2_nodRemove(:, nod) = 0;  % 将nod节点的所有状态置为0（去除本身节点影响）

      x0 = 0.9999;
      x = fsolve(@(x) myfunTS(x,X,Y,theta1),x0,options1);   % solve TS.Eq(4.27), x=alpha^i
      
      % solve TS.Eq(4.37)
      M = x.^theta1;                                % instead a^(x_0) by TS.Eq(4.24)
      f = M./(1-M+eps) - M.*log(M)./((1-M+eps).^2); % \={W}^i(t_m), TS.Eq(4.31)
      g = M./((1-M+eps).^2);                        % \={V}^i(t_m), TS.Eq(4.31)
      
      C = zeros(n, n);                  % A_1, TS.Eq(4.37)
      D = zeros(n, 1);                  % Y_1, TS.Eq(4.37)
      
      % Calculate A_1 and Y_1
      for i = 1:n
          % lambda1, first n columns for j values
          C(i,1:n) = sum(Y.*A2(:,i).*g .* A2, 1);
          D(i) = sum((X-Y.*f).*A2(:,i));            % Y_1(i), TS.Eq(4.37)
      end
      
      TS_X1 = lasso(C, D, 'Lambda', Lambda, 'RelTol', 1e-4, 'Options', options2);
      
      TS_X1 = gpuArray(-TS_X1);
      %tru = fun_cut(TS_X1);  % Find truncation value (on CPU)
      %tru = graythresh(TS_X1);
      tru = prctile(TS_X1, 50);  % 选择50%中位数为阈值
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
        A2j = A2(j,:);
        temp1=A2j'*A2j;
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
      Oi = M.*(1+M)./(2.*(1-M+eps).^3);                         % \={O}^i(t_m), [m1,1], in TST2.Eq(4.31)
      Vi = M./((1-M+eps).^2)-2.*Oi.*log(M);                     % \={V}^i(t_m), [m1,1], in TST2.Eq(4.31)
      Wi = M./(1-M+eps)-M.*log(M)./((1-M+eps).^2)+Oi.*(log(M).^2);  % \={W}^i(t_m), [m1,1], in TST2.Eq(4.31)
      
      
      % 混合LM-牛顿法优化      
      %theta = rand(n2+n2choose2,1);
      theta = zeros(n2+n2choose2,1);
      P0 = optimizer_GPU_torch(theta,A2,A3,Oi,Vi,Wi,X,Y,alpha,alpha_Delta);

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
            
      time = toc; % 结束计时并输出所用时间
      fprintf("nod: %d/%d, time: %.4f(s)\n", nod, n, time);
  end
  total_time = toc;
  fprintf("total time: %.4f(s)\n", total_time);
end


%% 计算残差和雅可比矩阵
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

  % 雅可比计算
  I_N = eye(n2, 'gpuArray');
  I_C = eye(n2choose2, 'double');
  
  % 计算非线性项的导数
  D_S1 = kron(S1, I_N) + kron(I_N, S1);
  D_S2 = kron(gather(S2), I_C) + kron(I_C, gather(S2));
  
  % 雅可比矩阵
  J = [G1 + G3 * D_S1 + 2 * G4 * kron(I_N, S2), G2 + 2 * G4 * kron(S1, I_C) + double(gpuArray(gather(G5) * D_S2)); ...
      G1_Delta + G3_Delta * D_S1 + 2 * G4_Delta * kron(I_N, S2), G2_Delta + 2 * G4_Delta * kron(S1, I_C) + double(gpuArray(G5_Delta * D_S2))];
  J = gather(J);
end


%% LM优化器
function theta = optimizer_GPU_torch(theta,A2,A3,Oi,Vi,Wi,X,Y,alpha,alpha_Delta)
    nvars = sum(size(A2,2) + size(A3,2));        % 待求解变量维度（根据实际问题调整）
  % ------------------- 最小二乘 -------------------
  % 定义参数传递
  fun = @(x) compute_residual_torch(x,A2,A3,Oi,Vi,Wi,X,Y,alpha,alpha_Delta);  % 注意参数传递，尤其是x

  % 配置优化选项
  options_lsq = optimoptions('lsqnonlin', ...
    'Algorithm', 'trust-region-reflective', ...
    'SpecifyObjectiveGradient', true, ...  % 使用自定义雅可比
    'StepTolerance', 1e-8, ...                 % 放宽步长容差
    'FunctionTolerance', 1e-6, ...             % 放宽函数值容差
    'MaxIterations', 30);
     %'Display', 'iter', ...

  % 变量边界（强制变量在[0,1]区间）
  lb = zeros(nvars, 1);
  ub = ones(nvars, 1);

  [theta, resnorm] = lsqnonlin(fun, theta, lb, ub, options_lsq);

end


%% 计算残差和雅可比矩阵（直接计算版本）
function [R, J] = compute_residual_torch(theta,A2,A3,Oi,Vi,Wi,X,Y,alpha,alpha_Delta)    
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
    
    % 计算投影 (关键优化)
    proj_A2_S1 = A2 * S1;      % [m1, 1]
    proj_A3_S2 = A3 * S2;      % [m1, 1]
    temp2S2 = temp2_matrix * S2;  % [m1, 1]
    
    % 直接计算残差 (类似PyTorch方式)
    % 一阶部分
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
    
    % 雅可比矩阵计算 - 避免显式克罗内克积% diag有问题
    % J11: n2 × n2
    J11 = alpha * (Y .* A2)' * (Vi .* A2) +...
          alpha^2 * (Y .* A2)' * (Oi .* (A2 .* proj_A2_S1)) +...
          4 * alpha * alpha_Delta * (Y .* A2)' * (Oi .* (A2 .* temp2S2));  

    % J12: n2 × n2choose2 %计算维度和其他不匹配
    %J12 = alpha_Delta * (Y .* A2)' * (Vi .* A3) +...
    %      4 * alpha * alpha_Delta * (Y .* A2)' * (Oi .* (A2 .* temp2S2) .* A3) * gpuArray.diag(S1) +...
    %      alpha_Delta^2 * (Y .* A2)' * (Oi .* (temp2_matrix .* (temp2_matrix * (S2 + S2'))));
    J12 = alpha_Delta * (Y .* A2)' * (Vi .* A3) +...
          4 * alpha * alpha_Delta * (Y .* A2)' * (Oi .* (A3 .* proj_A2_S1)) +...  
          alpha_Delta^2 * (Y .* A2)' * (Oi .* (temp2_matrix .* temp2S2));

    % J21: n2choose2 × n2
    %J21 = alpha * (Y .* A3)' * (Vi .* A2) +...
    %      alpha^2 * (Y .* A3)' * (Oi .* (A2 .* (A2 * (S1 + S1')))) +...
    %      4 * alpha * alpha_Delta * (Y .* A3)' * (Oi .* (A2 .* temp2S2) .* A2) * gpuArray.diag(S2);
    J21 = alpha * (Y .* A3)' * (Vi .* A2) +...
          alpha^2 * (Y .* A3)' * (Oi .* (A2 .* proj_A2_S1)) +...
          4 * alpha * alpha_Delta * (Y .* A3)' * (Oi .* (A2 .* temp2S2));

    % J22: n2choose2 × n2choose2
    J22 = alpha_Delta * (Y .* A3)' * (Vi .* A3) +...
          4 * alpha * alpha_Delta * (Y .* A3)' * (Oi .* (A3 .* proj_A2_S1)) +...
          alpha_Delta^2 * (Y .* A3)' * (Oi .* (temp2_matrix .* temp2S2));

    J = gather([J11, J12; J21, J22]);
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
      % A2: Filtered node state matrix, \={s}^i(t), in Eq(4.38)

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
end
