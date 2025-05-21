function [ori_A_adj, P3_tensor] = Reconstruction_TStaylor3_GPU(UAU_state_nodes, SIS_state_nodes)
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
      %UAU_state_nodes_neig = UAU_state_nodes(:, sort([nod, neig']));
      %SIS_state_nodes_neig = SIS_state_nodes(:, sort([nod, neig']));
      %nod_index = find(sort([nod, neig']) == nod);  % index in compressed matrix

      %%%step two --- Taylor2
      % solve 两个gamma（即向量x）
      %[X2,Y2,A22]=Extract(UAU_state_nodes_neig, SIS_state_nodes_neig, nod_index);
      % 矩阵裁剪后X, Y并不改变，仅对A2进行裁切
      A2 = A2(:,union(nod, neig));
      theta1=sum(A2,2);  % \={theta}^i(t_m), in Eq(4.22)
      [m1,n2]=size(A2);   % m1:有效时刻数量，n2:TS2节点数量
      n2choose2 = nchoosek(n2,2); % X2维度

      theta2=gpuArray.zeros(m1,1,'double');
      % 使用单精度浮点替代，避免内存不足
      A3 = gpuArray.zeros(m1,n2choose2,'double');       % 求A3，即[\={s}^j * \={s}^k]_{m1 \times C_N^2}
      A4 = gpuArray.zeros(m1,n2^2,'double');            % [\={s}^j * \={s}^k]_{m1 \times N^2}, in TST2.Eq(4.) part of A3
      A5 = gpuArray.zeros(m1, n2*n2choose2, 'single');  % [\={s}^j * \={s}^p * \={s}^q]_{m1 \times N*C_N^2}, in TST2.Eq(4.) part of A4
      cutA6 = ceil(m1/2);
      A6_1 = zeros(cutA6,n2choose2^2,'single');    % [\={s}^j * \={s}^k * \={s}^p * \={s}^q]_{m1 \times C_N^2*C_N^2}, in TST2.Eq(4.) part of A5
      for j=1:m1
        A2j = A2(j,:);  % A2j:第j行
        temp1=A2j'*A2j;
        temp2=(temp1(logical(tril(ones(size(temp1)),-1))))';  %tril(*,-1)取严格下三角，从下三角矩阵到一行的转换通过列优先顺序实现
        A3(j,:)=temp2;
        A4(j,:)=kron(A2j,A2j);
        A5(j,:)=single(kron(A2j,temp2));
        %A6(j,:)=gather(single(kron(temp2,temp2)));

        theta2(j,1)=sum(sum(temp1-diag(diag(temp1))))/2;  % \={theta}^i_{Delta}(t_m), in Eq(4.23)
      end
      clear A2j temp1 temp2;
  
      x0=[0.9999, 0.9999];
      x = fsolve(@(x) myfun(x,X,Y,theta1,theta2),x0, options1);  % solve Eq(4.29), x=[alpha^i, alpha^i_{Delta}]
      if min(x)<=0
          disp('fsolve alpha <= 0;\n')
          continue
      end

      % solve equation GX=D
      M=x(1).^theta1.*x(2).^theta2;              % instead a^(x_0)*b^(y_0) by Eq(4.24)
      Oi = M.*(1+M)./(2.*(1-M).^3);                         % \={O}^i(t_m), [m1,1], in TST2.Eq(4.31)
      Vi = M./((1-M).^2)-2.*Oi.*log(M);                     % \={V}^i(t_m), [m1,1], in TST2.Eq(4.31)
      Wi = M./(1-M)-M.*log(M)./((1-M).^2)+Oi.*(log(M).^2);  % \={W}^i(t_m), [m1,1], in TST2.Eq(4.31)
      clear M theta1 theta2;

      % G上、D上，一阶边
      G1 = (Y .* A2)' * (Vi .* A2);          % n2 × n2
      G2 = (Y .* A2)' * (Vi .* A3);          % n2 × n2choose2
      G3 = (Y .* A2)' * (Oi .* A4);          % n2 × n2^2
      G4 = (Y .* A2)' * (Oi .* A5);          % n2 × n2*n2choose2
      %G5 = (Y .* A2)' * (Oi .* A6);          % n2 × n2choose2^2
      D1 = sum((X - Y .* Wi) .* A2, 1)';     % n2 × 1
      
      % G下、D下，二阶边
      G1_Delta = (Y .* A3)' * (Vi .* A2);    % n2choose2 × n2
      G2_Delta = (Y .* A3)' * (Vi .* A3);    % n2choose2 × n2choose2
      G3_Delta = (Y .* A3)' * (Oi .* A4);    % n2choose2 × n2^2
      G4_Delta = (Y .* A3)' * (Oi .* A5);    % n2choose2 × n2*n2choose2
      %G5_Delta = (Y .* A3)' * (Oi .* A6);    % n2choose2 × n2choose2^2
      D2 = sum((X - Y .* Wi) .* A3, 1)';     % n2choose2 × n2choose2^2

      clear X Vi Wi A4 A5 A6;
      

      % 块循环计算G5和G5_Delta，避免内存不足
      YOi = Y.*Oi;
      G5 = gpuArray.zeros(n2,n2choose2^2, 'double');
      
      % G上、D上，一阶边
      for i=1:n2  %前n行为l的取值
        %fprintf("term_G5 i=%d\n",i);
        ROLi = YOi.*A2(:,i);  % [m1,1]
        if nnz(ROLi) == 0
          continue
        end
    
        G5_i = zeros(1,n2choose2^2,'gpuArray');
        nnz_indices = find(ROLi ~= 0);  % 找到非零元素的索引 
        block_length = 300;  % 每个块的大小，不是越大越好，过大的块会超出GPU专用内存大小，导致性能降低
        num_blocks = ceil(length(nnz_indices) / block_length);  % 计算块的数量
        % 生成严格下三角掩码（n2×n2）
        mask = tril(true(n2), -1);
    
        for k = 1:num_blocks  % 循环每个块
          start_index = (k-1) * block_length + 1;  % 块的起始索引
          end_index = min(k * block_length, length(nnz_indices));  % 块的结束索引
          nnz_indices_block = nnz_indices(start_index:end_index);  % 当前块的非零元素索引
          current_block_size =  length(nnz_indices_block);  % 当前块的大小
    
          ROLi_block = ROLi(nnz_indices_block);  % 获取当前块的非零元素值
          A2_block = A2(nnz_indices_block,:);  % 获取当前块对应的A2行
          
          % 批量计算所有行的外积矩阵（n×n×m1）
          temp1 = reshape(A2_block', n2, 1, []) .* reshape(A2_block', 1, n2, []);
    
          % 提取所有外积矩阵的严格下三角元素（m1 × (n(n-1)/2)）
          temp2 = reshape(temp1, n2*n2, current_block_size);
          temp2 = temp2(mask(:), :)';
          temp2 = reshape(temp2, current_block_size, []);  % m1行，每行n(n-1)/2个元素
    
          % 计算克罗内克积（m1 × (n(n-1)/2)^2）
          temp2_3d = reshape(temp2', n2choose2, 1, current_block_size);
          temp2_3d_transpose = reshape(temp2', 1, n2choose2, current_block_size);
          A6_block = reshape(temp2_3d .* temp2_3d_transpose, n2choose2^2, current_block_size)';
          
          % 计算当前块的结果
          G5_i = G5_i + sum(ROLi_block .* A6_block,1);
        end
        G5(i,:) = G5_i;  % 将当前块的结果添加到G5中
      end 
      clear G5_i nnz_indices nnz_indices_block ROLi_block A2_block ROLi temp1 temp2 temp2_3d_transpose temp2_3d A6_block;

      % G下、D下，二阶边
      G5_Delta = zeros(n2choose2,n2choose2^2, 'double');
      for i=1:n2choose2  %后C_n^2行为l1、l2的取值    
        %fprintf("term_G5Delta i=%d\n",i);
        ROLLi = YOi.*A3(:,i);  % [m1,1]
        if nnz(ROLLi) == 0
          continue
        end
    
        %block function
        G5Delta_i = zeros(1,n2choose2^2,'gpuArray');
        nnz_indices = find(ROLLi ~= 0);  % 找到非零元素的索引 
        block_length = 300;  % 每个块的大小
        num_blocks = ceil(length(nnz_indices) / block_length);  % 计算块的数量
        % 生成严格下三角掩码（n2×n2）
        mask = tril(true(n2), -1);
    
        for k = 1:num_blocks  % 循环每个块
          start_index = (k-1) * block_length + 1;  % 块的起始索引
          end_index = min(k * block_length, length(nnz_indices));  % 块的结束索引
          nnz_indices_block = nnz_indices(start_index:end_index);  % 当前块的非零元素索引
          current_block_size =  length(nnz_indices_block);  % 当前块的大小
    
          ROLLi_block = ROLLi(nnz_indices_block);  % 获取当前块的非零元素值
          A2_block = A2(nnz_indices_block,:);  % 获取当前块对应的A2行
          
          % 批量计算所有行的外积矩阵(n2×n2×m1)
          temp1 = reshape(A2_block', n2, 1, []) .* reshape(A2_block', 1, n2, []);
    
          % 提取所有外积矩阵的严格下三角元素(m1 × n2choose2)
          temp2 = reshape(temp1, n2*n2, current_block_size);
          temp2 = temp2(mask(:), :)';
          temp2 = reshape(temp2, current_block_size, []);  % m1行，每行n2choose2个元素
    
          % 计算克罗内克积（m1 × n2choose2^2）
          temp2_3d = reshape(temp2', n2choose2, 1, current_block_size);
          temp2_3d_transpose = reshape(temp2', 1, n2choose2, current_block_size);
          A6_block = reshape(temp2_3d .* temp2_3d_transpose, n2choose2^2, current_block_size)';
          
          % 计算当前块的结果
          G5Delta_i = G5Delta_i + sum(ROLLi_block .* A6_block,1);
        end
        G5_Delta(i,:) = gather(G5Delta_i);
      end
      clear G5Delta_i nnz_indices nnz_indices_block ROLLi_block A2_block ROLLi temp1 temp2 temp2_3d_transpose temp2_3d A6_block YOi X Vi Wi A2 A3 A4 A5 Y;


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

      P0 = LMoptimizer_GPU(1000,2,1e-4,X0,G1,G2,G3,G4,G5,G1_Delta,G2_Delta,G3_Delta,G4_Delta,G5_Delta,D1,D2);

      % 调用fsolve函数
      %[P0, fval, exitflag, output] = fsolve(@(x) myfun2(x,G1,G2,G3,G4,G5,G1_Delta,G2_Delta,G3_Delta,G4_Delta,G5_Delta,D1,D2),X0, options1);  % solve Eq(4.29), x=[alpha^i, alpha^i_{Delta}]
      
      % P1=eta1:该nod一阶边（含第nod个元素，n个元素组成的列向量）
      %P1=zeros(n,1);
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
function theta = LMoptimizer_GPU(max_iter,lambda,tol,theta,G1,G2,G3,G4,G5,G1_Delta,G2_Delta,G3_Delta,G4_Delta,G5_Delta,Y1_gpu,Y2_gpu)

  best_error = inf;
  error_list = zeros(max_iter,1,'gpuArray');

  % 计算残差
  [R,J] = compute_residual(theta,G1,G2,G3,G4,G5,G1_Delta,G2_Delta,G3_Delta,G4_Delta,G5_Delta,Y1_gpu,Y2_gpu);
  current_error = norm(R)^2;
  
  for iter = 1:max_iter
    % 计算Hessian和梯度
    H = J' * J;
    grad = J' * R;
    
    % 添加阻尼因子
    H_reg = H + lambda * diag(diag(H));
    
    % 求解增量
    delta = -H_reg \ grad;
    
    % 更新参数
    theta_new = theta + delta;
    [R_new, J_new] = compute_residual(theta_new, A1, A2, A3, A4, A5, A1d, A2d, A3d, A4d, A5d);
    new_error = norm(R_new)^2;
    
    error_list(iter) = new_error;
    % 更新判断
    if new_error < current_error
        theta = theta_new;
        lambda = max(lambda/2, 1e-10);
        if abs(new_error - best_error) < tol
            break;
        end
        best_error = new_error;
        current_error = new_error;
        R = R_new;
        J = J_new;
    else
        lambda = min(lambda*2, 64);
    end
    
    fprintf('Iter %d: Error = %.6f, Lambda = %.2e\n', iter, new_error, lambda);
  end
end


%%
function [F, J]= compute_residual(theta,G1,G2,G3,G4,G5,G1_Delta,G2_Delta,G3_Delta,G4_Delta,G5_Delta,Y1_gpu,Y2_gpu)
  [~,n2]=size(G1);   % 有效时刻数量
  n2choose2 = nchoosek(n2,2);

  X1 = gpuArray(theta(1:n2));
  X2 = gpuArray(theta(n2+1:end));
  kron_X1X1 = kron(X1,X1);
  kron_X1X2 = kron(X1,X2);
  kron_X2X2 = kron(X2,X2);
  
  % 按行计算方程1残差和方程2残差（带Δ的项）
  F = [gather(G1*X1 + G2*X2 + G3*kron_X1X1 + 2*double(G4*kron_X1X2) + G5*kron_X2X2 - Y1_gpu); gather(G1_Delta*X1 + G2_Delta*X2 + G3_Delta*kron_X1X1 + 2*double(G4_Delta*kron_X1X2) - Y2_gpu) + G5_Delta*gather(kron_X2X2)];

  I_N = eye(n2, 'gpuArray');
  I_C = eye(n2choose2, 'double');
  
  % 计算非线性项的导数
  D_X1 = kron(X1, I_N) + kron(I_N, X1);
  D_X2 = kron(gather(X2), I_C) + kron(I_C, gather(X2));
  
  % 雅可比矩阵
  J = [gather(G1 + G3 * D_X1 + 2 * double(G4 * kron(I_N, X2))), gather(G2 + 2 * double(G4 * kron(X1, I_C))) + gather(G5) * D_X2; gather(G1_Delta + G3_Delta * D_X1 + 2 * double(G4_Delta * kron(I_N, X2))), gather(G2_Delta + 2 * double(G4_Delta * kron(X1, I_C))) + G5_Delta * D_X2];
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


%% Find the two-body neighbor subset !!!非常重要，尚有优化空间
function neig = find_neig(X, n)
    % X=[X1, X2, X3], (N+N+(N-1)*N/2)x1
    % neig: prossible two-body neighbor index

    X1 = X(1:n);
    %X2 = X(n+1:2*n);
    %X3 = X(2*n+1:end);

    %thresh1 = graythresh(X1);  % Otsu's Method, 最大类间方差法
    %thresh2 = graythresh(X2);  % Otsu's Method, 最大类间方差法
    tru1 = fun_cut(-X1);  % Find the two-body truncation value
    neig = find(X1<=tru1);  % Get the neighbor subset of the reconstructed node
    %tru2 = fun_cut(X2);  % Find the two-body truncation value
    %neig2 = find(X2>=tru2);  % Get the neighbor subset of the reconstructed node

    %neig = union(neig1, neig2);
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

%% Function myfun2
function F=myfun2(theta,G1,G2,G3,G4,G5,G1_Delta,G2_Delta,G3_Delta,G4_Delta,G5_Delta,Y1_gpu,Y2_gpu)
  [~,n2]=size(G1);   % 有效时刻数量

  X1 = gpuArray(theta(1:n2));
  X2 = gpuArray(theta(n2+1:end));
  kron_X1X1 = kron(X1,X1);
  kron_X1X2 = kron(X1,X2);
  kron_X2X2 = kron(X2,X2);
  
  % 按行计算方程1残差和方程2残差（带Δ的项）
  F = [gather(G1*X1 + G2*X2 + G3*kron_X1X1 + 2*double(G4*kron_X1X2) + G5*kron_X2X2 - Y1_gpu); gather(G1_Delta*X1 + G2_Delta*X2 + G3_Delta*kron_X1X1 + 2*double(G4_Delta*kron_X1X2) - Y2_gpu) + G5_Delta*gather(kron_X2X2)];
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
