function [ori_A_adj, P3_tensor] = Reconstruction_TS(UAU_state_nodes, SIS_state_nodes, Lambda)
    % Reconstruction network by Two-step method
    % (https://doi.org/10.1038/s41467-022-30706-9|www.nature.com/naturecommunications)
    % Input:
        % UAU_state_nodes: the node state matrix of the virtual layer(UAU), [T, n]
        % SIS_state_nodes: the node state matrix of the physical layer(SIS), [T, n]
        % Lambda: the regularization parameter
    % Output:
        % ori_A_adj: the reconstructed network two-body interaction, [n, n]
        % P3_tensor: the reconstructed network three-body interaction, [n, n, n]
    
    tic; % 开始计时
    [~, n]=size(UAU_state_nodes);  % 共m个时刻，n个节点
    ori_A_adj = zeros(n, n);
    P3_tensor = zeros(n, n, n);  % 创建一个n*n*n的三维张量，在确定阈值前需收集原始小数

    options1 = optimoptions('fsolve', 'Display', 'none');
    options2 = statset('Display', 'off');

    % 循环求解所有节点的一阶边和二阶边
    parfor nod = 1:n
        fprintf("nod: %d \n", nod);
        
        %%%step one
        % solve 两个gamma（即向量x）
        [X, Y, theta1, ~, A2, ~]=Extract(UAU_state_nodes, SIS_state_nodes, nod);
        x0 = 0.9999;
        x = fsolve(@(x) myfunTS(x,X,Y,theta1),x0, options1);  % solve TS.Eq(4.27), x=alpha^i
        % solve TS.Eq(4.37)
        M=x.^theta1;                                % instead a^(x_0) by TS.Eq(4.24)
        f=M./(1-M+eps)-M.*log(M)./((1-M+eps).^2);   % \={W}^i(t_m), in TS.Eq(4.31)
        g=M./((1-M+eps).^2);                        % \={V}^i(t_m), in TS.Eq(4.31)
        C=zeros(n, n);                              % A_1, in TS.Eq(4.37)
        D=zeros(n, 1);                              % Y_1, in TS.Eq(4.37)
        % calulate A_1 and Y_1
        for i=1:n  %前n行为l的取值
            %lambda1，前n列为j的取值
            C(i,1:n) = sum(Y.*A2(:,i).*g .* A2, 1);
            %zeta
            D(i)=sum((X-Y.*f).*A2(:,i));  % Y_1(i), in TS.Eq(4.37)
        end
        % 解CX=D，得P0=X1，是一个列向量
        % 预处理技术
        %M_poly = @(x) x - 0.5*C*x + 0.25*C^2*x;  % 最小二乘多项式
        C_normalized = (2 / max(abs(eig(C)))) * C - eye(size(C));  % 归一化矩阵
        M_poly = @(x) chebyshev_preconditioner(C_normalized, x, 5);  % 构造四阶切比雪夫多项式预处理器

        [TS_X1, flag] = gmres(C, D, 10, 1e-6, 200, M_poly);  % 使用GMRES求解
        %TS_X=lasso(C, D, 'Lambda', Lambda,'RelTol',10^-4,Options=options2);
        %TS_X1 = lsqminnorm(C,D,1e-12);  % OLS
        
        TS_X1 = -TS_X1;
        tru = fun_cut(TS_X1);  % Find the two-body truncation value
        neig = find(TS_X1>=tru);  % Get the neighbor subset of the reconstructed node
        UAU_state_nodes_neig = UAU_state_nodes(:, sort([nod, neig']));  % Extract the nodes state matrix of the neighbor subset
        SIS_state_nodes_neig = SIS_state_nodes(:, sort([nod, neig']));
        nod_index = find(sort([nod, neig'])==nod);  % index of reconstructed node in the compressed matrix

        %%%step two
        % solve 两个gamma（即向量x）
        [X,Y,theta1,theta2,A2,A3]=Extract(UAU_state_nodes_neig, SIS_state_nodes_neig, nod_index);
        x0=[0.9999, 0.9999];
        x = fsolve(@(x) myfun(x,X,Y,theta1,theta2),x0, options1);  % solve Eq(4.29), x=[alpha^i, alpha^i_{Delta}]
        if min(x)<=0
            disp('fsolve alpha <= 0;\n')
            continue
        end

        % solve equation CX=D
        M=x(1).^theta1.*x(2).^theta2;              % instead a^(x_0)*b^(y_0) by Eq(4.24)
        f=M./(1-M+eps)-M.*log(M)./((1-M+eps).^2);  % \={W}^i(t_m), in Eq(4.31)
        g=M./((1-M+eps).^2);                       % \={V}^i(t_m), in Eq(4.31)

        n2 = length(neig) + 1;
        C=zeros(n2*(n2+1)/2,n2*(n2+1)/2);  % [A_1, A_2; A_1^delta, A_2^delta], in Eq(4.37)
        D=zeros(n2*(n2+1)/2,1);            % [Y_1, Y_2], in Eq(4.37)
        
        % Upper C and D (first order edges)
        for i = 1:n2
            C(i,1:n2) = sum(Y.*A2(:,i).*g .* A2, 1);      % A_1(i,:), Eq(4.40)
            C(i,(n2+1):n2*(n2+1)/2) = sum(Y.*A2(:,i).*g .* A3, 1); % A_2(i,:), Eq(4.41)
            D(i) = sum((X-Y.*f).*A2(:,i));                % Y_1(i), Eq(4.38)
        end
        % Lower C and D (second order edges)
        for i = (n2+1):n2*(n2+1)/2
            C(i,1:n2) = sum(Y.*A3(:,i-n2).*g .* A2, 1);   % A_1^delta(i,:), Eq(4.42)
            C(i,(n2+1):n2*(n2+1)/2) = sum(Y.*A3(:,i-n2).*g .* A3, 1); % A_2^delta(i,:), Eq(4.43)
            D(i) = sum((X-Y.*f).*A3(:,i-n2));             % Y_2(i), Eq(4.39)
        end

        % 解CX=D，得P0=[P1 P2]'=[eta1 eta2]'是一个列向量
        % 预处理技术
        %M_poly = @(x) x - 0.5*C*x + 0.25*C^2*x;  % 最小二乘多项式
        C_normalized = (2 / max(abs(eig(C)))) * C - eye(size(C));  % 归一化矩阵
        M_poly = @(x) chebyshev_preconditioner(C_normalized, x, 5);  % 构造四阶切比雪夫多项式预处理器
        [P0, flag] = gmres(C, D, 10, 1e-6, 200, M_poly);  % 使用GMRES求解
        %P0 = lsqminnorm(C,D,1e-12);  % OLS
        %P0=lasso(C, D,'Lambda', Lambda,'RelTol',10^-4,Options=options2);
        
        % P1=eta1:该nod一阶边（含第nod个元素，n个元素组成的列向量）
        %P1=zeros(n,1);
        P1=P0(1:n2);
        % ori_A_adj:重构的网络邻接矩阵，元素为小数，不一定对称
        P5 = zeros(n,1);
        P5(sort([nod; neig])) = P1;
        ori_A_adj(:,nod) = P5;

        % P2=eta2:该nod二阶边（C_n^2个元素组成的列向量）
        %P2=zeros(n*(n-1)/2,1);
        P2=P0((n2+1):n2*(n2+1)/2);
        % P3:将列向量P2转换为矩阵存到严格下三角阵P3里
        P3=zeros(n2,n2);
        temp2 = find(tril(ones(n2, n2), -1));  %获取P2的严格下三角部分的线性索引
        P3(temp2) = P2;  %将P2中的元素按照列优先顺序填充到P3的严格下三角部分
        P4 = zeros(n,n);
        P4(sort([nod; neig]), sort([nod; neig])) = P3;
        P3_tensor(:, :, nod) = P4;  %将计算出的P3存储到三维张量中的对应切片
    end
    toc; % 结束计时并输出所用时间

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


%% Function myfun 用来求解两个alpha（即向量x） Eq(4.29)
function F=myfun(x,X,Y,theta1,theta2)
    F(1)=sum(Y.*theta1./(1-x(1).^theta1.*x(2).^theta2+eps)-(X+Y).*theta1);
    %F(1)=sum(Y.*theta1.*(x(1).^theta1.*x(2).^theta2./(1-x(1).^theta1.*x(2).^theta2)))-sum(X.*theta1);
    F(2)=sum(Y.*theta2./(1-x(1).^theta1.*x(2).^theta2+eps)-(X+Y).*theta2);
end


%% Function myfunTS 两步法第一步中用来求解一个alpha（即向量x） TS.Eq(4.27)
function F=myfunTS(x,X,Y,theta1)
    F = sum(Y.*theta1./(1-x.^theta1)-(X+Y).*theta1);
end


%% Function Extract
function [X,Y,theta1,theta2,A2,A3]=Extract(SA,SB,nod)
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
    X = sparse(X);  Y = sparse(Y);  % 稀疏化

    theta1=sum(A2,2);  % \={theta}^i(t_m), in Eq(4.22)
    [m1,~]=size(A2);   % 有效时刻数量
    theta2=zeros(m1,1);
    for i=1:m1
        b=A2(i,:)'*A2(i,:);
        theta2(i,1)=sum(sum(b-diag(diag(b))))/2;  % \={theta}^i_{Delta}(t_m), in Eq(4.23)
    end

    A3=zeros(m1,n*(n-1)/2); % 求A3，即(\={s}^j * \={s}^k)
    for j=1:m1
        temp1=A2(j,:)'*A2(j,:);  % [n,n]
        A3(j,:)=(temp1(logical(tril(ones(size(temp1)),-1))))';
    end  %tril(*,-1)取严格下三角，从下三角矩阵到一行的转换通过列优先顺序实现
end
