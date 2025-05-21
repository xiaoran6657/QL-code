%% Function Extract
function [X,Y,A2]=Extract(SA,SB,nod)
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

  [m, ~]=size(SA); % [T, n]
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