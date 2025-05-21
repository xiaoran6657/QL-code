%%
function F = compute_residual_full(theta,Y,A2,A3,G1,G2,G1_Delta,G2_Delta,Oi,Y1_gpu,Y2_gpu)
  %%% 完全展开
  
  [m1,n2]=size(A2);   % 有效时刻数量
  n2choose2 = nchoosek(n2,2);

  X1 = gpuArray(theta(1:n2));
  X2 = gpuArray(theta(n2+1:end));
  kron_X1X1 = kron(X1,X1);
  kron_X1X2 = kron(X1,X2);
  kron_X2X2 = kron(X2,X2);
  YOi = Y.*Oi;

  % 按行计算方程1残差，使用矩阵计算，大小会超过设备限制
  F1 = zeros(n2,1,'gpuArray');
  for i=1:n2  %前n行为l的取值
    ROLi = YOi.*A2(:,i);  % [m1,1]
    if nnz(ROLi) == 0
      F1(i) = G1(i,:)*X1 + G2(i,:)*X2 - Y1_gpu(i);
      continue
    end

    G3_i = zeros(1,n2^2,'gpuArray');
    G4_i = zeros(1,n2*n2choose2,'gpuArray');
    G5_i = zeros(1,n2choose2^2,'gpuArray');

    nnz_indices = find(ROLi ~= 0);  % 找到非零元素的索引    
    for k = 1:length(nnz_indices)  % 循环每个索引
      j = nnz_indices(k);  % 获取非零元素的索引
      ROLij = ROLi(j);  % 获取非零元素的值

      A2j = A2(j,:);
      temp1=A2j'*A2j;
      temp2=(temp1(logical(tril(ones(size(temp1)),-1))))';  %tril(*,-1)取严格下三角，从下三角矩阵到一行的转换通过列优先顺序实现
  
      G3_i = G3_i + ROLij.*kron(A2j,A2j);      % A_3(i,:), in TST2.Eq(4.41)
      G4_i = G4_i + ROLij.*kron(A2j,temp2);      % A_4(i,:), in TST2.Eq(4.41)
      G5_i = G5_i + ROLij.*kron(temp2,temp2);      % A_5(i,:), in TST2.Eq(4.41)
    end

    F1(i) = G1(i,:)*X1 + G2(i,:)*X2 + G3_i*kron_X1X1 + 2*G4_i*kron_X1X2+ G5_i*kron_X2X2 - Y1_gpu(i);
  end

  % 按行计算方程2残差（带Δ的项），使用矩阵计算，大小会超过设备限制
  F2 = zeros(n2choose2,1,'gpuArray');
  for i=1:n2choose2  %后C_n^2行为l1、l2的取值    
    ROLLi = YOi.*A3(:,i);  % [m1,1]
    if nnz(ROLLi) == 0
      F2(i) = G1_Delta(i,:)*X1 + G2_Delta(i,:)*X2 - Y2_gpu(i);
      continue
    end

    G3Delta_i = zeros(1,n2^2,'gpuArray');
    G4Delta_i = zeros(1,n2*n2choose2,'gpuArray');
    G5Delta_i = zeros(1,n2choose2^2,'gpuArray');

    nnz_indices = find(ROLLi ~= 0);  % 找到非零元素的索引    
    for k = 1:length(nnz_indices)  % 循环每个索引
      j = nnz_indices(k);  % 获取非零元素的索引
      ROLLij = ROLLi(j);  % 获取非零元素的值

      A2j = A2(j,:);
      temp1=A2j'*A2j;
      temp2=(temp1(logical(tril(ones(size(temp1)),-1))))';  %tril(*,-1)取严格下三角，从下三角矩阵到一行的转换通过列优先顺序实现
  
      G3Delta_i = G3Delta_i + ROLLij.*kron(A2j,A2j);      % A_3(i,:), in TST2.Eq(4.41)
      G4Delta_i = G4Delta_i + ROLLij.*kron(A2j,temp2);      % A_4(i,:), in TST2.Eq(4.41)
      G5Delta_i = G5Delta_i + ROLLij.*kron(temp2,temp2);      % A_5(i,:), in TST2.Eq(4.41)
    end
    
    F2(i) = G1_Delta(i,:)*X1 + G2_Delta(i,:)*X2 + G3Delta_i*kron_X1X1 + 2*G4Delta_i*kron_X1X2 + G5Delta_i*kron_X2X2 - Y2_gpu(i);
  end

  F = gather([F1; F2]);
end