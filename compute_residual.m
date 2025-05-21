%%
function [F, J]= compute_residual(theta,G1,G2,G3,G4,G5,G1_Delta,G2_Delta,G3_Delta,G4_Delta,G5_Delta,Y1_gpu,Y2_gpu)
  [~,n2]=size(G1);   % 有效时刻数量
  n2choose2 = nchoosek(n2,2);

  X1 = gpuArray(theta(1:n2));
  X2 = gpuArray(theta(n2+1:end));
  kron_X1X1 = kron(X1,X1);
  kron_X1X2 = kron(X1,X2);
  kron_X2X2 = kron(X2,X2);
  
  % 按行计算方程1残差，使用矩阵计算，大小会超过设备限制
  F1 = gather(G1*X1 + G2*X2 + G3*kron_X1X1 + 2*double(G4*kron_X1X2) + G5*kron_X2X2 - Y1_gpu);

  % 按行计算方程2残差（带Δ的项），使用矩阵计算，大小会超过设备限制
  F2 = gather(G1_Delta*X1 + G2_Delta*X2 + G3_Delta*kron_X1X1 + 2*double(G4_Delta*kron_X1X2) - Y2_gpu) + G5_Delta*gather(kron_X2X2);
  F = [F1; F2];

  I_N = eye(n2, 'gpuArray');
  I_C = eye(n2choose2, 'double');
  
  % 计算非线性项的导数
  D_X1 = kron(X1, I_N) + kron(I_N, X1);
  D_X2 = kron(gather(X2), I_C) + kron(I_C, gather(X2));
  
  % 雅可比分块
  J11 = G1 + G3 * D_X1 + 2 * double(G4 * kron(I_N, X2));
  J12 = G2 + 2 * double(G4 * kron(X1, I_C)) + gather(G5) * D_X2;

  J21 = G1_Delta + G3_Delta * D_X1 + 2 * double(G4_Delta * kron(I_N, X2));
  J22 = G2_Delta + 2 * double(G4_Delta * kron(X1, I_C)) + G5_Delta * D_X2;

  % 合并雅可比矩阵
  J = gather([J11, J12; J21, J22]);
end