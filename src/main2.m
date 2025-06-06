clear,clc;
rng(12)

%%
pathname1 = '.\data\';  % 单纯复形
pathname2 = '.\redata\';


Timespan_list = 1:20;
Timespan_list = Timespan_list*10000;

filename = "ERm200000n100ka18kb3.mat";
load(strcat(pathname1, filename));

recon_state = repmat(struct('Timespan', [], 'ori_A_adj',[], 'P3_tensor',[]), 1, size(Timespan_list,2));  % 预分配结构体

% Transfer input data to GPU if available
if gpuDeviceCount > 0
  UAU_state_nodes = gpuArray(UAU_state_nodes);
  SIS_state_nodes = gpuArray(SIS_state_nodes);
end


for idx = 1:size(Timespan_list,2)
  if recon_state(idx).Timespan == Timespan_list(idx)
      fprintf("m%d is reconstructed!\n", Timespan_list(idx));
      continue
  end
  
  tic;
  if idx < 5
      parpool(10);
  else
      parpool(2);
  end

  UAU_state_nodes_copy = UAU_state_nodes(1:Timespan_list(idx), :);
  SIS_state_nodes_copy = SIS_state_nodes(1:Timespan_list(idx), :);

  [ori_A_adj, P3_tensor] = Reconstruction_TStaylor7_GPU(UAU_state_nodes_copy, SIS_state_nodes_copy);
  
  recon_state(idx).Timespan = Timespan_list(idx);
  recon_state(idx).ori_A_adj = ori_A_adj;
  recon_state(idx).P3_tensor = P3_tensor;

  delete(gcp('nocreate'))
  time = toc;
  fprintf("\n m%d, time: %.4f(s)\n", Timespan_list(idx), time);
end


save(strcat(pathname2, 'TST12', '_', filename), 'A1', 'A2', 'B', 'recon_state');






%{

pathname2 = '..\data2\';  % 超图
pathname4 = '..\redata2\';

% 获取指定文件夹下的所有 .mat 文件
files = dir(fullfile(pathname1, '*.mat'));
% 提取文件名
fileNames = {files.name};
networkType = 'ER';  % 选定网络类型
Lambda = 1e-3;  % lasso parameter


%% Reconstruction Network
for i = 1:length(fileNames)
    filename = fileNames{i};

    result = strsplit(filename, '_');
    % Extraction networks parameters
    numbers = regexp(result{3}, '\d+', 'match');
    nNodes = str2double(numbers);
    %numbers = regexp(result{5}, '\d+', 'match');
    %kB = str2double(numbers);

    %matches = regexp(filename, 'm(\d+)', 'tokens');
    %numbers = cellfun(@(x) str2double(x{1}), matches); % 将捕获的数字字符串转换为数值
    %if startsWith(filename, networkType) && endsWith(filename, strcat(num2str(numTriangles), '.mat'))
    if startsWith(filename, networkType) && nNodes==100
        disp(filename)
        load(strcat(pathname1, filename));
        tic;
        [ori_A_adj, P3_tensor] = Reconstruction_TStaylor6_GPU(UAU_state_nodes, SIS_state_nodes);
        total_time = toc;
        fprintf("total time: %.4f(s)\n", total_time);
        save(strcat(pathname3, 'TSL', num2str(Lambda), '_', filename), 'A1', 'A2', 'B', 'ori_A_adj', 'P3_tensor');
    end
end
%}