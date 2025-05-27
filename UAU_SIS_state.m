%% 生成状态时间序列
function [UAU_state_nodes,SIS_state_nodes]=UAU_SIS_state(A1,A2,lambda1,lambda2,delta,B,beta1,beta2,mu,T,rhoa,rhob)
    %Input:
    %  A1: adjacency matrices of virtual layer's network
    %  A2: triples in virtual layer's networks
    %  lambda1：Virtual layer's probability of informed between two-body
    %  lambda2：Virtual layer's probability of informed between three-body
    %  delta: Virtual layer's recovery rate
    
    %  B: adjacency matrices of Physical layer's network
    %  beta1: Physical layer's probability of infection in the U-state
    %  beta2: Physical layer's probability of infection in the A-state
    %  mu: Physical layer's recovery rate
    
    %  T: Time series length
    %  rhoa: Virtual layer's initial inform node density
    %  rhob: Physical layer's initial infection node density
    
    %Output:
    %  state_nodes: State matrix
    
    
    % 基本准备
    n = length(A1);  % Number of nodes
    UAU_state_nodes = zeros(T, n);  % Matrix to record states over time for UAU layer
    SIS_state_nodes = zeros(T, n);  % Matrix to record states over time for SIS layer
    UAU_tri_num = size(A2,1);  % Number of triangles in the UAU layer
    C = zeros(UAU_tri_num, 3);  % Triplet state matrix
    
    % 初始化第一个时刻的状态
    UAU_state = zeros(1, n);
    SIS_state = zeros(1, n);
    UAU_state(randperm(n, ceil(rhoa * n))) = 1;  % Randomly assign initial A states
    SIS_state(randperm(n, ceil(rhob * n))) = 1;  % Randomly assign initial I states  
    indices = (SIS_state == 1) & (UAU_state == 0);  % 条件索引找到所有SIS层为I状态且UAU层为U状态的节点
    UAU_state(indices) = 1;  % 直接将这些节点在UAU层的状态更新为A


    % 开始每个时刻的状态更新
    t=1;
    while t <= T
        UAU_state0=UAU_state;
        SIS_state0=SIS_state;   % Intermediate variable
    
        %%% Virtual layer's two-body inform process %%%
        A_node=find(UAU_state0==1);  % 找到当前所有A点，返回索引即下标，如[2 6 10]
        A_num=length(A_node);
        for i=1:A_num
            A_neig=find(A1(A_node(i),:)==1);  % 当前A点的所有邻居
            A_neig_U=A_neig(UAU_state0(A_neig)==0);  % 进一步过滤，只留下它的U邻居
            UAU_state(A_neig_U(rand(1,length(A_neig_U))<lambda1))=1;  % 当前A点只以概率lambda1感染其U邻居
        end


        %%% Virtual layer's three-body inform process %%%
        for j=1:UAU_tri_num  % 将三角形内的节点当前状态记在同等大小的矩阵C里
            C(j,1)=UAU_state0(A2(j,1));
            C(j,2)=UAU_state0(A2(j,2));
            C(j,3)=UAU_state0(A2(j,3));
        end
        % UAU_tri_inf=A2(find(sum(C,2)==2),:);  % Triples satisfying inform conditions
        UAU_tri_inf=A2(sum(C,2)==2,:);  % 记录当前时刻有传播可能的三角形Triples satisfying inform conditions
        UAU_tri_inf_num=size(UAU_tri_inf,1);  % 返回行的数量，即满足传播条件的三角形的个数
        if UAU_tri_inf_num>0
            tri_U=zeros(1,UAU_tri_inf_num);
            for h=1:UAU_tri_inf_num
                % tri_U(h)=UAU_tri_inf(h,find(UAU_state0(UAU_tri_inf(h,:))==0));%把有可能感染的三角形里，有可能被感染的点序号记下来
                tri_U(h)=UAU_tri_inf(h,UAU_state0(UAU_tri_inf(h,:))==0);  % 把当前时刻有可能感染的三角形里，有可能被感染的点（U点）序号记下来
            end
            UAU_state(tri_U(rand(1,UAU_tri_inf_num)<lambda2))=1;  % 对找到的三角形里的U点进行概率为lambda2的感染
        end


        %%% Virtual layer's recovery process %%%
        UAU_state(A_node(rand(1,length(A_node))<delta))=0;  % 只对当前时刻初始时的A点以概率delta恢复为U点


        %%% Physical layer's infection process %%%
        I_node=find(SIS_state0==1);
        I_num=length(I_node);
        for i=1:I_num
            I_neig=find(B(I_node(i),:)==1);  % 当前I点的所有邻居
            I_neig_S=I_neig(SIS_state0(I_neig)==0);  % 进一步过滤，只留下它的S邻居
            I_neig_S_U=I_neig_S(UAU_state0(I_neig_S)==0);  % I点的S邻居中，上层状态为U的
            I_neig_S_A=I_neig_S(UAU_state0(I_neig_S)==1);  % I点的S邻居中，上层状态为A的
            SIS_state(I_neig_S_U(rand(1,length(I_neig_S_U))<beta1))=1;
            SIS_state(I_neig_S_A(rand(1,length(I_neig_S_A))<beta2))=1;
        end


        %%% Physical layer's recovery process %%%
            SIS_state(I_node(rand(1,length(I_node))<mu))=0;


        %%% 最后检查：UI to AI %%%
            temp = (SIS_state == 1) & (UAU_state == 0);  % 条件索引找到所有SIS层为I状态且UAU层为U状态的节点
            UAU_state(temp) = 1; % 直接将这些节点在UAU层的状态更新为A



        %%% Check for synchronization and adjust states if necessary
        if all(UAU_state == 0) || all(UAU_state == 1)
            UAU_state(randperm(n, 2)) = 1 - UAU_state(randperm(n, 2));
        end
        if all(SIS_state == 0) || all(SIS_state == 1)
            SIS_state(randperm(n, 2)) = 1 - SIS_state(randperm(n, 2));
        end
    

        %%% Record states
        UAU_state_nodes(t,:)=UAU_state;
        SIS_state_nodes(t,:)=SIS_state;

        t=t+1;
    end
end