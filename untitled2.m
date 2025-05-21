% Minimize
%   obj: 1 x1 + 2 x2 + 1 x3 + 1 x4
% Subject To
%  c0 : 1 x1 + 1 x2 + 2 x3 + 3 x4 >= 1
%  c1 : 1 x1        - 1 x3 + 6 x4 = 1
% Bounds
%   0 <= x1 <= 10
%   0 <= x2
%   0 <= x3
%   0 <= x4
% End

% build model
model.A = sparse([1 1 2 3; ...
                  1 0 -1 6]);
model.modelsense = 'Min';
model.rhs = [1 1];
model.obj = [1 2 1 1];
model.lhsorsense = ['>' '='];
model.varnames = {'x1' 'x2' 'x3' 'x4'};
model.constrnames = {'c0' 'c1'};
model.modelname = 'lp';
model.ub = [10 inf inf inf];

% optimize
result = mindopt(model);

for v=1:length(result.X)
    fprintf('%s %d\n', model.varnames{v}, result.X(v));
end
fprintf('Obj: %e\n', result.ObjVal);
