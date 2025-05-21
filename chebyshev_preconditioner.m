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