function F=myfunTS(x,X,Y,theta1)
  F = sum(Y.*theta1./(1-x.^theta1)-(X+Y).*theta1);
  F = gather(F);
end