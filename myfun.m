function F=myfun(x,X,Y,theta1,theta2)
  F(1)=sum(Y.*theta1./(1-x(1).^theta1.*x(2).^theta2+eps)-(X+Y).*theta1);
  %F(1)=sum(Y.*theta1.*(x(1).^theta1.*x(2).^theta2./(1-x(1).^theta1.*x(2).^theta2)))-sum(X.*theta1);
  F(2)=sum(Y.*theta2./(1-x(1).^theta1.*x(2).^theta2+eps)-(X+Y).*theta2);
  F = gather(F);
end