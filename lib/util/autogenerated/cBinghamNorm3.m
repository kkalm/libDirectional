function expression = cBinghamNorm3(in1)
%CBINGHAMNORM3
%    EXPRESSION = CBINGHAMNORM3(IN1)

%    This function was generated by the Symbolic Math Toolbox version 8.1.
%    13-Mar-2019 17:55:19

x1 = in1(1,:);
x2 = in1(2,:);
x3 = in1(3,:);
t2 = pi.^2;
t3 = x1-x2;
t4 = 1.0./t3;
t5 = x1-x3;
t6 = 1.0./t5;
t7 = x2-x3;
t8 = 1.0./t7;
expression = t2.*pi.*(t4.*t6.*exp(x1)-t4.*t8.*exp(x2)+t6.*t8.*exp(x3)).*2.0;
