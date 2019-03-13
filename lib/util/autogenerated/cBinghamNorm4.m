function expression = cBinghamNorm4(in1)
%CBINGHAMNORM4
%    EXPRESSION = CBINGHAMNORM4(IN1)

%    This function was generated by the Symbolic Math Toolbox version 8.1.
%    13-Mar-2019 17:55:20

x1 = in1(1,:);
x2 = in1(2,:);
x3 = in1(3,:);
x4 = in1(4,:);
t2 = pi.^2;
t3 = x1-x2;
t4 = 1.0./t3;
t5 = x1-x3;
t6 = 1.0./t5;
t7 = x2-x3;
t8 = 1.0./t7;
t9 = x1-x4;
t10 = 1.0./t9;
t11 = x2-x4;
t12 = 1.0./t11;
t13 = x3-x4;
t14 = 1.0./t13;
expression = t2.^2.*(t4.*t6.*t10.*exp(x1)-t4.*t8.*t12.*exp(x2)+t6.*t8.*t14.*exp(x3)-t10.*t12.*t14.*exp(x4)).*2.0;
