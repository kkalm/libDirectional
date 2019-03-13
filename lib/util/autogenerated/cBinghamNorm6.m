function expression = cBinghamNorm6(in1)
%CBINGHAMNORM6
%    EXPRESSION = CBINGHAMNORM6(IN1)

%    This function was generated by the Symbolic Math Toolbox version 8.1.
%    13-Mar-2019 17:55:23

x1 = in1(1,:);
x2 = in1(2,:);
x3 = in1(3,:);
x4 = in1(4,:);
x5 = in1(5,:);
x6 = in1(6,:);
t2 = pi.^2;
t3 = t2.^2;
t4 = x1-x2;
t5 = 1.0./t4;
t6 = x1-x3;
t7 = 1.0./t6;
t8 = x2-x3;
t9 = 1.0./t8;
t10 = x1-x4;
t11 = 1.0./t10;
t12 = x2-x4;
t13 = 1.0./t12;
t14 = x3-x4;
t15 = 1.0./t14;
t16 = x1-x5;
t17 = 1.0./t16;
t18 = x2-x5;
t19 = 1.0./t18;
t20 = x3-x5;
t21 = 1.0./t20;
t22 = x4-x5;
t23 = 1.0./t22;
t24 = x1-x6;
t25 = 1.0./t24;
t26 = x2-x6;
t27 = 1.0./t26;
t28 = x3-x6;
t29 = 1.0./t28;
t30 = x4-x6;
t31 = 1.0./t30;
t32 = x5-x6;
t33 = 1.0./t32;
expression = t2.*t3.*(t5.*t7.*t11.*t17.*t25.*exp(x1)-t5.*t9.*t13.*t19.*t27.*exp(x2)+t7.*t9.*t15.*t21.*t29.*exp(x3)-t11.*t13.*t15.*t23.*t31.*exp(x4)+t17.*t19.*t21.*t23.*t33.*exp(x5)-t25.*t27.*t29.*t31.*t33.*exp(x6)).*2.0;
