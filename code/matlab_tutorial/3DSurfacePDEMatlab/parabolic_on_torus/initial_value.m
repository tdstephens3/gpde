% initial_value.m

function [u_0 ] = initial_value(x)

   x = chi_func_eval(x);
   [n,~] = size(x);
   u_0 = zeros(n,1);
   u_0(:,1) = sin(3*pi*x(:,3)).*cos(2*atan2(x(:,1),x(:,2)));
