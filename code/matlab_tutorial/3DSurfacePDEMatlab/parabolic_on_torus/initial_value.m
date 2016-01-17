% initial_value.m

function [u_0 ] = initial_value(x)

   x = chi_func_eval(x);
   [n,~] = size(x);
   u_0(:,1) = 0.001*(0.5 - rand(1,n));
   %u_0 = zeros(n,1);
   %u_0(:,1) = 0.005*sin(97*pi*x(:,3)).*cos(206*atan2(x(:,1),x(:,2)));
   %u_0(:,1) = ;
