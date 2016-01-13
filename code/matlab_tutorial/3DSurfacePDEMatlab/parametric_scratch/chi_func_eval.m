% chi_func_eval.m	Jan 13, 2016

%% torus parameterization, R and r hard coded.
%% input: Y = [theta,phi]
%%        where theta and phi are column vectors, 
%%        and (theta_k,phi_k) \subset  [-pi,pi] x [-pi.pi]

function [x] = chi_func_eval(Y)

   R = 1.0;
	r = 0.6;	
	
	x(:,1) = (R + r*cos(Y(:,1))).*cos(Y(:,2));
	x(:,2) = (R + r*cos(Y(:,1))).*sin(Y(:,2));
	x(:,3) = r*sin(Y(:,1));
		
end
