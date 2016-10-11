% grad_chi_at_q.m	Jan 13, 2016

%% gradient of torus parameterization, R and r hard coded.
%% input: Y = [theta,phi]

function [grad_chi] = grad_chi_at_q(Y)

   R = 1.0;
	r = 0.6;	
	
	theta = Y(1);
	phi   = Y(2);
   
   grad_chi = [-r*sin(theta)*cos(phi), -R*sin(phi)-r*sin(phi)*cos(theta);...
               -r*sin(theta)*sin(phi),  R*cos(phi) + r*cos(theta)*cos(phi);...
                r*cos(theta),           0];
			
end
