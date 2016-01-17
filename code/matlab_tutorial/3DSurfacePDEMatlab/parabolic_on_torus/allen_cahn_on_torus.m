% Standard finite element method on the unique square
% We approximate the following equation
%
% where \Delta is the Laplace operator and du/dn is 
% the normal derivative on the boundary.
%
addpath('../src');
clear all; close all; clc;
format long;
n=50;
% Generate the uniform mesh on the unit suqare
%[ n_node,n_ele,node,ele] = triangulation_square( n );
[ n_node,n_ele,pm_node,ele,global_ind,global_ind_inverse] = triangulation_surface( n );

% Initialization
A    = sparse([],[],[],n_node,n_node,7*n_node);
MASS = sparse([],[],[],n_node,n_node,7*n_node);

% Since we are going to compute the integral on the reference element,
% we need provide the quadrature rule for the reference triangle.
% Meanwhile, we can also provide infomation of shape functions on the 
% reference element, i.e. function values (hat_phi) and function gradients
% (hat_phix and hat_phiy) on each quadrature points.

% Quadrature on reference element
nq=4;
% quadrature weights [nqx1]
q_weights= [1./24,1./24,1./24,9./24]';
% quadrature points [nq x 2]
q_yhat = [0,1,0,1./3;... % x components  
          0,0,1,1./3]';  % y components


%% Time Loop
current_soln = A;
h = compute_mesh_statistics(pm_node,ele);
init_cond = initial_value(pm_node(global_ind_inverse,:));
plot_from_node_ele(pm_node,ele,global_ind,global_ind_inverse,init_cond);

del_t = h^2/2;
%del_t = h/2;
tend = 100;

% shape value and shape gradient (x and y components) each are [nqx1]
[ hat_phi_at_q, hat_phix_at_q, hat_phiy_at_q ] = FEEVAL( q_yhat,nq );

% Assembling
for cell = 1:n_ele
    
    % Get local stiffness matrix and local rhs
    cell_ind = ele(cell,1:3);     % [1x3]
    vertices = pm_node(cell_ind, :); % [3x2]  
    [ local_stiff,local_rhs ] ...
        = local_assembling(vertices,...
                           hat_phi_at_q, hat_phix_at_q, hat_phiy_at_q,...
                           q_yhat,nq,q_weights,...
                           1,0,1); % alpha, beta, rhs_flag
     
	 %% Copy local to global
    %A(cell_ind,cell_ind) ...
    %    = A(cell_ind,cell_ind) + local_stiff;   %[3x3]
    %rhs(cell_ind) = rhs(cell_ind) + local_rhs;  %[3x1]
    cell_global_ind=global_ind(cell_ind);
    A(cell_global_ind,cell_global_ind) ...
           = A(cell_global_ind,cell_global_ind) + local_stiff; %[3x3]
end

%% L2 error computation
%exact_sol = exact(pm_node(global_ind_inverse,:));


%err_vec = exact_sol - solution;
% Assemble mass matrix
for cell = 1:n_ele
    
    % Local mass matrix
    cell_ind = ele(cell,1:3);     % [1x3]
    vertices = pm_node(cell_ind, :); % [3x2]
    [local_mass,~] = ...
        local_assembling( vertices,...
                          hat_phi_at_q, hat_phix_at_q, hat_phiy_at_q,...
                          q_yhat,nq,q_weights,...
                          0,1,0); % a, beta, rhs_flag
                      
    % copy local to global
    cell_global_ind=global_ind(cell_ind);

    MASS(cell_global_ind,cell_global_ind)...
        =MASS(cell_global_ind,cell_global_ind) + local_mass; %[3x3]
end

N = 100;
u_old = init_cond;
u_new = zeros(size(u_old));
figure(100)
plot_from_node_ele(pm_node,ele,global_ind,global_ind_inverse,init_cond);
title('initial cond')
drawnow

eps = 0.0001;
lmda = 1/eps^2;
P_fixed = MASS + del_t*(A + lmda*MASS);

for t = 0:del_t:N*del_t
   
   t=t
  
   %N_u2 = spdiags(diag(MASS).*u_old.^2,0, n_node,n_node);
   %N_u2 = MASS.*(u_old.*u_old);

   P = P_fixed - del_t*lmda*N_u2;
   u_new = P\(MASS*u_old);


   fig = figure(200);
   set(fig,'Position', [100, 100, 1700, 940])
   plot_from_node_ele(pm_node,ele,global_ind,global_ind_inverse,u_new);
   title(sprintf('solution at time %0.4f',t))
   drawnow
   
   fig2 = figure(300);
   u_max = max(abs(u_new))
   plot(t,u_max,'gx')
   hold on
   axis([0 0.5*N*del_t -2 2])
   drawnow
   
   u_old = u_new;

end
  
   %%  
%%  % print out the error
%%  l2_err = sqrt(transpose(err_vec)*MASS*err_vec)
%%  
%%  %% % Visualization
%%  %% % Using the function 'patch' to visualize each triangle.
%%  %% % Colors are decided by the value on the vertices.
%%  %% 
%%  %% Xnodes = chi_func_eval(pm_node(global_ind_inverse,:));
%%  %% 
%%  %% % change the element connectivity list to use the unique nodes of
%%  %% % global_ind instead of the repeated ones of ele and save as sele.
%%  %% sele=global_ind(ele);
%%  %% figure(1);
%%  %% axis([-2,2,-2,2,-2,2]); title('Solution'); colormap('default'); colorbar;
%%  %% for i=1:n_ele
%%  %%     XX=Xnodes(sele(i,:),1);
%%  %%     YY=Xnodes(sele(i,:),2);
%%  %%     ZZ=Xnodes(sele(i,:),3);
%%  %%     CC=solution(sele(i,:),1);
%%  %%     patch(XX,YY,ZZ,CC,'EdgeColor','interp');
%%  %% end
%%  %% 
%%  %% figure(2);
%%  %% axis([-2,2,-2,2,-2,2]); title('Error'); colormap('jet'); colorbar;
%%  %% for i=1:n_ele
%%  %%     XX=Xnodes(sele(i,:),1);
%%  %%     YY=Xnodes(sele(i,:),2);
%%  %%     ZZ=Xnodes(sele(i,:),3);
%%  %%     CC=err_vec(sele(i,:),1);
%%  %%     patch(XX,YY,ZZ,CC,'EdgeColor','interp');
%%  %% end
%%  
%%  
%%  
%%  %% plot the solution
%%  %figure(1)
%%  %plot_solution(n,solution);
%%  %title('solution');
%%  %xlabel('X'); ylabel('Y');
%%  %
%%  %% plot the error
%%  %figure(2)
%%  %plot_solution(n,err_vec);
%%  %title('error');
%%  %xlabel('X'); ylabel('Y');
