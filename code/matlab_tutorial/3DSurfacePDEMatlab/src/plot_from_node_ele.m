% plot_from_node_ele.m Sept 18, 2015, modified Jan 15, 2016 for gpde

function plot_from_node_ele(node,ele,global_ind,global_ind_inverse,func)

% plot membrane from node and element matrices

n_ele = size(ele,1);

fig = figure(100);
set(fig,'Position', [100, 100, 1700, 940])

%% % Visualization
%% % Using the function 'patch' to visualize each triangle.
%% % Colors are decided by the value on the vertices.
%% 
Xnodes = chi_func_eval(node(global_ind_inverse,:));

% change the element connectivity list to use the unique nodes of
% global_ind instead of the repeated ones of ele and save as sele.
sele=global_ind(ele);
axis([-2,2,-2,2,-2,2]); title('Solution'); colormap('default'); colorbar;
for i=1:n_ele
    XX=Xnodes(sele(i,:),1);
    YY=Xnodes(sele(i,:),2);
    ZZ=Xnodes(sele(i,:),3);
    CC=func(sele(i,:),1);
    patch(XX,YY,ZZ,CC,'EdgeColor','k','facealpha',0.5);
end
axis equal
