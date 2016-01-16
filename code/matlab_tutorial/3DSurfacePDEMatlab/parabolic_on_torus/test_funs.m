% test_funs.m
clear all; close all


n=20;
[ n_node,n_ele,node,ele] = triangulation_square( n );

x = chi_func_eval(node);
x1 = x(:,1);
x2 = x(:,2);
x3 = x(:,3);
plot3(x1,x2,x3)
