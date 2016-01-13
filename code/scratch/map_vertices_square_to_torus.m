% map_vertices_square_to_torus.m  Jan 12, 2016
clear all; close all;
addpath('~/gpde/code/matlab_tutorial/3DSurfacePDEMatlab/src');

n = 2
% Generate the uniform mesh on the unit suqare
[ n_node,n_ele,node,ele] = triangulation_square( n )

A_sq  = zeros(n_node,n_node)
A_tor = zeros(n_node-1,n_node-1)


