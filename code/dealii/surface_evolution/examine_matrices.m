% examine_matrices.m
clear all

fid = fopen('data/M.txt', 'r');
M = fscanf(fid, '%f ');
fclose(fid);


sz = size(M)
ln = sqrt(length(M))
M = reshape(M, 4614, 4614);



%m = reshape(m, 3, length(m)/3)';
%
%M = load('data/M.txt');
%spy(M)

