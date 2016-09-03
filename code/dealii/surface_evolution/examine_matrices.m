% examine_matrices.m
clear all

fid_A = fopen('data/A.txt', 'r');
fid_B = fopen('data/B.txt', 'r');
fid_C = fopen('data/C.txt', 'r');
A = fscanf(fid_A, '%f ');
B = fscanf(fid_B, '%f ');
C = fscanf(fid_C, '%f ');
fclose(fid_A);
fclose(fid_B);
fclose(fid_C);

ln = sqrt(length(A))
A = reshape(A, ln, ln);
B = reshape(B, ln, ln);
C = reshape(C, ln, ln);


invD = inv(A);
S = A - B*invD*C;

figure(100)
subplot(2,2,1)
spy(A)
title('block(0,0)')
subplot(2,2,2)
spy(B)
title('block(0,1)')
subplot(2,2,3)
spy(C)
title('block(1,0)')

figure(200)
spy(S)
[~,p] = chol(S)
title('S')


%m = reshape(m, 3, length(m)/3)';
%
%M = load('data/M.txt');
%spy(M)

