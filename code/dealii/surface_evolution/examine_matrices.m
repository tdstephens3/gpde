% examine_matrices.m
clear all

fid_A = fopen('data/A.txt', 'r');
fid_B = fopen('data/B.txt', 'r');
fid_C = fopen('data/C.txt', 'r');
fid_D = fopen('data/D.txt', 'r');
fid_rhs0 = fopen('data/rhs0.txt', 'r');
fid_rhs1 = fopen('data/rhs1.txt', 'r');
A = fscanf(fid_A, '%f ');
B = fscanf(fid_B, '%f ');
C = fscanf(fid_C, '%f ');
D = fscanf(fid_D, '%f ');
rhs0 = fscanf(fid_rhs0, '%f ');
rhs1 = fscanf(fid_rhs1, '%f ');
fclose(fid_A);
fclose(fid_B);
fclose(fid_C);
fclose(fid_D);
fclose(fid_rhs0);
fclose(fid_rhs1);

ln = sqrt(length(A))
A = reshape(A, ln, ln);
B = reshape(B, ln, ln);
C = reshape(C, ln, ln);
D = reshape(D, ln, ln);
rhs0 = reshape(rhs0, ln,1);
rhs1 = reshape(rhs1, ln,1);


M = [A B; C D];
M = sparse(M);
rhs = [rhs0;rhs1];
rhs = sparse(rhs);

tic
VH = M\rhs;
toc

%
%figure(010)
%plot(VH)
%title('VH')

invD = inv(A);
S = A - B*invD*C;
S = sparse(S);
rhs_schur = sparse(rhs0 - B*invD*rhs1);


tic
V = S\rhs_schur;
t1 = toc;

rhs1_minus_CV = sparse(rhs1 - C*V);

tic
H = D\rhs1_minus_CV;
t2 = toc;

elapsed_time = t1-t2


%
figure(010)
VH_s = [V;H];
diff = VH_s - VH;
plot(diff)
title('diff between schur and non schur')
%
figure(100)
subplot(1,2,1)
spy(M)
title('[A B; C D]')
%
subplot(1,2,2)
spy(S)
title('S')
%






%figure(200)
%spy(S)
%[~,p] = chol(S)
%title('S')


%m = reshape(m, 3, length(m)/3)';
%
%M = load('data/M.txt');
%spy(M)

