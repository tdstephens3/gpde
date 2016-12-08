# convert_mphtxt_quads_to_ucd.py, Dec 2016
import sys
## 
## read comsol .mphtxt exported mesh files into ucd
## that can be read by deal.ii code.

##
## input:  .mphtxt file from comsol 4.3a, and optional directory name
##         to write output to (see output)
## 
## output: this function does not return anything, but it writes two
##         files, one called c4n_mphtxt_name (mphtxt_name taken from
##         fileparts()) and one called n4e_mphtxt_filename
##         into the directory specified by varargin{1} (default write
##         directory is the present working directory).
##
## notes:  deal.ii wants is nodes in lexicographic ordering, see 
##          https://www.dealii.org/8.4.1/doxygen/deal.II/structGeometryInfo.html
## 
##   For example, for the face and vertices in the triangular mesh below,
##   we would have 
##                   1, 0,0,0
##                   2, 1,0,0
##                   3, 0,1,0
##                   4, 1,0,1
##                   
##                   n4e(1,:) = [1,2,3]
##                                    
##
##                  (0,1,0) *-----*  (1,0,1)
##                          |     |
##                          |     |
##                          |     |
##                          |     |
##                 (0,0,0)  *-----*  (1,0,0)
##
mphtxt_fname = sys.argv[1]

output_fname = sys.argv[2]
ucd_file     = open(output_fname,"w")

with open(mphtxt_fname,"r") as input_file:
    for line in input_file:
        while '# Mesh point coordinates'





#while ~feof(mphtxt_fid)
#
#%% Build c4n:
#   line = fgets(mphtxt_fid);
#   if strfind(line,'# Mesh point coordinates')
#      %blank = fgets(mphtxt_fid);
#      v = fgets(mphtxt_fid);
#      i=1;
#      while ~strcmp(v,newline)
#         c4n(i,:) = str2num(v);
#         i=i+1;
#         v = fgets(mphtxt_fid);
#      end
#
#%% Build n4e:
#%% looking for '3 # number of nodes per element'
#   elseif strfind(line,'3 # number of nodes per element')
#      blank = fgets(mphtxt_fid); blank = fgets(mphtxt_fid); %blank = fgets(mphtxt_fid);
#      n = fgets(mphtxt_fid);
#      j=1;
#      while ~strcmp(n,newline)
#         n4e(j,:) = str2num(n);
#         j=j+1;
#         n = fgets(mphtxt_fid);
#      end
# 
#   end
#end
#fclose(mphtxt_fid);
#
#n4e=n4e+1; % the values in n4e are integer indices into c4n.  Comsol
#           % uses 0-based indexing and Matlab uses 1-based indexing,
#           % so add 1 to each value in n4e.
# 
#c4n_fid = fopen(sprintf('%s/c4n_%s.txt',write_path,mphtxt_name),'w');
#for i=1:size(c4n,1)
#   fprintf(c4n_fid,'%0.12f %0.12f %0.12f \n',c4n(i,1),c4n(i,2),c4n(i,3));
#end
#fclose(c4n_fid);
#
#n4e_fid = fopen(sprintf('%s/n4e_%s.txt',write_path,mphtxt_name),'w');
#for i=1:size(n4e,1)
#   fprintf(n4e_fid,'%0.12f %0.12f %0.12f \n',n4e(i,1),n4e(i,2),n4e(i,3));
#end
#fclose(n4e_fid);
#
#
#
