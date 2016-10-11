% compute_mesh_statistics.m   July 14, 2015, modified for gpde Jan 15,  2016

function [min_min_side_lengths] = compute_mesh_statistics(pm_node,ele)

          
   c4n = pm_node;
   n4e = ele; 
   number_of_nodes    = size(c4n,1);
   number_of_elements = size(n4e,1);
   
   %% compute the minimum (over all triangles T) of the max side length
   %% of T.
   max_side_lengths = zeros(1,number_of_elements);
   for i=1:number_of_elements
      face_coords = c4n(n4e(i,:),:);
      
      vertex_a = face_coords(1,:);
      vertex_b = face_coords(2,:);
      vertex_c = face_coords(3,:);
      
      side_a = vertex_b - vertex_c;
      side_b = vertex_a - vertex_c;
      side_c = vertex_a - vertex_b;
   
      min_side_lengths(i) = min([norm(side_a),norm(side_b),norm(side_c)]);
   end
   
   min_min_side_lengths = min(min_side_lengths);
