#create a mesh
cubit.cmd("reset")

#the "default names" comand will assign a name string to each 
#geometry entity at time of creation.  Your geometry file 
#should names it in already so we won't need to do this. I 
#did this just for the purpose of having names on the geometry.
cubit.cmd("set default names on")
cubit.cmd("brick x 1")

cubit.cmd("volume 1 int 3")
cubit.cmd("mesh volume 1")

#put the volume in a block so you successfully call get_block_elements_and_nodes
cubit.cmd("block 1 volume all")

#get the mesh
node_list= cubit.vectori()
sphere_list=cubit.vectori()
edge_list=cubit.vectori()
tri_list=cubit.vectori()
face_list=cubit.vectori()
pyramid_list=cubit.vectori()
wedge_list=cubit.vectori()
tet_list=cubit.vectori()
hex_list=cubit.vectori()
cubit.get_block_elements_and_nodes( 1, node_list, sphere_list, edge_list, tri_list, face_list, pyramid_list, wedge_list, tet_list, hex_list )

f=open('mesh_file.txt', 'w')

#write out the nodes 
tmp_str = "Number of Nodes = "+str( len(node_list) )+'\n'
f.write(tmp_str)
for tmp_node in node_list:
  my_coords=cubit.get_nodal_coordinates(tmp_node)
  tmp_str = str(tmp_node)+" "+str(my_coords[0])+" "+str(my_coords[1])+" "+str(my_coords[2])+'\n'
  #print tmp_str
  f.write(tmp_str)

#write out vertex associativity
all_verts = cubit.get_entities("vertex")
f.write("\n")
f.write("Vertex Associtivity\n")
tmp_str = "Number of Vertices = "+str( len(all_verts) )+'\n'
f.write(tmp_str)
#print tmp_str
for tmp_id in all_verts:
  tmp_name = cubit.get_entity_name("vertex", tmp_id )
  node_id = cubit.get_vertex_node(tmp_id)
  tmp_str = tmp_name+" "+str(node_id)+'\n'  
  #print tmp_str
  f.write(tmp_str)

#write out curve associativity
all_curves = cubit.get_entities("curve")
f.write("\n")
f.write("Curve Associtivity\n")
tmp_str = "Number of Curves = "+str( len(all_curves) )+'\n'
f.write(tmp_str)
#print tmp_str
for tmp_id in all_curves:
  tmp_name = cubit.get_entity_name("curve", tmp_id )
  edge_ids = cubit.get_curve_edges(tmp_id)
  tmp_str = tmp_name
  for edge_id in edge_ids:
    tmp_nodes = cubit.get_connectivity("edge", edge_id)
    #print tmp_nodes
    tmp_str += " "+str(tmp_nodes[0])+" "+str(tmp_nodes[1])
    #print tmp_str
  tmp_str += '\n'
  f.write(tmp_str)

#write out surface associativity
all_surfaces = cubit.get_entities("surface")
f.write("\n")
f.write("Surface Associtivity\n")
tmp_str = "Number of Surfaces = "+str( len(all_surfaces) )+'\n'
f.write(tmp_str)
#print tmp_str
for tmp_id in all_surfaces:
  tmp_name = cubit.get_entity_name("surface", tmp_id )
  face_ids = cubit.get_surface_quads(tmp_id)
  tmp_str = tmp_name
  for face_id in face_ids:
    tmp_nodes = cubit.get_connectivity("face", face_id)
    #print tmp_nodes
    tmp_str += " "+str(tmp_nodes[0])+" "+str(tmp_nodes[1])+" "+str(tmp_nodes[2])+" "+str(tmp_nodes[3])
    #print tmp_str
  tmp_str += '\n'
  f.write(tmp_str)

#write out volume associativity
all_volumes = cubit.get_entities("volume")
f.write("\n")
f.write("Volume Associtivity\n")
tmp_str = "Number of Volumes = "+str( len(all_volumes) )+'\n'
f.write(tmp_str)
#print tmp_str
for tmp_id in all_volumes:
  tmp_name = cubit.get_entity_name("volume", tmp_id )
  hex_ids = cubit.get_volume_hexes(tmp_id)
  tmp_str = tmp_name
  for hex_id in hex_ids:
    tmp_nodes = cubit.get_connectivity("hex", hex_id)
    #print tmp_nodes
    for node_id in tmp_nodes:
      tmp_str += " "+str(node_id)
    #print tmp_str
  tmp_str += '\n'
  f.write(tmp_str)
f.close()
