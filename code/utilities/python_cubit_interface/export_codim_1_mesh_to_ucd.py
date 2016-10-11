print("---------- start: --------------------\n")

###    Python script to be run as a 'Custom Tool' from the Trelis/Cubit GUI interface. 
###    Exports 3d vertex coordinates and quadrilateral face connectivity
###    information from a Trelis/Cubit mesh on codimension 1 surfaces in
###    what dealii (dealii.org) considers to be the .ucd format for codimension 1 surfaces:
###
###    <number of verts> <number of faces> 0 0 0
###    1 x1 y1 z1
###    2 x2 y2 z2
###    .
###    .
###    .
###    1 <material id> quad q1_v1 q1_v2 q1_v3 q1_v4
###    2 <material id> quad q2_v1 q2_v2 q2_v3 q2_v4 
###    .
###    .
###    .
###
###    where q1_v1 is the first of four vertices associated with quad 1.
###    For example, the codimension 1 surface described by the union of
###    faces of the unit cube
###
###                   4_____________________________________ 3
###                  /|                                    /|
###                 / |             (top) q1              / |
###                /  |                                  /  |
###               /   |                                 /   |
###              /    |                                /    |
###             /     |                               /     |
###           1/____________________________________2/      |
###           |       |                             |       |
###           |       |                             |       |
###           | (side)|                             | (side)|
###           |   q3  |                             |   q5  |
###           |       |                             |       |
###           |       |             (back) q4       |       |
###           |       7-----------------------------|-------8  
###           |      /                              |     / 
###           |     /                 (bottom) q2   |    /  
###           |    /                                |   /   
###           |   /                                 |  /      
###           |  /                                  | /      
###           | /    (front) q6                     |/       
###           6------------------------------------- 5
###
###     has output:
###
###    8 6 0 0 0
###    1 1 0 1
###    2 1 1 1
###    3 0 1 1
###    4 0 0 1
###    5 1 1 0
###    6 1 0 0 
###    7 0 0 0
###    8 0 1 0
###    1 1 quad 1 2 3 4
###    2 1 quad 5 6 7 8
###    3 1 quad 4 7 6 1
###    4 1 quad 3 8 7 4
###    5 1 quad 2 5 8 3
###    6 1 quad 1 6 5 2
###
###    where the <material_id> parameter is the same on all faces, and has been set (abitrarily) to 1



material_id = 1

#output_filename = "/home/tds3/gpde/code/dealii/sandbox/load_mesh/ellipsoid_mesh.ucd"
#output_filename = "/home/tds3/gpde/code/dealii/sandbox/load_mesh/sphere_mesh.ucd"
output_filename = "/home/tds3/gpde/code/dealii/sandbox/load_mesh/brick_mesh.ucd"


# get geometric entities: nodes(vertices) and quads(faces)
nodes = cubit.get_entities("node")
len_nodes = len(nodes)

surfaces = cubit.get_entities("surface")
quads = []
for surface_id in surfaces:
    surface_quads = cubit.get_surface_quads(surface_id)
    quads.extend(surface_quads) 
len_quads = len(quads)

print("nodes: %d, quads %d" %(len_nodes,len_quads))

output_file = open(output_filename,"w")
header = "%d %d 0 0 0\n" %(len_nodes,len_quads)
output_file.write(header)

node_mapping = {} 
for i,node_id in enumerate(nodes):
    coords = cubit.get_nodal_coordinates(node_id)
    coords_str = "%d %0.12f %0.12f %0.12f\n" %(i+1,coords[0],
                                                   coords[1],
                                                   coords[2])
    # map Cubit's node_id's to a more 'natural' enumeration
    node_mapping[node_id] = i+1
    #print(coords_str)
    output_file.write(coords_str)


for j,quad_id in enumerate(quads):
    cubit_connectivity = cubit.get_connectivity("Quad",quad_id)
    # connectivity information comes in the form of Cubit's node_id's,
    # but we want to point back to the enumeration above
    connectivity = (node_mapping[cubit_connectivity[0]],
                    node_mapping[cubit_connectivity[1]], 
                    node_mapping[cubit_connectivity[2]], 
                    node_mapping[cubit_connectivity[3]])
    conn_str = "%d %d quad %d %d %d %d\n" %(j+1,material_id, connectivity[0],
                                                             connectivity[1],
                                                             connectivity[2],
                                                             connectivity[3])
    #print(conn_str)
    output_file.write(conn_str)



print("written %s" % (output_filename))
output_file.close()
print("----- done ---------")
