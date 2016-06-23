/* ---------------------------------------------------------------------
 *
 * Copyright (C) 1999 - 2015 by the deal.II authors
 *
 * This file is part of the deal.II library.
 *
 * The deal.II library is free software; you can use it, redistribute
 * it, and/or modify it under the terms of the GNU Lesser General
 * Public License as published by the Free Software Foundation; either
 * version 2.1 of the License, or (at your option) any later version.
 * The full text of the license can be found in the file LICENSE at
 * the top level of the deal.II distribution.
 *
 * ---------------------------------------------------------------------

 */

// @sect3{Include files}

// The most fundamental class in the library is the Triangulation class, which
// is declared here:
#include <deal.II/grid/tria.h>
// We need the following two includes for loops over cells and/or faces:
#include <deal.II/grid/tria_accessor.h>
#include <deal.II/grid/tria_iterator.h>
// We would like to use faces and cells which are not straight lines,
// or bi-linear quads, so we import some classes which predefine some
// manifold descriptions:
#include <deal.II/grid/manifold_lib.h>
// Output of grids in various graphics formats:
#include <deal.II/grid/grid_in.h>
#include <deal.II/grid/grid_out.h>

// This is needed for C++ output:
#include <iostream>
#include <fstream>
// And this for the declarations of the `sqrt' and `fabs' functions:
#include <cmath>

// The final step in importing deal.II is this: All deal.II functions and
// classes are in a namespace <code>dealii</code>, to make sure they don't
// clash with symbols from other libraries you may want to use in conjunction
// with deal.II. One could use these functions and classes by prefixing every
// use of these names by <code>dealii::</code>, but that would quickly become
// cumbersome and annoying. Rather, we simply import the entire deal.II
// namespace for general use:
using namespace dealii;

// @sect3{Creating the first mesh}

// In the following, first function, we simply use the unit square as domain
// and produce a globally refined grid from it.
void first_grid ()
{
  const unsigned int dim = 2;
  const unsigned int spacedim = 3;
   
  Triangulation<dim,spacedim> triangulation;

  //std::string in_mesh_filename = "ellipsoid_mesh.ucd";
  std::string in_mesh_filename = "unit_cube_surface.ucd";
  std::string out_mesh_filename = "mesh.vtk";
    
  std::ifstream in;
  in.open(in_mesh_filename.c_str());
  
  GridIn<dim,spacedim> grid_in;
  grid_in.attach_triangulation (triangulation);
  grid_in.read (in);


  GridOut grid_out;
  std::ofstream out_vtk (out_mesh_filename);
  grid_out.write_vtk (triangulation, out_vtk);
  std::cout << "Grid written to mesh.vtk" << std::endl;

  //triangulation.refine_global (0);

}





// @sect3{The main function}

// Finally, the main function. There isn't much to do here, only to call the
// two subfunctions, which produce the two grids.
int main ()
{
  first_grid ();
}
