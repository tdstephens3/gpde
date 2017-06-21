/* ---------------------------------------------------------------------
 *
 * surface_of_rotation.cc
 modified from step-49 of the tutorial    Feb 12, 2017

 Purpose: to generate a grid using dealii tools, modify it according to a
 smooth function, and then attach a manifold to it.

 * Copyright (C) 2013 - 2015 by the deal.II authors
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

 *
 * Author: Timo Heister, Texas A&M University, 2013
 */

// This tutorial program is odd in the sense that, unlike for most other
// steps, the introduction already provides most of the information on how to
// use the various strategies to generate meshes. Consequently, there is
// little that remains to be commented on here, and we intersperse the code
// with relatively little text. In essence, the code here simply provides a
// reference implementation of what has already been described in the
// introduction.

// @sect3{Include files}

#include <deal.II/grid/tria.h>
#include <deal.II/grid/tria_accessor.h>
#include <deal.II/grid/tria_iterator.h>
#include <deal.II/grid/grid_generator.h>
#include <deal.II/grid/grid_tools.h>
#include <deal.II/grid/manifold_lib.h>
#include <deal.II/grid/grid_out.h>
#include <deal.II/grid/grid_in.h>

#include <iostream>
#include <fstream>

#include <map>

using namespace dealii;


// @sect3{Generating output for a given mesh}

// The following function generates some output for any of the meshes we will
// be generating in the remainder of this program. In particular, it generates
// the following information:
//
// - Some general information about the number of space dimensions in which
//   this mesh lives and its number of cells.
// - The number of boundary faces that use each boundary indicator, so that
//   it can be compared with what we expect.
//
// Finally, the function outputs the mesh in encapsulated postscript (EPS)
// format that can easily be visualized in the same way as was done in step-1.
template <int dim,int spacedim>
void print_mesh_info(const Triangulation<dim,spacedim> &tria,
                     const std::string        &filename)
{
  std::cout << "Mesh info:" << std::endl
            << " dimension: " << dim << std::endl
            << " no. of cells: " << tria.n_active_cells() << std::endl;

  // Next loop over all faces of all cells and find how often each
  // boundary indicator is used (recall that if you access an element
  // of a std::map object that doesn't exist, it is implicitly created
  // and default initialized -- to zero, in the current case -- before
  // we then increment it):
  {
    std::map<unsigned int, unsigned int> boundary_count;
    typename Triangulation<dim,spacedim>::active_cell_iterator
    cell = tria.begin_active(),
    endc = tria.end();
    for (; cell!=endc; ++cell)
      {
        for (unsigned int face=0; face<GeometryInfo<dim>::faces_per_cell; ++face)
          {
            if (cell->face(face)->at_boundary())
              boundary_count[cell->face(face)->boundary_id()]++;
          }
      }

    std::cout << " boundary indicators: ";
    for (std::map<unsigned int, unsigned int>::iterator it=boundary_count.begin();
         it!=boundary_count.end();
         ++it)
      {
        std::cout << it->first << "(" << it->second << " times) ";
      }
    std::cout << std::endl;
  }

  // Finally, produce a graphical representation of the mesh to an output
  // file:
  std::ofstream out (filename.c_str());
  GridOut grid_out;
  grid_out.write_vtk (tria, out);
  std::cout << " written to " << filename
            << std::endl
            << std::endl;
}


  
Point<2> grid_transform(const Point<1> &X)
{
  // transform points X from line to semi circle
  double x,y;
  double R_C, R_B, R_N, r_N;
  R_C = 10, R_B = 2, R_N = 1, r_N = 1;   
  
  x = R_C*cos(X(0));
  y = R_C*sin(X(0));
  
  return Point<2>(x,y);  
}


void assemble_mesh_and_manifold()
{
  
  const int dim = 1;
  const int spacedim = 2;
  
  //  double R_C, R_B, R_N, r_N;
  //  R_C = 10, R_B = 2, R_N = 1, r_N = 1;   

  Triangulation<dim,spacedim> tria;
  
  // generate coarse mesh
  GridGenerator::hyper_rectangle (tria, Point<dim>(0.0), Point<dim>(3.14159));
  //for (Triangulation<dim,spacedim>::active_cell_iterator cell=tria.begin_active(); cell!=tria.end(); ++cell)
  //  cell->set_all_manifold_ids(0);
  
  print_mesh_info(tria, "linear_mesh.vtk");

  GridTools::transform(&grid_transform, tria);
  //
  //GridTools::transform(std_cxx11::bind(&Ellipsoid<dim,spacedim>::grid_transform,std_cxx11::cref(ellipsoid),std_cxx11::_1), tria); // error when trying to bind to member function in same way as step-53
  
  //tria.set_manifold(0,ellipsoid);
  
  tria.refine_global(3);
  
  print_mesh_info(tria, "parametric_mesh.vtk");
 
}


// @sect3{The main function}

// Finally, the main function. There isn't much to do here, only to call the
// subfunctions.
  
int main ()
{
  assemble_mesh_and_manifold();
}
