/* ---------------------------------------------------------------------
 *
 * sphere_to_ellipsoid.cc_
 modified from step-49 of the tutorial    July 25, 2016

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


template <int dim,int spacedim>
class Ellipsoid: public SphericalManifold<dim,spacedim>
{
public:

  Ellipsoid(double,double,double);   

  Point<spacedim> pull_back(const Point<spacedim> &space_point) const;
  
  Point<spacedim> push_forward(const Point<spacedim> &chart_point) const;
  
  Point<spacedim> get_new_point(const Quadrature<spacedim> &quad) const;

  Point<spacedim> grid_transform(const Point<spacedim> &X);

private:
  double  a,b,c;
  double max_axis;
  const Point<spacedim> center; 
  
  Point<dim> ellipsoid_pull_back(const Point<spacedim> &space_point) const;
  
  Point<spacedim> ellipsoid_push_forward(const Point<dim> &chart_point) const;
  
};


template <int dim, int spacedim>
Ellipsoid<dim,spacedim>::Ellipsoid(double a, double b, double c) : SphericalManifold<dim,spacedim>(center), a(a), b(b),c(c), center(0,0,0)       
{
  max_axis = std::max(std::max(a,b),c);
}


template <int dim,int spacedim>
Point<spacedim> Ellipsoid<dim,spacedim>::pull_back(const Point<spacedim> &space_point) const
{
  Point<dim> chart_point = ellipsoid_pull_back(space_point);
  Point<spacedim> p;
  p[0] = -1; // dummy radius to match return of SphericalManifold::pull_back()
  p[1] = chart_point[0];
  p[2] = chart_point[1];
  
  return p;
}


template <int dim,int spacedim>
Point<spacedim> Ellipsoid<dim,spacedim>::push_forward(const Point<spacedim> &chart_point) const
{
  
  Point<dim> p;  // 
  p[0] = chart_point[1];
  p[1] = chart_point[2];

  Point<spacedim> space_point = ellipsoid_push_forward(p);
  return space_point;

}

 
template <int dim,int spacedim>
Point<spacedim> Ellipsoid<dim,spacedim>::get_new_point(const Quadrature<spacedim> &quad) const 
{
  double u,v,w;
  std::vector< Point<spacedim> > space_points;
  for (unsigned int i=0; i<quad.size(); ++i) 
  {
    u = quad.point(i)[0]/a;
    v = quad.point(i)[1]/b;
    w = quad.point(i)[2]/c;
    space_points.push_back(Point<spacedim>(u,v,w));
  }
  
  Quadrature<spacedim> spherical_quad = Quadrature<spacedim>(space_points, quad.get_weights());

  Point<spacedim> p = SphericalManifold<dim,spacedim>::get_new_point(spherical_quad); 
  double x,y,z;
  x = a*p[0];
  y = b*p[1];
  z = c*p[2];

  Point<spacedim> new_point = Point<spacedim>(x,y,z);
  return new_point;
}


template <int dim,int spacedim>
Point<dim> Ellipsoid<dim,spacedim>::ellipsoid_pull_back(const Point<spacedim> &space_point) const
{
  double x,y,z, u,v,w;
  
  // get point on ellipsoid
  x = space_point[0];
  y = space_point[1];
  z = space_point[2];

  std::cout << "using a,b,c: " << std::endl;
  std::cout << a << " " << b << " "  << c << std::endl;
  std::cout << "from pull_back: " << std::endl;
  std::cout << "space_point: " << std::endl;
  std::cout << x << " " << y << " "  << z << std::endl;

  // map ellipsoid point onto sphere
  u = x/a;
  v = y/b;
  w = z/c;

  std::cout << "pulls back to : " << std::endl;
  std::cout << u << " " << v << " "  << w << std::endl;
  std::cout << "on sphere." << std::endl;
  
  Point<spacedim> p(u,v,w);

  // use reference_sphere's pull_back function
  Point<spacedim> q = pull_back(p);
  Point<dim> chart_point;

  
  std::cout << "sphere pull_back: " << std::endl;
  std::cout << q[0] << " " << q[1] << " "  << q[2] << std::endl;
  std::cout << "r theta phi" << std::endl;
  std::cout << "..........." << std::endl;
 
  chart_point[0] = q[1];
  chart_point[1] = q[2];

  // return (theta,phi) in the chart domain 
  return chart_point;

}

template <int dim,int spacedim>
Point<spacedim> Ellipsoid<dim,spacedim>::ellipsoid_push_forward(const Point<dim> &chart_point) const
{
  double theta,phi, x,y,z;
  
  phi   = chart_point[0];
  theta = chart_point[1];
  

  Point<spacedim> p(max_axis,theta,phi);
  // map theta,phi in chart domain onto reference_sphere with radius max_axis
  Point<spacedim> X = push_forward(p);
 
  // map point on sphere onto ellipsoid
  
  x = a*X[0];
  y = b*X[1];
  z = c*X[2];
  
  Point<spacedim> space_point(x,y,z);
  
  // return point on ellipsoid
  return space_point;
}

template<int dim, int spacedim>
Point<spacedim> Ellipsoid<dim,spacedim>::grid_transform(const Point<spacedim> &X)
{
  // transform points X from sphere onto ellipsoid
  double x,y,z;
  
  x = a*X(0);
  y = b*X(1);
  z = c*X(2);
  
  return Point<spacedim>(x,y,z);  
}

Point<3> grid_transform(const Point<3> &X)
{
  // transform points X from sphere onto ellipsoid
  double x,y,z;
  double a = 1; double b = 3; double c = 5;
  x = a*X(0);
  y = b*X(1);
  z = c*X(2);
  
  return Point<3>(x,y,z);  
}


void assemble_mesh_and_manifold()
{
  
  const int dim = 2;
  const int spacedim = 3;
  
  double a,b,c;
  a = 1; b=3; c=5;

  Ellipsoid<dim,spacedim> ellipsoid(a,b,c);

  Triangulation<dim,spacedim> tria;
  
  // generate coarse spherical mesh
  GridGenerator::hyper_sphere (tria, Point<spacedim>(0.0,0.0,0.0), 1.0);
  for (Triangulation<dim,spacedim>::active_cell_iterator cell=tria.begin_active(); cell!=tria.end(); ++cell)
    cell->set_all_manifold_ids(0);
  
  print_mesh_info(tria, "spherical_mesh.vtk");

  GridTools::transform(&grid_transform, tria);
  //
  //GridTools::transform(std_cxx11::bind(&Ellipsoid<dim,spacedim>::grid_transform,std_cxx11::cref(ellipsoid),std_cxx11::_1), tria); // error when trying to bind to member function in same way as step-53
  
  tria.set_manifold(0,ellipsoid);
  
  tria.refine_global(3);
  
  print_mesh_info(tria, "ellipsoidal_mesh.vtk");
 
}


// @sect3{The main function}

// Finally, the main function. There isn't much to do here, only to call the
// subfunctions.
  
int main ()
{
  assemble_mesh_and_manifold();
}
