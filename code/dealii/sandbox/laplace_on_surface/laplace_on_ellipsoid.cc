/* ---------------------------------------------------------------------
 *
 * laplace_on_ellipsoid.cc_
 modified from step-38 and step-26 of the tutorial    August 9, 2016

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
*/


#include <deal.II/base/quadrature_lib.h>
#include <deal.II/base/function.h>
#include <deal.II/grid/tria.h>
#include <deal.II/grid/tria_iterator.h>
#include <deal.II/grid/tria_accessor.h>

#include <deal.II/grid/manifold_lib.h>
#include <deal.II/grid/grid_generator.h>
#include <deal.II/grid/grid_tools.h>
#include <deal.II/grid/grid_out.h>
#include <deal.II/grid/grid_in.h>



#include <deal.II/lac/full_matrix.h>
#include <deal.II/lac/vector.h>
#include <deal.II/lac/solver_control.h>
#include <deal.II/lac/solver_cg.h>
#include <deal.II/lac/precondition.h>
#include <deal.II/lac/sparse_matrix.h>
#include <deal.II/lac/dynamic_sparsity_pattern.h>
#include <deal.II/dofs/dof_handler.h>
#include <deal.II/dofs/dof_accessor.h>
#include <deal.II/dofs/dof_tools.h>
#include <deal.II/fe/fe_values.h>
#include <deal.II/numerics/data_out.h>
#include <deal.II/numerics/vector_tools.h>
#include <deal.II/numerics/matrix_tools.h>

#include <fstream>
#include <iostream>






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
  /*{{{*/
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
  /*}}}*/
}

template <int dim,int spacedim>
class Ellipsoid: public SphericalManifold<dim,spacedim>
{
  /*{{{*/
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
  /*}}}*/
};


template <int dim, int spacedim>
Ellipsoid<dim,spacedim>::Ellipsoid(double a, double b, double c) : SphericalManifold<dim,spacedim>(center), a(a), b(b),c(c), center(0,0,0)       
{
  /*{{{*/
  max_axis = std::max(std::max(a,b),c);
  /*}}}*/
}


template <int dim,int spacedim>
Point<spacedim> Ellipsoid<dim,spacedim>::pull_back(const Point<spacedim> &space_point) const
{
  /*{{{*/
  Point<dim> chart_point = ellipsoid_pull_back(space_point);
  Point<spacedim> p;
  p[0] = -1; // dummy radius to match return of SphericalManifold::pull_back()
  p[1] = chart_point[0];
  p[2] = chart_point[1];
  
  return p;
  /*}}}*/
}


template <int dim,int spacedim>
Point<spacedim> Ellipsoid<dim,spacedim>::push_forward(const Point<spacedim> &chart_point) const
{
  /*{{{*/
  Point<dim> p;  // 
  p[0] = chart_point[1];
  p[1] = chart_point[2];

  Point<spacedim> space_point = ellipsoid_push_forward(p);
  return space_point;
  /*}}}*/
}

 
template <int dim,int spacedim>
Point<spacedim> Ellipsoid<dim,spacedim>::get_new_point(const Quadrature<spacedim> &quad) const 
{
  /*{{{*/
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
  /*}}}*/
}


template <int dim,int spacedim>
Point<dim> Ellipsoid<dim,spacedim>::ellipsoid_pull_back(const Point<spacedim> &space_point) const
{
  /*{{{*/
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
  /*}}}*/
}

template <int dim,int spacedim>
Point<spacedim> Ellipsoid<dim,spacedim>::ellipsoid_push_forward(const Point<dim> &chart_point) const
{
  /*{{{*/
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
  /*}}}*/
}

template<int dim, int spacedim>
Point<spacedim> Ellipsoid<dim,spacedim>::grid_transform(const Point<spacedim> &X)
{
  /*{{{*/
  // transform points X from sphere onto ellipsoid
  double x,y,z;
  
  x = a*X(0);
  y = b*X(1);
  z = c*X(2);
  
  return Point<spacedim>(x,y,z);  
  /*}}}*/
}

Point<3> grid_transform(const Point<3> &X)
{
  /*{{{*/
  // transform points X from sphere onto ellipsoid
  double x,y,z;
  double a = 1; double b = 3; double c = 5;
  x = a*X(0);
  y = b*X(1);
  z = c*X(2);
  
  return Point<3>(x,y,z);  
  /*}}}*/
}

// @sect3{The <code>LaplaceBeltramiProblem</code> class template}

// This class is almost exactly similar to the <code>LaplaceProblem</code>
// class in step-4.

// The essential differences are these:
//
// - The template parameter now denotes the dimensionality of the embedding
//   space, which is no longer the same as the dimensionality of the domain
//   and the triangulation on which we compute. We indicate this by calling
//   the parameter @p spacedim , and introducing a constant @p dim equal to
//   the dimensionality of the domain -- here equal to
//   <code>spacedim-1</code>.
// - All member variables that have geometric aspects now need to know about
//   both their own dimensionality as well as that of the embedding
//   space. Consequently, we need to specify both of their template
//   parameters one for the dimension of the mesh @p dim, and the other for
//   the dimension of the embedding space, @p spacedim. This is exactly what
//   we did in step-34, take a look there for a deeper explanation.
// - We need an object that describes which kind of mapping to use from the
//   reference cell to the cells that the triangulation is composed of. The
//   classes derived from the Mapping base class do exactly this. Throughout
//   most of deal.II, if you don't do anything at all, the library assumes
//   that you want an object of kind MappingQ1 that uses a (bi-, tri-)linear
//   mapping. In many cases, this is quite sufficient, which is why the use
//   of these objects is mostly optional: for example, if you have a
//   polygonal two-dimensional domain in two-dimensional space, a bilinear
//   mapping of the reference cell to the cells of the triangulation yields
//   an exact representation of the domain. If you have a curved domain, one
//   may want to use a higher order mapping for those cells that lie at the
//   boundary of the domain -- this is what we did in step-11, for
//   example. However, here we have a curved domain, not just a curved
//   boundary, and while we can approximate it with bilinearly mapped cells,
//   it is really only prudent to use a higher order mapping for all
//   cells. Consequently, this class has a member variable of type MappingQ;
//   we will choose the polynomial degree of the mapping equal to the
//   polynomial degree of the finite element used in the computations to
//   ensure optimal approximation, though this iso-parametricity is not
//   required.
template <int spacedim>
class LaplaceBeltramiProblem
{
  /*{{{*/
public:
  LaplaceBeltramiProblem (const unsigned degree = 2);
  void run ();

private:
  static const unsigned int dim = spacedim-1;

  void make_grid_and_dofs ();
  void assemble_mesh_and_manifold();
  void assemble_system ();
  void solve ();
  void output_results () const;
  void compute_error () const;


  Triangulation<dim,spacedim>   triangulation;
  FE_Q<dim,spacedim>            fe;
  DoFHandler<dim,spacedim>      dof_handler;
  MappingQ<dim, spacedim>       mapping;

  SparsityPattern               sparsity_pattern;
  SparseMatrix<double>          system_matrix;

  Vector<double>                solution;
  Vector<double>                system_rhs;
  /*}}}*/
};

// @sect3{Equation data}

// Next, let us define the classes that describe the exact solution and the
// right hand sides of the problem. This is in analogy to step-4 and step-7
// where we also defined such objects. Given the discussion in the
// introduction, the actual formulas should be self-explanatory. A point of
// interest may be how we define the value and gradient functions for the 2d
// and 3d cases separately, using explicit specializations of the general
// template. An alternative to doing it this way might have been to define
// the general template and have a <code>switch</code> statement (or a
// sequence of <code>if</code>s) for each possible value of the spatial
// dimension.
template <int dim>
class Solution  : public Function<dim>
{
/*{{{*/
public:
  Solution () : Function<dim>() {}

  virtual double value (const Point<dim>   &p,
                        const unsigned int  component = 0) const;

  virtual Tensor<1,dim> gradient (const Point<dim>   &p,
                                  const unsigned int  component = 0) const;

/*}}}*/
};


template <>
double Solution<2>::value (const Point<2> &p, const unsigned int) const
{
  /*{{{*/
  return ( -2. * p(0) * p(1) );
  /*}}}*/
}


template <>
Tensor<1,2> Solution<2>::gradient (const Point<2>   &p, const unsigned int) const
/*{{{*/
{
  Tensor<1,2> return_value;
  return_value[0] = -2. * p(1) * (1 - 2. * p(0) * p(0));
  return_value[1] = -2. * p(0) * (1 - 2. * p(1) * p(1));

  return return_value;
  /*}}}*/
}


template <>
double Solution<3>::value (const Point<3> &p, const unsigned int) const
{
  /*{{{*/
  return (std::sin(numbers::PI * p(0)) *
          std::cos(numbers::PI * p(1))*exp(p(2)));
  /*}}}*/
}


template <>
Tensor<1,3> Solution<3>::gradient (const Point<3>   &p, const unsigned int) const
{
  /*{{{*/
  using numbers::PI;

  Tensor<1,3> return_value;

  return_value[0] = PI *cos(PI * p(0))*cos(PI * p(1))*exp(p(2));
  return_value[1] = -PI *sin(PI * p(0))*sin(PI * p(1))*exp(p(2));
  return_value[2] = sin(PI * p(0))*cos(PI * p(1))*exp(p(2));

  return return_value;
  /*}}}*/
}



template <int dim>
class RightHandSide : public Function<dim>
{
/*{{{*/
public:
  RightHandSide () : Function<dim>() {}

  virtual double value (const Point<dim>   &p,
                        const unsigned int  component = 0) const;
/*}}}*/
};

template <>
double RightHandSide<2>::value (const Point<2> &p, const unsigned int /*component*/) const
{
  /*{{{*/
  return ( -8. * p(0) * p(1) );
  /*}}}*/
}


template <>
double RightHandSide<3>::value (const Point<3> &p, const unsigned int /*component*/) const
{
/*{{{*/

  using numbers::PI;

  Tensor<2,3> hessian;

  hessian[0][0] = -PI*PI*sin(PI*p(0))*cos(PI*p(1))*exp(p(2));
  hessian[1][1] = -PI*PI*sin(PI*p(0))*cos(PI*p(1))*exp(p(2));
  hessian[2][2] = sin(PI*p(0))*cos(PI*p(1))*exp(p(2));

  hessian[0][1] = -PI*PI*cos(PI*p(0))*sin(PI*p(1))*exp(p(2));
  hessian[1][0] = -PI*PI*cos(PI*p(0))*sin(PI*p(1))*exp(p(2));

  hessian[0][2] = PI*cos(PI*p(0))*cos(PI*p(1))*exp(p(2));
  hessian[2][0] = PI*cos(PI*p(0))*cos(PI*p(1))*exp(p(2));

  hessian[1][2] = -PI*sin(PI*p(0))*sin(PI*p(1))*exp(p(2));
  hessian[2][1] = -PI*sin(PI*p(0))*sin(PI*p(1))*exp(p(2));

  Tensor<1,3> gradient;
  gradient[0] = PI * cos(PI*p(0))*cos(PI*p(1))*exp(p(2));
  gradient[1] = - PI * sin(PI*p(0))*sin(PI*p(1))*exp(p(2));
  gradient[2] = sin(PI*p(0))*cos(PI*p(1))*exp(p(2));

  Point<3> normal = p;
  normal /= p.norm();

  return (- trace(hessian)
          + 2 * (gradient * normal)
          + (hessian * normal) * normal);
/*}}}*/
}


// @sect3{Implementation of the <code>LaplaceBeltramiProblem</code> class}

// The rest of the program is actually quite unspectacular if you know
// step-4. Our first step is to define the constructor, setting the
// polynomial degree of the finite element and mapping, and associating the
// DoF handler to the triangulation:
template <int spacedim>
LaplaceBeltramiProblem<spacedim>::LaplaceBeltramiProblem (const unsigned degree) :
  fe (degree),
  dof_handler(triangulation),
  mapping (degree)
{}


// @sect4{LaplaceBeltramiProblem::make_grid_and_dofs}

// The next step is to create the mesh, distribute degrees of freedom, and
// set up the various variables that describe the linear system. All of
// these steps are standard with the exception of how to create a mesh that
// describes a surface. We could generate a mesh for the domain we are
// interested in, generate a triangulation using a mesh generator, and read
// it in using the GridIn class. Or, as we do here, we generate the mesh
// using the facilities in the GridGenerator namespace.
//
// In particular, what we're going to do is this (enclosed between the set
// of braces below): we generate a <code>spacedim</code> dimensional mesh
// for the half disk (in 2d) or half ball (in 3d), using the
// GridGenerator::half_hyper_ball function. This function sets the boundary
// indicators of all faces on the outside of the boundary to zero for the
// ones located on the perimeter of the disk/ball, and one on the straight
// part that splits the full disk/ball into two halves. The next step is the
// main point: The GridGenerator::extract_boundary_mesh function creates a mesh
// that consists of those cells that are the faces of the previous mesh,
// i.e. it describes the <i>surface</i> cells of the original (volume)
// mesh. However, we do not want all faces: only those on the perimeter of
// the disk or ball which carry boundary indicator zero; we can select these
// cells using a set of boundary indicators that we pass to
// GridGenerator::extract_boundary_mesh.
//
// There is one point that needs to be mentioned. In order to refine a
// surface mesh appropriately if the manifold is curved (similarly to
// refining the faces of cells that are adjacent to a curved boundary), the
// triangulation has to have an object attached to it that describes where
// new vertices should be located. If you don't attach such a boundary
// object, they will be located halfway between existing vertices; this is
// appropriate if you have a domain with straight boundaries (e.g. a
// polygon) but not when, as here, the manifold has curvature. So for things
// to work properly, we need to attach a manifold object to our (surface)
// triangulation, in much the same way as we've already done in 1d for the
// boundary. We create such an object (with indefinite, <code>static</code>,
// lifetime) at the top of the function and attach it to the triangulation
// for all cells with boundary indicator zero that will be created
// henceforth.
//
// The final step in creating the mesh is to refine it a number of
// times. The rest of the function is the same as in previous tutorial
// programs.
template <int spacedim>
void LaplaceBeltramiProblem<spacedim>::make_grid_and_dofs ()
{
  /*{{{*/
  double a = 1.0; double b = 3.0; double c = 5.0;
  static Ellipsoid<dim,spacedim> ellipsoid(a,b,c);

  GridGenerator::hyper_sphere(triangulation,Point<spacedim>(0,0,0), 1.0);
  triangulation.set_all_manifold_ids(0);
  
  GridTools::transform(&grid_transform, triangulation);

  triangulation.set_manifold (0, ellipsoid);
  triangulation.refine_global(5);


  std::cout << "Surface mesh has " << triangulation.n_active_cells()
            << " cells."
            << std::endl;

  dof_handler.distribute_dofs (fe);

  std::cout << "Surface mesh has " << dof_handler.n_dofs()
            << " degrees of freedom."
            << std::endl;

  print_mesh_info(triangulation, "ellipsoidal_mesh.vtk");
  
  
  DynamicSparsityPattern dsp (dof_handler.n_dofs(), dof_handler.n_dofs());
  DoFTools::make_sparsity_pattern (dof_handler, dsp);
  sparsity_pattern.copy_from (dsp);

  system_matrix.reinit (sparsity_pattern);

  solution.reinit (dof_handler.n_dofs());
  system_rhs.reinit (dof_handler.n_dofs());
  /*}}}*/
}


// @sect4{LaplaceBeltramiProblem::assemble_system}

// The following is the central function of this program, assembling the
// matrix that corresponds to the surface Laplacian (Laplace-Beltrami
// operator). Maybe surprisingly, it actually looks exactly the same as for
// the regular Laplace operator discussed in, for example, step-4. The key
// is that the FEValues::shape_gradient function does the magic: It returns
// the surface gradient $\nabla_K \phi_i(x_q)$ of the $i$th shape function
// at the $q$th quadrature point. The rest then does not need any changes
// either:
template <int spacedim>
void LaplaceBeltramiProblem<spacedim>::assemble_system ()
{
  /*{{{*/
  system_matrix = 0;
  system_rhs = 0;

  const QGauss<dim>  quadrature_formula(2*fe.degree);
  FEValues<dim,spacedim> fe_values (mapping, fe, quadrature_formula,
                                    update_values              |
                                    update_gradients           |
                                    update_quadrature_points   |
                                    update_JxW_values);

  const unsigned int        dofs_per_cell = fe.dofs_per_cell;
  const unsigned int        n_q_points    = quadrature_formula.size();

  FullMatrix<double>        cell_matrix (dofs_per_cell, dofs_per_cell);
  Vector<double>            cell_rhs (dofs_per_cell);

  std::vector<double>       rhs_values(n_q_points);
  std::vector<types::global_dof_index> local_dof_indices (dofs_per_cell);

  const RightHandSide<spacedim> rhs;

  for (typename DoFHandler<dim,spacedim>::active_cell_iterator
       cell = dof_handler.begin_active(),
       endc = dof_handler.end();
       cell!=endc; ++cell)
    {
      cell_matrix = 0;
      cell_rhs = 0;

      fe_values.reinit (cell);

      rhs.value_list (fe_values.get_quadrature_points(), rhs_values);

      for (unsigned int i=0; i<dofs_per_cell; ++i)
        for (unsigned int j=0; j<dofs_per_cell; ++j)
          for (unsigned int q_point=0; q_point<n_q_points; ++q_point)
            cell_matrix(i,j) += fe_values.shape_grad(i,q_point) *
                                fe_values.shape_grad(j,q_point) *
                                fe_values.JxW(q_point);

      for (unsigned int i=0; i<dofs_per_cell; ++i)
        for (unsigned int q_point=0; q_point<n_q_points; ++q_point)
          cell_rhs(i) += fe_values.shape_value(i,q_point) *
                         rhs_values[q_point]*
                         fe_values.JxW(q_point);

      cell->get_dof_indices (local_dof_indices);
      for (unsigned int i=0; i<dofs_per_cell; ++i)
        {
          for (unsigned int j=0; j<dofs_per_cell; ++j)
            system_matrix.add (local_dof_indices[i],
                               local_dof_indices[j],
                               cell_matrix(i,j));

          system_rhs(local_dof_indices[i]) += cell_rhs(i);
        }
    }

  std::map<types::global_dof_index,double> boundary_values;
  VectorTools::interpolate_boundary_values (mapping,
                                            dof_handler,
                                            0,
                                            Solution<spacedim>(),
                                            boundary_values);

  MatrixTools::apply_boundary_values (boundary_values,
                                      system_matrix,
                                      solution,
                                      system_rhs,false);
/*}}}*/
}

// @sect4{LaplaceBeltramiProblem::solve}

// The next function is the one that solves the linear system. Here, too, no
// changes are necessary:
template <int spacedim>
void LaplaceBeltramiProblem<spacedim>::solve ()
{
  /*{{{*/
  SolverControl solver_control (solution.size(),
                                1e-7 * system_rhs.l2_norm());
  SolverCG<>    cg (solver_control);

  PreconditionSSOR<> preconditioner;
  preconditioner.initialize(system_matrix, 1.2);

  cg.solve (system_matrix, solution, system_rhs,
            preconditioner);
  /*}}}*/
}

// @sect4{LaplaceBeltramiProblem::output_result}

// This is the function that generates graphical output from the
// solution. Most of it is boilerplate code, but there are two points worth
// pointing out:
//
// - The DataOut::add_data_vector function can take two kinds of vectors:
//   Either vectors that have one value per degree of freedom defined by the
//   DoFHandler object previously attached via DataOut::attach_dof_handler;
//   and vectors that have one value for each cell of the triangulation, for
//   example to output estimated errors for each cell. Typically, the
//   DataOut class knows to tell these two kinds of vectors apart: there are
//   almost always more degrees of freedom than cells, so we can
//   differentiate by the two kinds looking at the length of a vector. We
//   could do the same here, but only because we got lucky: we use a half
//   sphere. If we had used the whole sphere as domain and $Q_1$ elements,
//   we would have the same number of cells as vertices and consequently the
//   two kinds of vectors would have the same number of elements. To avoid
//   the resulting confusion, we have to tell the DataOut::add_data_vector
//   function which kind of vector we have: DoF data. This is what the third
//   argument to the function does.
// - The DataOut::build_patches function can generate output that subdivides
//   each cell so that visualization programs can resolve curved manifolds
//   or higher polynomial degree shape functions better. We here subdivide
//   each element in each coordinate direction as many times as the
//   polynomial degree of the finite element in use.
template <int spacedim>
void LaplaceBeltramiProblem<spacedim>::output_results () const
{
  /*{{{*/
  DataOut<dim,DoFHandler<dim,spacedim> > data_out;
  data_out.attach_dof_handler (dof_handler);
  data_out.add_data_vector (solution,
                            "solution",
                            DataOut<dim,DoFHandler<dim,spacedim> >::type_dof_data);
  data_out.build_patches (mapping,
                          mapping.get_degree());

  std::string filename ("solution-");
  filename += static_cast<char>('0'+spacedim);
  filename += "d.vtk";
  std::ofstream output (filename.c_str());
  data_out.write_vtk (output);
  /*}}}*/
}



// @sect4{LaplaceBeltramiProblem::compute_error}

// This is the last piece of functionality: we want to compute the error in
// the numerical solution. It is a verbatim copy of the code previously
// shown and discussed in step-7. As mentioned in the introduction, the
// <code>Solution</code> class provides the (tangential) gradient of the
// solution. To avoid evaluating the error only a superconvergence points,
// we choose a quadrature rule of sufficiently high order.
template <int spacedim>
void LaplaceBeltramiProblem<spacedim>::compute_error () const
{
  /*{{{*/
  Vector<float> difference_per_cell (triangulation.n_active_cells());
  VectorTools::integrate_difference (mapping, dof_handler, solution,
                                     Solution<spacedim>(),
                                     difference_per_cell,
                                     QGauss<dim>(2*fe.degree+1),
                                     VectorTools::H1_norm);

  std::cout << "H1 error = "
            << difference_per_cell.l2_norm()
            << std::endl;
  /*}}}*/
}







template<int spacedim>                                                                              
void LaplaceBeltramiProblem<spacedim>::run() 
{                                                                
/*{{{*/                                                                                        
//          // @sect4{LaplaceBeltramiProblem::run}
//          
//          // The last function provides the top-level logic. Its contents are
//          // self-explanatory:
//          template <int spacedim>
//          void LaplaceBeltramiProblem<spacedim>::run ()
  /*{{{*/
  make_grid_and_dofs();
  assemble_system ();
  solve ();
  output_results ();
  compute_error ();
  /*}}}*/
                                                                                                 
//           const unsigned int initial_global_refinement = 3;                                            
//                                                                                                        
//           assemble_mesh_and_manifold()
//           
//           triangulation.refine_global (initial_global_refinement);                                     
//                                                                                                        
//           setup_system();                                                                              
//                                                                                                        
//           unsigned int pre_refinement_step = 0;                                                        
//                                                                                                        
//           Vector<double> tmp;                                                                          
//           Vector<double> forcing_terms;                                                                
//                                                                                                        
//         start_time_iteration:                                                                            
//                                                                                                        
//           tmp.reinit (solution.size());                                                                
//           forcing_terms.reinit (solution.size());                                                      
//                                                                                                        
//                                                                                                        
//           VectorTools::interpolate(dof_handler,                                                        
//                                    ZeroFunction<dim>(),                                                
//                                    old_solution);                                                      
//           solution = old_solution;                                                                     
//                                                                                                        
//           timestep_number = 0;                                                                         
//           time            = 0;                                                                         
//                                                                                                        
//           output_results();                                                                            
//                                                                                                        
//           // Then we start the main loop until the computed time exceeds our                           
//           // end time of 0.5. The first task is to build the right hand                                
//           // side of the linear system we need to solve in each time step.                             
//           // Recall that it contains the term $MU^{n-1}-(1-\theta)k_n AU^{n-1}$.                       
//           // We put these terms into the variable system_rhs, with the                                 
//           // help of a temporary vector:                                                               
//           while (time <= 0.5)                                                                          
//           {                                                                                          
//             time += time_step;                                                                       
//             ++timestep_number;                                                                       
//                                                                                                      
//             std::cout << "Time step " << timestep_number << " at t=" << time                         
//                       << std::endl;                                                                  
//                                                                                                      
//             mass_matrix.vmult(system_rhs, old_solution);                                             
//                                                                                                      
//             laplace_matrix.vmult(tmp, old_solution);                                                 
//             system_rhs.add(-(1 - theta) * time_step, tmp);                                           
//                                                                                                      
//             // The second piece is to compute the contributions of the source                        
//             // terms. This corresponds to the term $k_n                                              
//             // \left[ (1-\theta)F^{n-1} + \theta F^n \right]$. The following                         
//             // code calls VectorTools::create_right_hand_side to compute the                         
//             // vectors $F$, where we set the time of the right hand side                             
//             // (source) function before we evaluate it. The result of this                           
//             // all ends up in the forcing_terms variable:                                            
//             RightHandSide<spacedim> rhs_function;                                                         
//             rhs_function.set_time(time);                                                             
//             VectorTools::create_right_hand_side(dof_handler,                                         
//                                                 QGauss<dim>(fe.degree+1),                            
//                                                 rhs_function,                                        
//                                                 tmp);                                                
//             forcing_terms = tmp;                                                                     
//             forcing_terms *= time_step * theta;                                                      
//                                                                                                      
//             rhs_function.set_time(time - time_step);                                                 
//             VectorTools::create_right_hand_side(dof_handler,                                         
//                                                 QGauss<dim>(fe.degree+1),                            
//                                                 rhs_function,                                        
//                                                 tmp);                                                
//                                                                                                      
//             forcing_terms.add(time_step * (1 - theta), tmp);                                         
//                                                                                                      
//             // Next, we add the forcing terms to the ones that                                       
//             // come from the time stepping, and also build the matrix                                
//             // $M+k_n\theta A$ that we have to invert in each time step.                             
//             // The final piece of these operations is to eliminate                                   
//             // hanging node constrained degrees of freedom from the                                  
//             // linear system:                                                                        
//             system_rhs += forcing_terms;                                                             
//                                                                                                      
//             system_matrix.copy_from(mass_matrix);                                                    
//             system_matrix.add(theta * time_step, laplace_matrix);                                    
//                                                                                                      
//             constraints.condense (system_matrix, system_rhs);                                        
//                                                                                                      
//             // There is one more operation we need to do before we                                   
//             // can solve it: boundary values. To this end, we create                                 
//             // a boundary value object, set the proper time to the one                               
//             // of the current time step, and evaluate it as we have                                  
//             // done many times before. The result is used to also                                    
//             // set the correct boundary values in the linear system:                                 
//             {                                                                                        
//               BoundaryValues<dim> boundary_values_function;                                          
//               boundary_values_function.set_time(time);                                               
//                                                                                                      
//               std::map<types::global_dof_index, double> boundary_values;                             
//               VectorTools::interpolate_boundary_values(dof_handler,                                  
//                                                        0,                                            
//                                                        boundary_values_function,                     
//                                                        boundary_values);                             
//                                                                                                      
//               MatrixTools::apply_boundary_values(boundary_values,                                    
//                                                  system_matrix,                                      
//                                                  solution,                                           
//                                                  system_rhs);                                        
//             }                                                                                        
//                                                                                                      
//             // With this out of the way, all we have to do is solve the                              
//             // system, generate graphical data, and...                                               
//             solve_time_step();                                                                       
//                                                                                                      
//             output_results();                                                                        
//                                                                                                      
//             // ...take care of mesh refinement. Here, what we want to do is                          
//             // (i) refine the requested number of times at the very beginning                        
//             // of the solution procedure, after which we jump to the top to                          
//             // restart the time iteration, (ii) refine every fifth time                              
//             // step after that.                                                                      
//             //                                                                                       
//             // The time loop and, indeed, the main part of the program ends                          
//             // with starting into the next time step by setting old_solution                         
//             // to the solution we have just computed.                                                
//             if ((timestep_number == 1) &&                                                            
//                 (pre_refinement_step < n_adaptive_pre_refinement_steps))                             
//               {                                                                                      
//                 refine_mesh (initial_global_refinement,                                              
//                              initial_global_refinement + n_adaptive_pre_refinement_steps);           
//                 ++pre_refinement_step;                                                               
//                                                                                                      
//                 tmp.reinit (solution.size());                                                        
//                 forcing_terms.reinit (solution.size());                                              
//                                                                                                      
//                 std::cout << std::endl;                                                              
//                                                                                                      
//                 goto start_time_iteration;                                                           
//               }                                                                                      
//             else if ((timestep_number > 0) && (timestep_number % 5 == 0))                            
//               {                                                                                      
//                 refine_mesh (initial_global_refinement,                                              
//                              initial_global_refinement + n_adaptive_pre_refinement_steps);           
//                 tmp.reinit (solution.size());                                                        
//                 forcing_terms.reinit (solution.size());                                              
//               }                                                                                      
//                                                                                                      
//             old_solution = solution;                                                                 
//           }                                                                                          
/*}}}*/                                                                                    
}                                                                                              

// @sect3{The <code>main</code> function}
//
// Having made it this far,  there is, again, nothing
// much to discuss for the main function of this
// program: it looks like all such functions since step-6.
// @sect3{The main() function}

// The remainder of the program is taken up by the <code>main()</code>
// function. It follows exactly the general layout first introduced in step-6
// and used in all following tutorial programs:
int main ()
{
  try
    {
      using namespace dealii;

      LaplaceBeltramiProblem<3> laplace_beltrami;
      laplace_beltrami.run();
    }
  catch (std::exception &exc)
    {
      std::cerr << std::endl << std::endl
                << "----------------------------------------------------"
                << std::endl;
      std::cerr << "Exception on processing: " << std::endl
                << exc.what() << std::endl
                << "Aborting!" << std::endl
                << "----------------------------------------------------"
                << std::endl;
      return 1;
    }
  catch (...)
    {
      std::cerr << std::endl << std::endl
                << "----------------------------------------------------"
                << std::endl;
      std::cerr << "Unknown exception!" << std::endl
                << "Aborting!" << std::endl
                << "----------------------------------------------------"
                << std::endl;
      return 1;
    }

  return 0;
}
