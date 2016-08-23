/* ---------------------------------------------------------------------
 *
 * mean_curvature_via_step38.cc
 *
 *  MODIFIED VERSION OF STEP-38, TOM STEPHENS, August 2016
 *
 * Copyright (C) 2010 - 2015 by the deal.II authors
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
 * Authors: Andrea Bonito, Sebastian Pauletti.
 */


#include <deal.II/base/quadrature_lib.h>
#include <deal.II/base/function.h>
#include <deal.II/grid/tria.h>
#include <deal.II/grid/tria_iterator.h>
#include <deal.II/grid/tria_accessor.h>
#include <deal.II/grid/manifold_lib.h>
#include <deal.II/grid/grid_tools.h>
#include <deal.II/grid/grid_generator.h>
#include <deal.II/lac/full_matrix.h>
#include <deal.II/lac/vector.h>
#include <deal.II/lac/solver_control.h>
#include <deal.II/lac/solver_cg.h>
#include <deal.II/lac/solver_gmres.h>
#include <deal.II/lac/precondition.h>
#include <deal.II/lac/sparse_matrix.h>
#include <deal.II/lac/dynamic_sparsity_pattern.h>
#include <deal.II/dofs/dof_handler.h>
#include <deal.II/dofs/dof_accessor.h>
#include <deal.II/dofs/dof_tools.h>
#include <deal.II/fe/fe_system.h>
#include <deal.II/fe/fe_values.h>
#include <deal.II/numerics/data_out.h>
#include <deal.II/numerics/vector_tools.h>
#include <deal.II/numerics/matrix_tools.h>

#include <stdio.h>
#include <fstream>
#include <iostream>

// additional cpp code that is reusable
#include "../../utilities/my_manifolds_lib.cc"

namespace Step38
{
  using namespace dealii;

template <int spacedim>
class LaplaceBeltramiProblem
{
  /*{{{*/
  public:
    LaplaceBeltramiProblem (const unsigned degree = 2);
    void run ();
  
  private:
    static const unsigned int dim = spacedim-1;
  
    void make_grid_and_dofs (double,double,double,Point<spacedim>);
    void assemble_system ();
    void solve ();
    void output_results () const;
    void compute_error (double,double,double,Point<3>) const;
  
  
    Triangulation<dim,spacedim>   triangulation;
    FE_Q<dim,spacedim>            fe; 
    double                        degree = 2;
    DoFHandler<dim,spacedim>      dof_handler;
    MappingQ<dim, spacedim>       mapping;
  
    SparsityPattern               sparsity_pattern;
    SparseMatrix<double>          system_matrix;
  
    Vector<double>                solution_x;
    Vector<double>                solution_y;
    Vector<double>                solution_z;
    Vector<double>                system_rhs_x;
    Vector<double>                system_rhs_y;
    Vector<double>                system_rhs_z;
    
    Vector<double>                mean_curvature_squared;
    Vector<double>                exact_solution_values;
    /*}}}*/
};

template <int spacedim>
class Identity : public Function<spacedim>
{
/*{{{*/
  public:
    Identity() : Function<spacedim>() {}
    
    virtual void vector_value_list (const std::vector<Point<spacedim> > &points,
                                            std::vector<Vector<double> >   &value_list) const;
    virtual void vector_value (const Point<spacedim> &p, Vector<double> &value) const;
    virtual double value (const Point<spacedim> &p, const unsigned int component = 0) const;
    
    virtual Tensor<2,spacedim> shape_grad(const Tensor<1,spacedim> &unit_normal) const;
    virtual Tensor<1,spacedim> shape_grad_component(const Tensor<1,spacedim> &unit_normal, const unsigned int component) const;
/*}}}*/
};

template<int spacedim>
double Identity<spacedim>::value(const Point<spacedim> &p, const unsigned int component)  const
{
  /*{{{*/
  return p(component);
  /*}}}*/
}

template<int spacedim>
void Identity<spacedim>::vector_value(const Point<spacedim> &p, Vector<double> &value) const
{
  /*{{{*/
  for (unsigned int c=0; c<this->n_components; ++c) 
  {
    value(c) = Identity<spacedim>::value(p,c);
  }
  /*}}}*/
}

template <int spacedim>
void Identity<spacedim>::vector_value_list (const std::vector<Point<spacedim> > &points,
                                            std::vector<Vector<double> >   &value_list) const
{
  /*{{{*/
  Assert (value_list.size() == points.size(),
          ExcDimensionMismatch (value_list.size(), points.size()));
  const unsigned int n_points = points.size();
  for (unsigned int p=0; p<n_points; ++p)
    Identity<spacedim>::vector_value (points[p],
                                      value_list[p]);
  /*}}}*/
}

template <int spacedim>
Tensor<2,spacedim> Identity<spacedim>::shape_grad(const Tensor<1,spacedim> &unit_normal) const
{
  /*{{{*/
  Tensor<2,spacedim> eye;
  eye = 0; eye[0][0] = 1; eye[1][1] = 1; eye[2][2] = 1;
  Tensor<2,spacedim> nnT;
  nnT = outer_product(unit_normal,unit_normal);
  return eye - nnT;
  /*}}}*/
}

template <int spacedim>
Tensor<1,spacedim> Identity<spacedim>::shape_grad_component(const Tensor<1,spacedim> &unit_normal, const unsigned int component) const
{
  /*{{{*/
  Tensor<2,spacedim> full_shape_grad = shape_grad(unit_normal);
  Tensor<1,spacedim> grad_component;

  grad_component[0] = full_shape_grad[component][0];
  grad_component[1] = full_shape_grad[component][1];
  grad_component[2] = full_shape_grad[component][2];

  return grad_component;
  /*}}}*/
}


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
template <int spacedim>
class ExactSolution : public Function<spacedim>
{
  /*{{{*/
  public:
    ExactSolution<spacedim> (double a,double b, double c, Point<3> center) 
      : Function<spacedim>(), a(a),b(b),c(c),center(center),ellipsoid(a,b,c,center)  {}
  
    virtual double value (const Point<3>   &p,
                          const unsigned int  component = 0) const;
  private:
    double a,b,c;
    Point<3> center;
    Ellipsoid<2,3> ellipsoid;
    /*}}}*/
};

template <int spacedim>
double ExactSolution<spacedim>::value (const Point<3> &p,
                             const unsigned int) const
{
  /*{{{*/
  Point<3> chart_point = ellipsoid.pull_back(p);
  
  double theta = chart_point(1);
  double phi   = chart_point(2);

  double mean_curv = 2*a*b*c*( 3*(pow(a,2) + pow(b,2)) + 2*pow(c,2) 
                               + (pow(a,2) + pow(b,2) - 2*pow(c,2))*cos(2*theta) 
                              - 2*(pow(a,2) - pow(b,2))*cos(2*phi)*pow(sin(theta),2) ) 
                           / ( 8*pow((pow(a,2)*pow(b,2)*pow(cos(theta),2)
                                + pow(c,2)*(pow(b,2)*pow(cos(phi),2) 
                                + pow(a,2)*pow(sin(phi),2))*pow(sin(theta),2)),1.5) );
  
  double mean_curv_squared = pow(mean_curv,2);
  return mean_curv_squared;
  /*}}}*/
}

  
template <int spacedim>
LaplaceBeltramiProblem<spacedim>::LaplaceBeltramiProblem (const unsigned degree)
  :
  fe(degree),
  dof_handler(triangulation),
  mapping (degree)
{}


template <int spacedim>
void LaplaceBeltramiProblem<spacedim>::make_grid_and_dofs (double a, double b, double c, Point<spacedim> center)
{
  /*{{{*/
  static Ellipsoid<dim,spacedim> ellipsoid(a,b,c,center);

  GridGenerator::hyper_sphere(triangulation,center, 1);
  
  triangulation.set_all_manifold_ids(0);
  
  GridTools::transform(std_cxx11::bind(&Ellipsoid<dim,spacedim>::grid_transform, &ellipsoid, std_cxx11::_1), 
                       triangulation);

  triangulation.set_manifold (0, ellipsoid);
  triangulation.refine_global(3);

  std::cout << "Surface mesh has " << triangulation.n_active_cells()
            << " cells."
            << std::endl;

  dof_handler.distribute_dofs (fe);

  std::cout << "Surface mesh has " << dof_handler.n_dofs()
            << " degrees of freedom."
            << std::endl;

  DynamicSparsityPattern dsp (dof_handler.n_dofs(), dof_handler.n_dofs());
  DoFTools::make_sparsity_pattern (dof_handler, dsp);
  sparsity_pattern.copy_from (dsp);

  system_matrix.reinit (sparsity_pattern);

  solution_x.reinit (dof_handler.n_dofs());
  solution_y.reinit (dof_handler.n_dofs());
  solution_z.reinit (dof_handler.n_dofs());
  system_rhs_x.reinit (dof_handler.n_dofs());
  system_rhs_y.reinit (dof_handler.n_dofs());
  system_rhs_z.reinit (dof_handler.n_dofs());
  /*}}}*/
}

template <int spacedim>
void LaplaceBeltramiProblem<spacedim>::assemble_system ()
{
  /*{{{*/
  system_matrix = 0;
  system_rhs_x  = 0;
  system_rhs_y  = 0;
  system_rhs_z  = 0;

  Identity<spacedim> identity_on_manifold;

  const QGauss<dim>  quadrature_formula(2*fe.degree);
  FEValues<dim,spacedim> fe_values (mapping, fe, quadrature_formula,
                                    update_values              |
                                    update_normal_vectors      |
                                    update_gradients           |
                                    update_quadrature_points   |
                                    update_JxW_values);


  const unsigned int  dofs_per_cell = fe.dofs_per_cell;
  const unsigned int  n_q_points    = quadrature_formula.size();

  FullMatrix<double>  cell_matrix (dofs_per_cell, dofs_per_cell);
  Vector<double>      cell_rhs_x (dofs_per_cell);
  Vector<double>      cell_rhs_y (dofs_per_cell);
  Vector<double>      cell_rhs_z (dofs_per_cell);

  std::vector<types::global_dof_index> local_dof_indices (dofs_per_cell);

  for (typename DoFHandler<dim,spacedim>::active_cell_iterator
       cell = dof_handler.begin_active(),
       endc = dof_handler.end();
       cell!=endc; ++cell)
    {
      cell_matrix = 0;
      cell_rhs_x  = 0;
      cell_rhs_y  = 0;
      cell_rhs_z  = 0;

      fe_values.reinit (cell);

      for (unsigned int i=0; i<dofs_per_cell; ++i)
        for (unsigned int j=0; j<dofs_per_cell; ++j)
          for (unsigned int q_point=0; q_point<n_q_points; ++q_point)
          {
            
            cell_matrix(i,j) += fe_values.shape_value(i,q_point) *
                                fe_values.shape_value(j,q_point) *
                                fe_values.JxW(q_point);
          }

      for (unsigned int i=0; i<dofs_per_cell; ++i)
        for (unsigned int q_point=0; q_point<n_q_points; ++q_point)
        {
        cell_rhs_x(i) += fe_values.shape_grad(i,q_point)*
                         identity_on_manifold.shape_grad_component(fe_values.normal_vector(q_point),0)* 
                         fe_values.JxW(q_point);
        cell_rhs_y(i) += fe_values.shape_grad(i,q_point)*
                         identity_on_manifold.shape_grad_component(fe_values.normal_vector(q_point),1)* 
                         fe_values.JxW(q_point);
        cell_rhs_z(i) += fe_values.shape_grad(i,q_point)*
                         identity_on_manifold.shape_grad_component(fe_values.normal_vector(q_point),2)* 
                         fe_values.JxW(q_point);
        }
      cell->get_dof_indices (local_dof_indices);
      for (unsigned int i=0; i<dofs_per_cell; ++i)
      {
        for (unsigned int j=0; j<dofs_per_cell; ++j)
          system_matrix.add (local_dof_indices[i],
                             local_dof_indices[j],
                             cell_matrix(i,j));

        system_rhs_x(local_dof_indices[i]) += cell_rhs_x(i);
        system_rhs_y(local_dof_indices[i]) += cell_rhs_y(i);
        system_rhs_z(local_dof_indices[i]) += cell_rhs_z(i);
      }
    }
  /*}}}*/
}

template <int spacedim>
void LaplaceBeltramiProblem<spacedim>::solve ()
{
  /*{{{*/
  SolverControl solver_control (solution_x.size(), 1e-12 );
  SolverCG<>    cg (solver_control);


  cg.solve (system_matrix, solution_x, system_rhs_x, PreconditionIdentity());
  std::cout << "Solved x component" << std::endl;
  cg.solve (system_matrix, solution_y, system_rhs_y, PreconditionIdentity());
  std::cout << "Solved y component" << std::endl;
  cg.solve (system_matrix, solution_z, system_rhs_z, PreconditionIdentity());
  std::cout << "Solved z component" << std::endl;

  mean_curvature_squared.reinit (dof_handler.n_dofs());
  
  double avg = 0; double summ = 0;
  for (unsigned int i=0; i<dof_handler.n_dofs(); ++i )
  {
    mean_curvature_squared(i) = pow(solution_x(i),2) + pow(solution_y(i),2) + pow(solution_z(i),2);
    summ += mean_curvature_squared(i);
  }
  avg = summ/dof_handler.n_dofs();
  std::cout << "avg mean curvature: " << avg << std::endl;
  /*}}}*/
}

template <int spacedim>
void LaplaceBeltramiProblem<spacedim>::output_results () const
{
  /*{{{*/
  DataOut<dim,DoFHandler<dim,spacedim> > data_out;
  data_out.attach_dof_handler (dof_handler);
  data_out.add_data_vector (mean_curvature_squared,
                            "computed_mean_curvature_squared",
                            DataOut<dim,DoFHandler<dim,spacedim> >::type_dof_data);


  data_out.add_data_vector (exact_solution_values,
                            "exact_solution",
                            DataOut<dim,DoFHandler<dim,spacedim> >::type_dof_data);
  data_out.build_patches (mapping,
                          mapping.get_degree());

  std::string filename ("./data/mean_curvature_squared-");
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
void LaplaceBeltramiProblem<spacedim>::compute_error (double a, double b, double c, Point<3> center) const
{
  /*{{{*/
  Vector<float> difference_per_cell_L2 (triangulation.n_active_cells());
  VectorTools::integrate_difference (mapping, dof_handler, mean_curvature_squared,
                                     ExactSolution<3>(a,b,c,center),
                                     difference_per_cell_L2,
                                     QGauss<dim>(2*fe.degree+1),
                                     VectorTools::L2_norm);
  
  Vector<float> difference_per_cell_Linfty (triangulation.n_active_cells());
  VectorTools::integrate_difference (mapping, dof_handler, mean_curvature_squared,
                                     ExactSolution<3>(a,b,c,center),
                                     difference_per_cell_Linfty,
                                     QGauss<dim>(2*fe.degree+1),
                                     VectorTools::Linfty_norm);


  std::cout << "L2 error = "
            << difference_per_cell_L2.l2_norm()
            << std::endl;
  std::cout << "Linfty error = "
            << difference_per_cell_Linfty.linfty_norm()
            << std::endl;
  /*}}}*/
}


template <int spacedim>
void LaplaceBeltramiProblem<spacedim>::run ()
{
  double a = 1; double b = 1; double c = 1;
  Point<3> center(0,0,0);
  
  make_grid_and_dofs(a,b,c,center);
  std::cout << "grid and dofs made " << std::endl;
  
  assemble_system ();
  std::cout << "system assembled " << std::endl;
  
  solve ();
  std::cout << "solved " << std::endl;
  
                            
  exact_solution_values.reinit(dof_handler.n_dofs());
  VectorTools::interpolate(dof_handler, 
                           ExactSolution<3>(a,b,c,center),
                           exact_solution_values);
  output_results ();
  std::cout << "results written" << std::endl;
  
  compute_error(a,b,c,center);
  std::cout << "error computed" << std::endl;
}

}


int main ()
{
  try
  {
    using namespace dealii;
    using namespace Step38;
    
    LaplaceBeltramiProblem<3> laplace_beltrami;
    laplace_beltrami.run();
    return 0;
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
      std::cout << "problem! " << std::endl;
      return 2;
    }
}
