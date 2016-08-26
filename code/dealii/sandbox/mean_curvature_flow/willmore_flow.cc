/* ---------------------------------------------------------------------
 *
 * willmore_flow.cc
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
#include <deal.II/lac/block_vector.h>
#include <deal.II/lac/vector.h>
#include <deal.II/lac/solver_control.h>
#include <deal.II/lac/solver_cg.h>
#include <deal.II/lac/solver_gmres.h>
#include <deal.II/lac/precondition.h>
#include <deal.II/lac/block_sparse_matrix.h>
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
class VectorWillmoreFlow
{
  /*{{{*/
  public:
    VectorWillmoreFlow (const unsigned degree = 2);
    void run ();
  
  private:
    static const unsigned int dim = spacedim-1;
  
    void make_grid_and_dofs (double,double,double,Point<spacedim>);
    void assemble_system ();
    void solve ();
    void output_results () const;
    //void compute_error (double, double, double, Point<3>) const;
  
  
    Triangulation<dim,spacedim>   triangulation;
    FESystem<dim,spacedim>        fe; 
    double                        degree = 2;
    DoFHandler<dim,spacedim>      dof_handler;
    MappingQ<dim, spacedim>       mapping;
  
    BlockSparsityPattern          sparsity_pattern;
    BlockSparseMatrix<double>     system_matrix;
    BlockVector<double>           solution;
    BlockVector<double>           system_rhs;
    
    Vector<double>                computed_mean_curvature_squared;
    //Vector<double>                exact_mean_curvature_squared;
    /*}}}*/
};


template <int spacedim>
class ComputedMeanCurvatureSquared : public DataPostprocessorScalar<spacedim>
{
/*{{{*/
public:
  ComputedMeanCurvatureSquared ();
  virtual
  void
  compute_derived_quantities_vector (const std::vector<Vector<double> >                    &uh,
                                     const std::vector<std::vector<Tensor<1, spacedim> > > &duh,
                                     const std::vector<std::vector<Tensor<2, spacedim> > > &dduh,
                                     const std::vector<Point<spacedim> >                   &normals,
                                     const std::vector<Point<spacedim> >                   &evaluation_points,
                                     std::vector<Vector<double> >                          &computed_quantities) const;
/*}}}*/
};

template <int spacedim>
ComputedMeanCurvatureSquared<spacedim>::ComputedMeanCurvatureSquared () : DataPostprocessorScalar<spacedim> ("computed_mean_curvature_squared", update_values) {}

template <int spacedim> 
void ComputedMeanCurvatureSquared<spacedim>::compute_derived_quantities_vector (const std::vector<Vector<double> >     &uh,
                                                               const std::vector<std::vector<Tensor<1, spacedim> > >   & /*duh*/,
                                                               const std::vector<std::vector<Tensor<2, spacedim> > >   & /*dduh*/,
                                                               const std::vector<Point<spacedim> >                     & /*normals*/,
                                                               const std::vector<Point<spacedim> >                     & /*evaluation_points*/,
                                                               std::vector<Vector<double> >                            &computed_quantities) const
{
/*{{{*/
  Assert(computed_quantities.size() == uh.size(),
         ExcDimensionMismatch (computed_quantities.size(), uh.size()));
  
  for (unsigned int i=0; i<computed_quantities.size(); i++)
    {
      Assert(computed_quantities[i].size() == 1, ExcDimensionMismatch (computed_quantities[i].size(), 1));
      Assert(uh[i].size() == 3, ExcDimensionMismatch (uh[i].size(), 3));
      computed_quantities[i](0) = uh[i](0)*uh[i](0) + uh[i](1)*uh[i](1) + uh[i](2)*uh[i](2) ;
    }
/*}}}*/
}


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


template <int spacedim>
VectorWillmoreFlow<spacedim>::VectorWillmoreFlow (const unsigned degree)
  :
  fe(FE_Q<dim,spacedim>(degree),spacedim),
  dof_handler(triangulation),
  mapping (degree)
{}


template <int spacedim>
void VectorWillmoreFlow<spacedim>::make_grid_and_dofs (double a, double b, double c, Point<spacedim> center)
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
            << " cells,\n"
            << "                 " << triangulation.n_used_vertices() 
            << " used vertices"
            << std::endl;

  dof_handler.distribute_dofs (fe);

  std::cout << "Surface mesh has " << dof_handler.n_dofs()
            << " degrees of freedom."
            << std::endl;

  /*    [  M    L-hL+d ][ V_n+1 ]   [  0  ]
   *    |              ||       | = |     |
   *    [ -L      M    ][ H_n+1 ]   [ rhs ]
   *
   *    system_matrix*VH = system_rhs
   *
   * 
   */

  BlockDynamicSparsityPattern dsp (2,2);
  dsp.block(0,0).reinit(dof_handler.n_dofs(),dof_handler.n_dofs());  
  dsp.block(0,1).reinit(dof_handler.n_dofs(),dof_handler.n_dofs());
  dsp.block(1,0).reinit(dof_handler.n_dofs(),dof_handler.n_dofs());
  dsp.block(1,1).reinit(dof_handler.n_dofs(),dof_handler.n_dofs());
  dsp.collect_sizes();

  DoFTools::make_sparsity_pattern (dof_handler, dsp);
  sparsity_pattern.copy_from (dsp);

  system_matrix.reinit(sparsity_pattern);

  solution.reinit(2);
  solution.block(0).reinit (dof_handler.n_dofs());
  solution.block(1).reinit (dof_handler.n_dofs());
  solution.collect_sizes();

  system_rhs.reinit(2);
  system_rhs.block(0).reinit (dof_handler.n_dofs());
  system_rhs.block(1).reinit (dof_handler.n_dofs());
  system_rhs.collect_sizes();
 
  /*}}}*/
}

template <int spacedim>
void VectorWillmoreFlow<spacedim>::assemble_system ()
{
  /*{{{*/
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

  FullMatrix<double>  local_M (dofs_per_cell, dofs_per_cell);
  FullMatrix<double>  local_L (dofs_per_cell, dofs_per_cell);
  FullMatrix<double>  local_hL (dofs_per_cell, dofs_per_cell);
  FullMatrix<double>  local_d (dofs_per_cell, dofs_per_cell);
  Vector<double>      local_rhs (dofs_per_cell);

  
  const FEValuesExtractors::Vector W(0);
  std::vector<types::global_dof_index> local_dof_indices (dofs_per_cell);
  
  for (typename DoFHandler<dim,spacedim>::active_cell_iterator
       cell = dof_handler.begin_active(),
       endc = dof_handler.end();
       cell!=endc; ++cell)
    {
      local_M   = 0;
      local_L   = 0;
      local_hL  = 0;
      local_d   = 0;
      local_rhs = 0;

      fe_values.reinit (cell);

      for (unsigned int i=0; i<dofs_per_cell; ++i)
        for (unsigned int j=0; j<dofs_per_cell; ++j)
          for (unsigned int q_point=0; q_point<n_q_points; ++q_point)
          {
            
            local_M(i,j)  += fe_values[W].value(i,q_point)*
                             fe_values[W].value(j,q_point)*
                             fe_values.JxW(q_point);
            
            local_L(i,j)  += fe_values[W].gradient(i,q_point)*
                             fe_values[W].gradient(j,q_point)*
                             fe_values.JxW(q_point);
            
            local_hL(i,j) += fe_values[W].gradient(i,q_point)*
                             fe_values[W].gradient(j,q_point)*
                             fe_values.JxW(q_point);
            local_hL(i)   += scalar_product(identity_on_manifold.shape_grad(fe_values.normal_vector(q_point))*
                                            fe_values[W].gradient(i,q_point),
                                            fe_values[W].gradient(j,q_point)
                                           )* fe_values.JxW(q_point);
            local_d(i,j)  += fe_values[W].divergence(i,q_point)*
                             fe_values[W].divergence(j,q_point)*
                             fe_values.JxW(q_point);
          }

      for (unsigned int i=0; i<dofs_per_cell; ++i)
        for (unsigned int q_point=0; q_point<n_q_points; ++q_point)
        {
          local_rhs(i) += scalar_product(fe_values[curv_components].gradient(i,q_point),
                                         identity_on_manifold.shape_grad(fe_values.normal_vector(q_point))
                                        )*fe_values.JxW(q_point);
        }
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
  /*}}}*/
}

template <int spacedim>
void VectorWillmoreFlow<spacedim>::solve ()
{
  /*{{{*/
  std::cout << "ABOUT TO SOLVE!" << std::endl;
  std::cout << solution.block(0).size() << std::endl;
  SolverControl solver_control (solution.block(0).size(), 1e-12 );
  SolverCG<>    cg (solver_control);


  cg.solve (system_matrix.block(0,0), solution.block(0), system_rhs.block(0), PreconditionIdentity());
  //cg.solve (system_matrix, solution, system_rhs, PreconditionIdentity());
  std::cout << "Solved all components" << std::endl;

  //std::vector< Vector<double> > curvature_components(n_q_points, Vector<double>(spacedim));
  //
  //for (typename DoFHandler<dim,spacedim>::active_cell_iterator
  //     cell = dof_handler.begin_active(),
  //     endc = dof_handler.end();
  //     cell!=endc; ++cell)
  //{
  //  fe_values.reinit(cell);
  //  fe_values.get_function_values(solution,curvature_components);
  //}

  //mean_curvature_squared.reinit (dof_handler.n_dofs());
  //double avg = 0; double summ = 0;
  //for (unsigned int i=0; i<dof_handler.n_dofs(); ++i )
  //{

  //  std::cout << "comps: " << solution.block(0)(i) << std::endl;
  //  //mean_curvature_squared(i) = pow(solution(i)(0),2) + pow(solution(i)(1),2) + pow(solution(i)(2),2);
  //  //summ += mean_curvature_squared(i);
  //}
  //avg = summ/dof_handler.n_dofs();
  //std::cout << "avg mean curvature: " << avg << std::endl;
  /*}}}*/
}

template <int spacedim>
void VectorWillmoreFlow<spacedim>::output_results () const
{
  /*{{{*/

  ComputedMeanCurvatureSquared<spacedim> computed_mean_curvature_squared;
  DataOut<dim,DoFHandler<dim,spacedim> > data_out;
  data_out.attach_dof_handler (dof_handler);

  data_out.add_data_vector (solution, computed_mean_curvature_squared);

  //data_out.add_data_vector (exact_solution_values,
  //                          "exact_solution",
  //                          DataOut<dim,DoFHandler<dim,spacedim> >::type_dof_data);
  
  data_out.build_patches (mapping,
                          mapping.get_degree());

  std::string filename ("./data/mean_curvature_squared-");
  filename += static_cast<char>('0'+spacedim);
  filename += "d.vtk";
  std::ofstream output (filename.c_str());
  data_out.write_vtk (output);
  /*}}}*/
}

// @sect4{VectorWillmoreFlow::compute_error}

// This is the last piece of functionality: we want to compute the error in
// the numerical solution. It is a verbatim copy of the code previously
// shown and discussed in step-7. As mentioned in the introduction, the
// <code>Solution</code> class provides the (tangential) gradient of the
// solution. To avoid evaluating the error only a superconvergence points,
// we choose a quadrature rule of sufficiently high order.
//template <int spacedim>
//void VectorWillmoreFlow<spacedim>::compute_error (double a, double b, double c, Point<3> center) const
//{
//  /*{{{*/
//  Vector<float> difference_per_cell_L2 (triangulation.n_active_cells());
//  VectorTools::integrate_difference (mapping, dof_handler, mean_curvature_squared,
//                                     ExactSolution<3>(a,b,c,center),
//                                     difference_per_cell_L2,
//                                     QGauss<dim>(2*fe.degree+1),
//                                     VectorTools::L2_norm);
//  
//  Vector<float> difference_per_cell_Linfty (triangulation.n_active_cells());
//  VectorTools::integrate_difference (mapping, dof_handler, mean_curvature_squared,
//                                     ExactSolution<3>(a,b,c,center),
//                                     difference_per_cell_Linfty,
//                                     QGauss<dim>(2*fe.degree+1),
//                                     VectorTools::Linfty_norm);
//
//
//  std::cout << "L2 error = "
//            << difference_per_cell_L2.l2_norm()
//            << std::endl;
//  std::cout << "Linfty error = "
//            << difference_per_cell_Linfty.linfty_norm()
//            << std::endl;
//  /*}}}*/
//}

template <int spacedim>
void VectorWillmoreFlow<spacedim>::run ()
{
  double a = 1; double b = 2; double c = 3;
  Point<3> center(0,0,0);
  

  make_grid_and_dofs(a,b,c,center);
  std::cout << "grid and dofs made " << std::endl;
  
  assemble_system ();
  std::cout << "system assembled " << std::endl;
  
  solve ();
  std::cout << "solved " << std::endl;
  
  //exact_solution_values.reinit(dof_handler.n_dofs());
  //VectorTools::interpolate(dof_handler, 
  //                         ExactSolution<3>(a,b,c,center),
  //                         exact_solution_values);
  output_results ();
  std::cout << "results written" << std::endl;
  //
  //compute_error(a,b,c,center);
  //std::cout << "error computed" << std::endl;
}

}


int main ()
{
  try
  {
    using namespace dealii;
    using namespace Step38;
    
    VectorWillmoreFlow<3> laplace_beltrami;
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
