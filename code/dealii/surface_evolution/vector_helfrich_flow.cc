/* ---------------------------------------------------------------------
 *
 * vector_helfrich_flow.cc      August 2016
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
#include <deal.II/lac/block_matrix_array.h>
#include <deal.II/lac/identity_matrix.h>
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
#include <typeinfo>

// additional cpp code that is reusable
#include "../utilities/my_manifolds_lib.cc"

namespace VectorHelfrich
{
  using namespace dealii;

template <int spacedim>
class VectorHelfrichFlow
{
  /*{{{*/
  public:
    VectorHelfrichFlow (const unsigned int fe_degree);
    void run ();

    
  
  private:
    static const unsigned int dim = spacedim-1;
  
    void make_grid_and_dofs (double,double,double,Point<spacedim>);
    void assemble_system (double);
    void move_mesh (double, Vector<double>) const;
    void output_results (int &step) const;
    //void compute_error (double, double, double, Point<3>) const;
    double get_max_norm_wrt_cell(Vector<double>); 
  
  
    Triangulation<dim,spacedim>   triangulation;
    FESystem<dim,spacedim>        fe; 
    double                        fe_degree;
    const unsigned int            global_refinements = 3;
    DoFHandler<dim,spacedim>      dof_handler;
    MappingQ<dim, spacedim>       mapping;
  
    BlockSparsityPattern          block_sparsity_pattern;
    
    BlockSparseMatrix<double>     system_matrix;
    BlockVector<double>           VH;
    BlockVector<double>           system_rhs;
    
    //Vector<double>                computed_mean_curvature_squared;
    //Vector<double>                computed_velocity_squared;
    //Vector<double>                exact_mean_curvature_squared;
    double kappa = 0.5;
    /*}}}*/
};

template <int spacedim>
class VectorValuedSolutionSquared : public DataPostprocessorScalar<spacedim>
{
/*{{{*/
public:
  VectorValuedSolutionSquared (std::string = "dummy");
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
VectorValuedSolutionSquared<spacedim>::VectorValuedSolutionSquared (std::string data_name) : DataPostprocessorScalar<spacedim> (data_name, update_values) {}

template <int spacedim> 
void VectorValuedSolutionSquared<spacedim>::compute_derived_quantities_vector (const std::vector<Vector<double> >     &uh,
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
    
    virtual Tensor<2,spacedim> symmetric_grad(const Tensor<1,spacedim> &unit_normal) const;
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
Tensor<2,spacedim> Identity<spacedim>::symmetric_grad(const Tensor<1,spacedim> &unit_normal) const
{
  /*{{{*/
  Tensor<2,spacedim> eye, shape_grad, shape_grad_T;
  eye = 0; eye[0][0] = 1; eye[1][1] = 1; eye[2][2] = 1;
  Tensor<2,spacedim> nnT;
  nnT = outer_product(unit_normal,unit_normal);
  shape_grad = eye - nnT;
  shape_grad_T = transpose(shape_grad);
  return shape_grad + shape_grad_T;
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
VectorHelfrichFlow<spacedim>::VectorHelfrichFlow (const unsigned int fe_degree)
  :
  fe(FE_Q<dim,spacedim>(fe_degree),spacedim),
  dof_handler(triangulation),
  mapping (fe_degree)
{}

template <int spacedim>
void VectorHelfrichFlow<spacedim>::make_grid_and_dofs (double a, double b, double c, Point<spacedim> center)
{
  /*{{{*/
  static Ellipsoid<dim,spacedim> ellipsoid(a,b,c,center);

  GridGenerator::hyper_sphere(triangulation,center, 1);
  
  triangulation.set_all_manifold_ids(0);
  
  GridTools::transform(std_cxx11::bind(&Ellipsoid<dim,spacedim>::grid_transform, &ellipsoid, std_cxx11::_1), 
                       triangulation);

  triangulation.set_manifold (0, ellipsoid);
  triangulation.refine_global(global_refinements);

  std::cout << "Surface mesh has " << triangulation.n_active_cells()
            << " cells,\n"
            << "                 " << triangulation.n_used_vertices() 
            << " used vertices"
            << std::endl;

  dof_handler.distribute_dofs (fe);

  std::cout << "Surface mesh has " << dof_handler.n_dofs()
            << " degrees of freedom."
            << std::endl;

  /*
   *    [   M      L-2*hL+0.5*d ][ V_n+1 ]   [  0  ]
   *    |                       ||       | = |     |
   *    [ -zn*L          M      ][ H_n+1 ]   [ rhs ]
   *
   *    system_matrix*VH = system_rhs
   *    
   *    system_matrix has size 2*n_dofs x 2*n_dofs, 
   *    each block has size n_dofs x n_dofs, and
   *    rhs has size n_dofs
   *
   */

  BlockDynamicSparsityPattern dsp (2,2);
  dsp.block(0,0).reinit (dof_handler.n_dofs(), dof_handler.n_dofs());
  dsp.block(0,1).reinit (dof_handler.n_dofs(), dof_handler.n_dofs());
  dsp.block(1,0).reinit (dof_handler.n_dofs(), dof_handler.n_dofs());
  dsp.block(1,1).reinit (dof_handler.n_dofs(), dof_handler.n_dofs());
  dsp.collect_sizes();
 
  DoFTools::make_sparsity_pattern (dof_handler, dsp.block(0,0)); 
  DoFTools::make_sparsity_pattern (dof_handler, dsp.block(0,1)); 
  DoFTools::make_sparsity_pattern (dof_handler, dsp.block(1,0)); 
  DoFTools::make_sparsity_pattern (dof_handler, dsp.block(1,1)); 
  
  block_sparsity_pattern.copy_from (dsp);
  system_matrix.reinit (block_sparsity_pattern);
  
  VH.reinit (2);
  VH.block(0).reinit (dof_handler.n_dofs());
  VH.block(1).reinit (dof_handler.n_dofs());
  VH.collect_sizes();
  
  system_rhs.reinit (2);
  system_rhs.block(0).reinit (dof_handler.n_dofs());
  system_rhs.block(1).reinit (dof_handler.n_dofs());
  system_rhs.collect_sizes();
 
  /*}}}*/
}


template <int spacedim>
void VectorHelfrichFlow<spacedim>::assemble_system (double zn)
{
  /*{{{*/
  Identity<spacedim> identity_on_manifold;

  const QGauss<dim>  quadrature_formula (2*fe.degree);
  FEValues<dim,spacedim> fe_values (mapping, fe, quadrature_formula,
                                    update_values              |
                                    update_normal_vectors      |
                                    update_gradients           |
                                    update_quadrature_points   |
                                    update_JxW_values);


  const unsigned int  dofs_per_cell = fe.dofs_per_cell;
  const unsigned int  n_q_points    = quadrature_formula.size();

  FullMatrix<double>  local_M   (dofs_per_cell, dofs_per_cell);
  FullMatrix<double>  local_L   (dofs_per_cell, dofs_per_cell);
  FullMatrix<double>  local_hL  (dofs_per_cell, dofs_per_cell);
  FullMatrix<double>  local_d   (dofs_per_cell, dofs_per_cell);
  Vector<double>      local_rhs (dofs_per_cell);

  
  const FEValuesExtractors::Vector W (0);
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
          
          local_L(i,j)  += scalar_product(fe_values[W].gradient(i,q_point),
                                          fe_values[W].gradient(j,q_point)
                                         )* fe_values.JxW(q_point);
          
          local_hL(i,j) += scalar_product(identity_on_manifold.symmetric_grad(fe_values.normal_vector(q_point))*
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
        local_rhs(i) += scalar_product(fe_values[W].gradient(i,q_point),
                                       identity_on_manifold.shape_grad(fe_values.normal_vector(q_point))
                                      )*fe_values.JxW(q_point);
      }

    cell->get_dof_indices (local_dof_indices);
    for (unsigned int i=0; i<dofs_per_cell; ++i)
    {
      for (unsigned int j=0; j<dofs_per_cell; ++j)
      {
        system_matrix.block(0,0).add (local_dof_indices[i],
                                      local_dof_indices[j],
                                      local_M(i,j));

        system_matrix.block(0,1).add (local_dof_indices[i],
                                      local_dof_indices[j],
                               kappa*(local_L(i,j)
                                    - local_hL(i,j) 
                                    + 0.5*local_d(i,j)));
        
        system_matrix.block(1,0).add (local_dof_indices[i],
                                      local_dof_indices[j],
                                    - zn*local_L(i,j));
        
        system_matrix.block(1,1).add (local_dof_indices[i],
                                      local_dof_indices[j],
                                      local_M(i,j));

      

      }
    system_rhs.block(1)(local_dof_indices[i]) += local_rhs(i);
    }
  }
  //system_matrix.block(0,0) = 0;
  //system_matrix.block(0,1) = 0;
  //system_matrix.block(1,0) = 0;
  
  system_rhs.block(0) = 0;
  /*}}}*/
}

template <int spacedim>
void VectorHelfrichFlow<spacedim>::output_results (int &step) const
{
  /*{{{*/

  VectorValuedSolutionSquared<spacedim> computed_velocity_squared("vel");
  VectorValuedSolutionSquared<spacedim> computed_mean_curvature_squared("H2");
  
  DataOut<dim,DoFHandler<dim,spacedim> > data_out;
  data_out.attach_dof_handler (dof_handler);

  data_out.add_data_vector (VH.block(0), computed_velocity_squared);
  data_out.add_data_vector (VH.block(1), computed_mean_curvature_squared);

  //data_out.add_data_vector (exact_solution_values,
  //                          "exact_solution",
  //                          DataOut<dim,DoFHandler<dim,spacedim> >::type_dof_data);
  
  data_out.build_patches (mapping,
                          mapping.get_degree());

  std::string filename ("./data/test_willmore_flow-" + Utilities::int_to_string(step, 5));
  filename += ".vtk";
  std::ofstream output (filename.c_str());
  data_out.write_vtk (output);
  /*}}}*/
}

template <int spacedim>
void VectorHelfrichFlow<spacedim>::move_mesh (double zn, Vector<double> node_velocity) const
{
  /*{{{*/
  std::cout << "    Moving mesh..." <<  std::endl;
  std::vector<bool> vertex_touched (triangulation.n_vertices(), false);

  for (typename DoFHandler<2,spacedim>::active_cell_iterator
       cell = dof_handler.begin_active ();
       cell != dof_handler.end(); ++cell)
  {
    for (unsigned int v=0; v<GeometryInfo<2>::vertices_per_cell; ++v)
      if (vertex_touched[cell->vertex_index(v)] == false)
      { 
        vertex_touched[cell->vertex_index(v)] = true;
        
        // add displacement to vertex
        Point<spacedim> vertex_displacement;
        for (unsigned int d=0; d<spacedim; ++d)
        {
          vertex_displacement[d] = zn*node_velocity(cell->vertex_dof_index(v,d));
          cell->vertex(v) += vertex_displacement;
        }
      }  
  }
  /*}}}*/
}

template <int spacedim>
double VectorHelfrichFlow<spacedim>::get_max_norm_wrt_cell(Vector<double> vector_data) 
{
/*{{{*/
  const unsigned int dim = spacedim-1;
  const QGauss<dim>  quadrature_formula (2*fe.degree);
  const unsigned int n_q_points = quadrature_formula.size();
  FEValues<dim,spacedim> fe_values (fe, quadrature_formula, update_values);
  std::vector<Vector<double> > vector_data_values(n_q_points, Vector<double>(spacedim));
  

  double max_wrt_cell = 0;
  typename DoFHandler<dim,spacedim>::active_cell_iterator
  cell = dof_handler.begin_active(),
  endc = dof_handler.end();
  for (; cell!=endc; ++cell)
  {
    fe_values.reinit (cell);
    fe_values.get_function_values (vector_data, vector_data_values);
    for (unsigned int q=0; q<n_q_points; ++q)
      {
        Tensor<1,spacedim> vector_values;
        for (unsigned int i=0; i<spacedim; ++i)
          vector_values[i] = vector_data_values[q](i);
        max_wrt_cell = std::max (max_wrt_cell, vector_values.norm());
      }
  }
  return max_wrt_cell;
/*}}}*/
}

template <int spacedim>
void VectorHelfrichFlow<spacedim>::run ()
{
  /*{{{*/
  double a = 1; double b = 2; double c = 3;
  Point<3> center(0,0,0);
  
  make_grid_and_dofs(a,b,c,center);
  std::cout << "grid and dofs made " << std::endl;
            
  double time = 0.0;
  double end_time = 5.0;
  double time_step = 5e-7;
  double max_time_step = 1e-3;
  double min_time_step = 1e-7;
  double max_allowable_displacement = 2e-5;
  double max_velo  = 0;
  double solver_tol; 
  
  int step = 0;
  while (time < end_time)
  {
    /*{{{*/
    
    std::cout << "\n===================" << std::endl;
    printf("iteration:    %d\n", step);
    printf("current time: %0.8f\n", time);
    printf("time_step:    %0.8f\n", time_step);
    std::cout << "-------------------" << std::endl;
    
    assemble_system(time_step);
    solver_tol = 1e-7*system_rhs.linfty_norm();
    
    std::cout << "system assembled" << std::endl;
    std::cout << "max_norm rhs: " << get_max_norm_wrt_cell(system_rhs.block(1)) << std::endl;
    std::cout << "solver_tol:   " << solver_tol  << std::endl;
    
    SolverControl solver_control (VH.size(), solver_tol);
    SolverGMRES< BlockVector<double> > gmres (solver_control);
    //SolverCG< BlockVector<double> > cg (solver_control);
    PreconditionIdentity preconditioner;
    
    try
    {
      // equation: system_matrix*VH = system_rhs
      gmres.solve(system_matrix, VH, system_rhs, preconditioner);
     
      max_velo = get_max_norm_wrt_cell(VH.block(0));
      
      if (time_step*max_velo > max_allowable_displacement)
      {
        std::cout << "........." << std::endl;
        std::cout << "          time step too large" << std::endl;
        std::cout << "          attempted displacement of: " << time_step*max_velo << std::endl;
        time_step = std::max(0.9*time_step,min_time_step);
        std::cout << "          reducing time step" << std::endl;
        std::cout << "........." << std::endl;
      }
      else  // else: everything is fine
      {
        std::cout << "........." << std::endl;
        std::cout << "          success!" << std::endl;
        std::cout << "........." << std::endl;
        
        step += 1;
        time += time_step;   
        
        move_mesh(time_step,VH.block(0));
        
        output_results(step);
        
        std::cout << "max displacement: " << time_step*max_velo  << std::endl;
        std::cout << "H:                " << get_max_norm_wrt_cell(VH.block(1)) << std::endl;
        if (step%5==0)
        {
          std::cout << "increasing time_step" << std::endl;
          time_step = std::min(1.25*time_step,max_time_step);
        }
        std::cout << "\n___________________" << std::endl;
      }
    }
    catch (dealii::SolverControl::NoConvergence)
    {
      std::cout << "\nooooooooooooooooooooooooooooooooooooooooooooooooooooo\n" << std::endl;
      std::cout << "            solver did not converge for this time step" << std::endl;
      std::cout << "\nooooooooooooooooooooooooooooooooooooooooooooooooooooo\n" << std::endl;
      time_step = std::max(0.75*time_step,min_time_step);
    }
    
  /*}}}*/
  }
  


  bool able_to_start = false;
  while (able_to_start == false)
  {
  /*{{{*/
    printf("time_step:    %0.8f\n", time_step);
    assemble_system(time_step);
    solver_tol = 1e-11*system_rhs.linfty_norm();
    std::cout << "solver_tol        " << solver_tol  << std::endl;
    SolverControl solver_control (VH.size(), solver_tol);
    SolverGMRES< BlockVector<double> > gmres (solver_control);
    //SolverCG< BlockVector<double> > cg (solver_control);

    PreconditionIdentity preconditioner;

    try
    {
      // equation: system_matrix*VH = system_rhs
      gmres.solve(system_matrix, VH, system_rhs, preconditioner);
      
      max_velo = get_max_norm_wrt_cell(VH.block(0));
      if (time_step*max_velo < max_allowable_displacement)
      {
        able_to_start = true;
        move_mesh(time_step,VH.block(0));
      }
      else
        time_step = std::max(0.5*time_step,min_time_step);
    }
    catch (dealii::SolverControl::NoConvergence)
    {
      std::cout << "solver did not converge for this time step" << std::endl;
      time_step = std::max(0.5*time_step,min_time_step);
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
    }
  /*}}}*/
  }  
  
}
}  

int main ()
{
  try
  {
    using namespace dealii;
    using namespace VectorHelfrich;
    
    const unsigned int spacedim = 3;
    VectorHelfrichFlow<spacedim> laplace_beltrami(2);
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

