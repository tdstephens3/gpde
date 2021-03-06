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
#include <deal.II/lac/sparse_direct.h>
#include <deal.II/lac/precondition.h>
#include <deal.II/lac/precondition_block.h>
#include <deal.II/lac/iterative_inverse.h>
#include <deal.II/lac/block_sparse_matrix.h>
#include <deal.II/lac/block_matrix_array.h>
#include <deal.II/lac/identity_matrix.h>
#include <deal.II/lac/sparse_matrix.h>
#include <deal.II/lac/sparse_ilu.h>
#include <deal.II/lac/dynamic_sparsity_pattern.h>
#include <deal.II/dofs/dof_handler.h>
#include <deal.II/dofs/dof_accessor.h>
#include <deal.II/dofs/dof_tools.h>
#include <deal.II/dofs/dof_renumbering.h>
#include <deal.II/fe/fe_system.h>
#include <deal.II/fe/fe_values.h>
#include <deal.II/fe/fe_nothing.h>
#include <deal.II/fe/mapping_q_eulerian.h>
#include <deal.II/numerics/data_out.h>
#include <deal.II/numerics/vector_tools.h>
#include <deal.II/numerics/matrix_tools.h>

#include <stdio.h>
#include <fstream>
#include <iostream>
#include <typeinfo>

// additional cpp code that is reusable
#include "../utilities/my_manifolds_lib.cc"

using namespace dealii;

template <int spacedim>
class Initial_map_sphere_to_ellipsoid: public Function<spacedim>
{
/*{{{*/
  public:
    Initial_map_sphere_to_ellipsoid(double a,double b,double c) : Function<spacedim>(spacedim),abc_coeffs{a,b,c} {}
    
    virtual void vector_value_list (const std::vector<Point<spacedim> > &points,
                                            std::vector<Vector<double> >   &value_list) const;
    virtual void vector_value (const Point<spacedim> &p, Vector<double> &value) const;
    virtual double value (const Point<spacedim> &p, const unsigned int component = 0) const;
  private:
    double abc_coeffs[3];
/*}}}*/
};

template<int spacedim>
double Initial_map_sphere_to_ellipsoid<spacedim>::value(const Point<spacedim> &p, const unsigned int component)  const
{
  /*{{{*/
  return abc_coeffs[component]*p(component);
  /*}}}*/
}

template<int spacedim>
void Initial_map_sphere_to_ellipsoid<spacedim>::vector_value(const Point<spacedim> &p, Vector<double> &value) const
{
  /*{{{*/
  for (unsigned int c=0; c<this->n_components; ++c) 
  {
    value(c) = Initial_map_sphere_to_ellipsoid<spacedim>::value(p,c);
  }
  /*}}}*/
}

template <int spacedim>
void Initial_map_sphere_to_ellipsoid<spacedim>::vector_value_list (const std::vector<Point<spacedim> > &points,
                                            std::vector<Vector<double> >   &value_list) const
{
  /*{{{*/
  Assert (value_list.size() == points.size(),
          ExcDimensionMismatch (value_list.size(), points.size()));
  const unsigned int n_points = points.size();
  for (unsigned int p=0; p<n_points; ++p)
    Initial_map_sphere_to_ellipsoid<spacedim>::vector_value (points[p], value_list[p]);
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

class SchurComplement : public Subscriptor
{
public:
  SchurComplement (const BlockSparseMatrix<double> &system_matrix,
                   const IterativeInverse<Vector<double>> &A_inverse);
  void vmult (Vector<double>       &dst,
              const Vector<double> &src) const;
private:
  const SmartPointer<const BlockSparseMatrix<double> > system_matrix;
  const SmartPointer<const IterativeInverse<Vector<double> > > A_inverse;
  mutable Vector<double> tmp1, tmp2, tmp3, tmp4;
};
SchurComplement::
SchurComplement (const BlockSparseMatrix<double> &system_matrix,
                 const IterativeInverse<Vector<double>> &A_inverse)
  :
  system_matrix (&system_matrix),
  A_inverse (&A_inverse),
  tmp1 (system_matrix.block(0,0).m()),
  tmp2 (system_matrix.block(0,0).m()),
  tmp3 (system_matrix.block(0,0).m()),
  tmp4 (system_matrix.block(0,0).m())
{}
void SchurComplement::vmult (Vector<double>       &dst,
                             const Vector<double> &src) const
{
  /*{{{*/
  system_matrix->block(1,0).vmult (tmp1, src);  // Cv
  A_inverse->vmult (tmp2, tmp1);                // invM*Cv
  system_matrix->block(0,1).vmult (tmp3, tmp2); // B*invM*Cv
  system_matrix->block(0,0).vmult(tmp4,src);    
  dst += tmp4;                                  // Av - B*invM*Cv 
  dst -= tmp3;
  /*}}}*/
}

int main ()
{

  // definitions
  /*{{{*/

  static const unsigned int spacedim = 3;
  static const unsigned int dim = spacedim-1;
  
  Triangulation<dim,spacedim>   triangulation;
  double                        fe_degree = 2;
  FESystem<dim,spacedim>        fe(FE_Q<dim,spacedim>(fe_degree),spacedim);
  const unsigned int            global_refinements = 3;
  DoFHandler<dim,spacedim>      dof_handler(triangulation);
  //MappingQ<dim, spacedim>       mapping;
  //MappingQEulerian<dim,Vector<double>,spacedim> eulerian_mapping;
  
  BlockSparsityPattern          block_sparsity_pattern;
  
  BlockSparseMatrix<double>     system_matrix;
  BlockVector<double>           VH;
  BlockVector<double>           system_rhs;
  
  //Vector<double>                computed_mean_curvature_squared;
  //Vector<double>                computed_velocity_squared;
  //Vector<double>                exact_mean_curvature_squared;
  double kappa = 1;
  /*}}}*/

  // make grid and dofs
  /*{{{*/

  double a = 1; double b = 2; double c = 3;
  Point<3> center(0,0,0);
  
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
  DoFRenumbering::component_wise(dof_handler);

  std::cout << "Surface mesh has " << dof_handler.n_dofs()
            << " degrees of freedom."
            << std::endl;

  /* build an Eulerian mapping, MappingQEulerian eulerian_mapping, that approximates
   * grid_transform : sphere --> ellipsoid. This will be updated by the
   * velocity field computed at each time step.
  */
  Vector<double> deformation(dof_handler.n_dofs());
  VectorTools::interpolate(dof_handler, 
                           Initial_map_sphere_to_ellipsoid<spacedim>(a,b,c), 
                           deformation);
  //std::cout << "deformation has n_components: " << deformation.n_components() << std::endl;
  MappingQEulerian<dim,Vector<double>,spacedim> eulerian_mapping(1, dof_handler, deformation);


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

  // run
  /*{{{*/
  //
  //make_grid_and_dofs(a,b,c,center);
  //std::cout << "grid and dofs made " << std::endl;
            
  double time = 0.0;
  double end_time = 1e-1;
  double time_step = 1e-5;
  double max_time_step = 1e-2;
  double min_time_step = 1e-8;
  double max_allowable_displacement = 0.1;
  double max_disp = 0;
  //double max_velo  = 0;
  //double l2_norm_velo  = 0;
  bool write_mats              = false;
  bool compute_mesh_statistics = true;
  double min_mesh_diam      = 0;
  double init_surface_area  = 0;
  double delta_surface_area = 0;
  double init_volume        = 0;
  double delta_volume       = 0;

  
  int step = 0, write_solution_step = 0;
  //output_results(write_solution_step);
  //if (compute_mesh_statistics) {
  //  init_surface_area = compute_surface_area();
  //  init_volume       = compute_volume();
  //}
  while (time < end_time && time_step >= min_time_step)
  {
    /*{{{*/
    
    std::cout << "\n===================" << std::endl;
    printf("iteration:         %d\n", step);
    printf("current time:      %0.9f\n", time);
    printf("time_step:         %0.9f\n", time_step);
    if (compute_mesh_statistics) 
    {
      printf("min mesh diam:     %0.9f\n", min_mesh_diam);
      printf("delta surf. area:  %+0.3f %%\n", delta_surface_area);
      printf("delta volume:      %+0.3f %%\n", delta_volume);
    }
    std::cout << "-------------------" << std::endl;
    
    // assemble system
    /*{{{*/
    
    /*
     *    [   M      L-2*hL+0.5*d ][ V_n+1 ]   [  0  ]
     *    |                       ||       | = |     |
     *    [ -time_step*L          M      ][ H_n+1 ]   [ rhs ]
     *
     *    system_matrix*VH = system_rhs
     *    
     *    system_matrix has size 2*n_dofs x 2*n_dofs, 
     *    each block has size n_dofs x n_dofs, and
     *    rhs has size n_dofs
     *
     */
    
    system_matrix = 0;
    VH = 0;
    system_rhs = 0;
  
    const QGauss<dim>  quadrature_formula (2*fe.degree);
    FEValues<dim,spacedim> fe_values (eulerian_mapping, fe, quadrature_formula,
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
  
    
    Identity<spacedim> identity_on_manifold;
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
            
            local_hL(i,j) += scalar_product(2.0*identity_on_manifold.shape_grad(fe_values.normal_vector(q_point))*
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
        system_rhs.block(1)(local_dof_indices[i]) += local_rhs(i);
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
                                      - time_step*local_L(i,j));
          
          system_matrix.block(1,1).add (local_dof_indices[i],
                                        local_dof_indices[j],
                                        local_M(i,j));
  
        
  
        }
      }
    }
    //system_matrix.block(0,0) = 0;
    //system_matrix.block(0,1) = 0;
    //system_matrix.block(1,0) = 0;
    
    system_rhs.block(0) = 0;
    /*}}}*/
    
    std::cout << "system assembled" << std::endl;
    //std::cout << "max_norm rhs: "   << get_max_norm_wrt_cell(system_rhs.block(1)) << std::endl;
    std::cout << "l2_norm rhs: "   << system_rhs.block(1).l2_norm() << std::endl;
    
    try
    {
      // solve_using_umfpack
      /*{{{*/
        // equation: system_matrix*VH = system_rhs
        std::cout << "using SparseDirectUMFPACK" << std::endl;
        
        SparseDirectUMFPACK A_direct;
        
        A_direct.initialize(system_matrix);
      
        A_direct.vmult(VH, system_rhs);
        
        //try
        //{
        //  
        //  //std::cout << solver_control.last_step()
        //  //          << "  gmres iterations to obtain convergence.\n"
        //  //          << "  last value: "
        //  //          << solver_control.last_value()
        //  //          << "\n  last check: "
        //  //          << solver_control.last_check()
        //  //          << std::endl;
        //}
        //catch (dealii::SolverControl::NoConvergence &nc)
        //{
        //  std::cout << solver_control.last_step()
        //            << "  gmres iterations to obtain convergence.\n"
        //            << "  last value: "
        //            << solver_control.last_value()
        //            << "\n  last check"
        //            << solver_control.last_check()
        //            << std::endl;
        //  throw nc;
        //}
      /*}}}*/

      Vector<double> node_velocity = VH.block(0);
      //max_velo = get_max_norm_wrt_cell(VH.block(0));

      
      if (time_step*max_disp > max_allowable_displacement)
      {
        std::cout << "........." << std::endl;
        std::cout << "          time step too large" << std::endl;
        std::cout << "          attempted displacement of: " << time_step*max_disp << std::endl;
        time_step = std::max(0.9*time_step,min_time_step);
        std::cout << "          reducing time step" << std::endl;
        std::cout << "........." << std::endl;
      }
      else  // else: everything is fine
      {
        std::cout << "........." << std::endl;
        std::cout << "          success!" << std::endl;
        std::cout << "........." << std::endl;
        
        if (step%1==0) // write output
        {
          write_solution_step+=1;
          // output_results
          /*{{{*/
          
          VectorValuedSolutionSquared<spacedim> computed_velocity_squared("scalar_velocity");
          VectorValuedSolutionSquared<spacedim> computed_mean_curvature_squared("H2");
          
          DataOut<dim,DoFHandler<dim,spacedim> > data_out;
          data_out.attach_dof_handler (dof_handler);
          
          data_out.add_data_vector (VH.block(0), computed_velocity_squared);
          data_out.add_data_vector (VH.block(1), computed_mean_curvature_squared);


          std::vector<std::string> solution_names (spacedim, "vector_velocity");
          
          std::vector<DataComponentInterpretation::DataComponentInterpretation>
               data_component_interpretation(spacedim,
                                             DataComponentInterpretation::component_is_part_of_vector);


          data_out.add_data_vector (VH.block(0), solution_names,
                                    DataOut<dim,DoFHandler<dim,spacedim> >::type_dof_data,
                                    data_component_interpretation);
          
          
          //data_out.add_data_vector (exact_solution_values,
          //                          "exact_solution",
          //                          DataOut<dim,DoFHandler<dim,spacedim> >::type_dof_data);
          
          data_out.build_patches (eulerian_mapping,
                                  eulerian_mapping.get_degree());

          std::string filename ("./data/no_class_test_willmore_flow-" + Utilities::int_to_string(step, 5));
          filename += ".vtk";
          std::ofstream output (filename.c_str());
          data_out.write_vtk (output);
          /*}}}*/
        }

        std::cout << "max displacement: " << time_step*max_disp << std::endl;
        //std::cout << "H:                " << get_max_norm_wrt_cell(VH.block(1)) << std::endl;
        std::cout << "H.l2_norm:        " << VH.block(1).l2_norm() << std::endl;
        
        // move_mesh
        /*{{{*/
        std::cout << "    Moving mesh..." <<  std::endl;
        std::vector<bool> vertex_touched (triangulation.n_vertices(), false);

        
        //Vector<double> new_deformation(dof_handler.n_dofs());
        // move vertices of mesh using velocity
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
                const unsigned int vec_index = cell->vertex_dof_index(v,d);
                vertex_displacement[d] = time_step*node_velocity(vec_index);
                

                cell->vertex(v) += vertex_displacement;

                // update eulerian_mapping with velocity info
                deformation(vec_index) += time_step*node_velocity(vec_index);
                //std::cout << "deformation: " << time_step*node_velocity(vec_index) << std::endl;
              }
              if (vertex_displacement.distance(Point<spacedim>(0,0,0)) > max_disp)
                max_disp = vertex_displacement.distance(Point<spacedim>(0,0,0));
            }  
        }
        
        //Vector<double> deformation = new_deformation;
        // complete the update: mapping_eulerian with velocity
        MappingQEulerian<dim,Vector<double>,spacedim> eulerian_mapping = MappingQEulerian<dim,Vector<double>,spacedim>(1,dof_handler,deformation); 
        /*}}}*/

        //if (compute_mesh_statistics) {
        //  min_mesh_diam      = compute_min_mesh_diam();
        //  delta_surface_area = (compute_surface_area() - init_surface_area)/init_surface_area;
        //  delta_volume       = (compute_volume() - init_volume)/init_volume;
        //}
        
        step += 1;
        time += time_step;   
        
        if (step%500==0)  // increase time step
        {
          std::cout << "increasing time_step" << std::endl;
          time_step = std::min(1.25*time_step,max_time_step);
        }
        std::cout << "\n___________________" << std::endl;
      }
    }
    //catch (dealii::SolverControl::NoConvergence)
    //{
    //  std::cout << "\nooooooooooooooooooooooooooooooooooooooooooooooooooooo\n" << std::endl;
    //  std::cout << "            solver did not converge for this time step" << std::endl;
    //  std::cout << "\nooooooooooooooooooooooooooooooooooooooooooooooooooooo\n" << std::endl;
    //  std::cout << "decreasing time step" << std::endl;
    //  time_step = 0.75*time_step;
    //}
    catch (std::exception &exc) 
    {
        std::cerr << std::endl << std::endl
                  << "----------------------------------------------------"
                  << std::endl;
        std::cerr << "unknown exception, what does it say? " << std::endl
                  << exc.what() << std::endl
                  << "Aborting!" << std::endl
                  << "----------------------------------------------------"
                  << std::endl;
    }
    
  /*}}}*/
  }
  /*}}}*/


  // output results
//template <int spacedim>
//double VectorHelfrichFlow<spacedim>::compute_min_mesh_diam ()
//{
//  /*{{{*/
//  double diam = 0;
//  double min_diam = 1E9;
//  for (typename DoFHandler<dim,spacedim>::active_cell_iterator
//       cell = dof_handler.begin_active(),
//       endc = dof_handler.end();
//       cell!=endc; ++cell)
//  { 
//    diam = cell->diameter();
//    if (diam < min_diam)
//      min_diam = diam;
//  }
//  return min_diam;
//  /*}}}*/
//}
//
//  template <int spacedim>
//double VectorHelfrichFlow<spacedim>::compute_surface_area ()
//{
//  /*{{{*/
//  double surface_area = GridTools::volume(triangulation,eulerian_mapping);
//  return surface_area;
//  /*}}}*/
//}
//
//template<int spacedim>
//double VectorHelfrichFlow<spacedim>::compute_volume ()
//{
///*{{{*/
//  
//  // initialize an appropriate quadrature formula
//  const QGauss<dim> quadrature_formula (eulerian_mapping.get_degree() + 1);
//  const unsigned int n_q_points = quadrature_formula.size();
//  
//  // we really want the JxW values from the FEValues object, but it
//  // wants a finite element. create a cheap element as a dummy
//  // element
//  FE_Nothing<dim,spacedim> dummy_fe;
//  FEValues<dim,spacedim> fe_values (eulerian_mapping, dummy_fe, quadrature_formula,
//                                    update_JxW_values | update_normal_vectors);
//  
//  typename Triangulation<dim,spacedim>::active_cell_iterator
//  cell = triangulation.begin_active(),
//  endc = triangulation.end();
//   
//  double cell_area = 0;
//  double volume = 0;
//  
//  // compute the integral quantities by quadrature
//  for (; cell!=endc; ++cell)
//  {
//    fe_values.reinit (cell);
//    Point<spacedim> sum_of_verts(0.0,0.0,0.0);
//    Point<spacedim> center_point;
//    
//    for (int v=0; v<4; ++v)
//      sum_of_verts += cell->vertex(v);
//    center_point = sum_of_verts/4.0;
//    
//    //Tensor<1,spacedim> q_normal({0,0,0});
//    //Tensor<1,spacedim> unit_normal;
//    Point<spacedim> q_normal(0,0,0);
//    Point<spacedim> unit_normal;
//    cell_area = 0;
//    for (unsigned int q=0; q<n_q_points; ++q)
//    {
//      q_normal  += fe_values.normal_vector(q);
//      cell_area += fe_values.JxW(q);
//    }
//    unit_normal = -q_normal/n_q_points; 
//    volume += cell_area*center_point*unit_normal;
//  }
//  return (1.0/3.0)*volume;
///*}}}*/
//}
//
//template <int spacedim>
//void VectorHelfrichFlow<spacedim>::write_matrices() 
//{
//  std::ofstream A_file, B_file, C_file, D_file, rhs0_file, rhs1_file;
//  A_file.open("data/A.txt");
//  B_file.open("data/B.txt");
//  C_file.open("data/C.txt");
//  D_file.open("data/D.txt");
//  rhs0_file.open("data/rhs0.txt");
//  rhs1_file.open("data/rhs1.txt");
//  std::cout << "about to call print_formatted" << std::endl;
//  system_matrix.block(0,0).print_formatted(A_file,8,true,0,"0");
//  system_matrix.block(0,1).print_formatted(B_file,8,true,0,"0");
//  system_matrix.block(1,0).print_formatted(C_file,8,true,0,"0");
//  system_matrix.block(1,1).print_formatted(D_file,8,true,0,"0");
//  system_rhs.block(0).print(rhs0_file,8);
//  system_rhs.block(1).print(rhs1_file,8);
//  A_file.close();
//  B_file.close();
//  C_file.close();
//  D_file.close();
//  rhs0_file.close();
//  rhs1_file.close();
//  std::cout << "matrices written" << std::endl;
//}
//
//template <int spacedim>
//void VectorHelfrichFlow<spacedim>::solve_using_schur() 
//{
///*{{{*/
////  std::cout << "using Schur" << std::endl;
////  
////  PreconditionIdentity identity;
////  IterativeInverse<Vector<double> > m_inverse;
////  m_inverse.initialize(system_matrix.block(0,0), identity);
////  m_inverse.solver.select("gmres");
////  static ReductionControl inner_control(1000, 1.0e-10);
////  m_inverse.solver.set_control(inner_control);
////  Vector<double> tmp (VH.block(0).size());
////  
////  Vector<double> schur_rhs (VH.block(0).size());
////  m_inverse.vmult (tmp, system_rhs.block(1));
////  tmp *= -1;
////  system_matrix.block(0,1).vmult (schur_rhs, tmp); // schur_rhs = -B*invM*rhs
////  
////  ApproximateSchurComplement approx_schur(system_matrix);
////  IterativeInverse<Vector<double> > preconditioner;
////  preconditioner.initialize(approx_schur, identity);
////  preconditioner.solver.select("gmres");
////  preconditioner.solver.set_control(inner_control);
////
////  SolverControl solver_control (system_matrix.block(0,0).m(),
////                                1e-8*schur_rhs.l2_norm());
////  
////  SolverGMRES<>    gmres (solver_control);
////  try
////  {
////    gmres.solve (SchurComplement(system_matrix, m_inverse),
////              VH.block(0),
////              schur_rhs,
////              PreconditionIdentity());
////    
////
////    std::cout << solver_control.last_step()
////              << "  Schur complement iterations to obtain convergence."
////              << std::endl;
////  }
////  catch (dealii::SolverControl::NoConvergence &nc)
////  {
////    std::cerr << "Failure in Schur Complement solver!" << std::endl;
////    throw nc;
////  }
////
////  std::cout << "made it through Schur part, now solving for VH.block(1)" << std::endl;
////  system_matrix.block(1,0).vmult (tmp, VH.block(0));
////  tmp *= -1;
////  tmp += system_rhs.block(1);
////  m_inverse.vmult (VH.block(1), tmp);
////   
///*}}}*/
//}
  return 0;
}

