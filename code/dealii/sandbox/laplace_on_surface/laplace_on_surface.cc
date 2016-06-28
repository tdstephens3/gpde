/* ---------------------------------------------------------------------
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


// @sect3{Include files}

// If you've read through step-4 and step-7, you will recognize that we have
// used all of the following include files there already. Consequently, we
// will not explain their meaning here again.
#include <deal.II/base/quadrature_lib.h>
#include <deal.II/base/function.h>
#include <deal.II/grid/tria.h>
#include <deal.II/grid/tria_iterator.h>
#include <deal.II/grid/tria_accessor.h>
#include <deal.II/grid/manifold_lib.h>
#include <deal.II/grid/grid_generator.h>
#include <deal.II/grid/grid_in.h>
#include <deal.II/grid/grid_out.h>
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

#include <deal.II/opencascade/boundary_lib.h>
#include <deal.II/opencascade/utilities.h>

#include <fstream>
#include <iostream>


namespace laplace_on_surface
{
  using namespace dealii;

              
//// class LaplaceBeltramiProblem
  template <int spacedim>
  class LaplaceBeltramiProblem
  {
  public:
    LaplaceBeltramiProblem (const unsigned degree = 2);
    void run ();

  private:
    static const unsigned int dim = spacedim-1;

    void make_grid_and_dofs ();
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
  };


//// class Solution
  template <int dim>
  class Solution  : public Function<dim>
  {
  public:
    Solution () : Function<dim>() {}

    virtual double value (const Point<dim>   &p,
                          const unsigned int  component = 0) const;

    virtual Tensor<1,dim> gradient (const Point<dim>   &p,
                                    const unsigned int  component = 0) const;

  };
// function definitions for Solution class
/*{{{*/
  template <>
  double Solution<2>::value (const Point<2> &p, const unsigned int) const {
    return ( -2. * p(0) * p(1) );
  }

  template <>
  Tensor<1,2> Solution<2>::gradient (const Point<2>   &p, const unsigned int) const {
    Tensor<1,2> return_value;
    return_value[0] = -2. * p(1) * (1 - 2. * p(0) * p(0));
    return_value[1] = -2. * p(0) * (1 - 2. * p(1) * p(1));

    return return_value;
  }

  template <>
  double Solution<3>::value (const Point<3> &p, const unsigned int) const {
    return (std::sin(numbers::PI * p(0)) *
            std::cos(numbers::PI * p(1))*exp(p(2)));
  }

  template <>
  Tensor<1,3> Solution<3>::gradient (const Point<3>   &p, const unsigned int) const {
    using numbers::PI;

    Tensor<1,3> return_value;

    return_value[0] = PI *cos(PI * p(0))*cos(PI * p(1))*exp(p(2));
    return_value[1] = -PI *sin(PI * p(0))*sin(PI * p(1))*exp(p(2));
    return_value[2] = sin(PI * p(0))*cos(PI * p(1))*exp(p(2));

    return return_value;
  }
/*}}}*/



// class RightHandSide
  template <int dim>
  class RightHandSide : public Function<dim> {
  
  public:
    RightHandSide () : Function<dim>() {}

    virtual double value (const Point<dim>   &p,
                          const unsigned int  component = 0) const;
  };
// function definitions for RightHandSide class
/*{{{*/
  template <>
  double RightHandSide<2>::value (const Point<2> &p, const unsigned int /*component*/) const {
  /*{{{*/
    return ( -8. * p(0) * p(1) );
  /*}}}*/
  }

  template <>
  double RightHandSide<3>::value (const Point<3> &p, const unsigned int /*component*/) const {
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
/*}}}*/


  // constructor for LaplaceBeltramiOperator
  template <int spacedim>
  LaplaceBeltramiProblem<spacedim>::LaplaceBeltramiProblem (const unsigned degree)
    :
    fe (degree),
    dof_handler(triangulation),
    mapping (degree)
  {}

  template <int spacedim>
  void LaplaceBeltramiProblem<spacedim>::make_grid_and_dofs () {
  /*{{{*/
      
    //const std::string cad_file_name = "ellipsoid_cad_surface.iges";
    const std::string cad_file_name = "sphere.iges";
    TopoDS_Shape cad_surface = OpenCASCADE::read_IGES(cad_file_name, 1);

    const double tolerance = OpenCASCADE::get_shape_tolerance(cad_surface) * 5;

    std::vector<TopoDS_Compound>  compounds;
    std::vector<TopoDS_CompSolid> compsolids;
    std::vector<TopoDS_Solid>     solids;
    std::vector<TopoDS_Shell>     shells;
    std::vector<TopoDS_Wire>      wires;

    OpenCASCADE::extract_compound_shapes(cad_surface,
                                         compounds,
                                         compsolids,
                                         solids,
                                         shells,
                                         wires);

    std::ifstream in;
    //std::string in_mesh_filename = "ellipsoid_mesh_140.ucd";
    std::string in_mesh_filename = "sphere_mesh.ucd";
    
    in.open(in_mesh_filename.c_str());

    GridIn<2,3> gi;
    gi.attach_triangulation(triangulation);
    gi.read (in);

    Triangulation<2,3>::active_cell_iterator cell = triangulation.begin_active();
    cell->set_all_manifold_ids(1);


    Assert(wires.size() > 0,
           ExcMessage("I could not find any wire in the CAD file you gave me. Bailing out."));

    static OpenCASCADE::NormalProjectionBoundary<2,3> normal_projector(cad_surface, tolerance);
    
    triangulation.set_manifold(1,normal_projector);
  
    triangulation.refine_global(2);
    
    
    // output results
    const std::string out_filename = "sphere_surface.vtk";
    std::ofstream logfile(out_filename.c_str());
    GridOut grid_out;
    grid_out.write_vtk(triangulation, logfile);
    cout << "output file written" << endl;
    //


    std::cout << "Surface mesh has " << triangulation.n_active_cells()
              << " active cells."
              << std::endl;

    dof_handler.distribute_dofs (fe);

    std::cout << "Surface mesh has " << dof_handler.n_dofs()
              << " degrees of freedom."
              << std::endl;

    DynamicSparsityPattern dsp (dof_handler.n_dofs(), dof_handler.n_dofs());
    DoFTools::make_sparsity_pattern (dof_handler, dsp);
    sparsity_pattern.copy_from (dsp);

    system_matrix.reinit (sparsity_pattern);

    solution.reinit (dof_handler.n_dofs());
    system_rhs.reinit (dof_handler.n_dofs());
  /*}}}*/
  }

  template <int spacedim>
  void LaplaceBeltramiProblem<spacedim>::assemble_system () {
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

        cout << "about to reinit..." << endl;
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

    //std::map<types::global_dof_index,double> boundary_values;
    //VectorTools::interpolate_boundary_values (mapping,
    //                                          dof_handler,
    //                                          0,
    //                                          Solution<spacedim>(),
    //                                          boundary_values);

    //MatrixTools::apply_boundary_values (boundary_values,
    //                                    system_matrix,
    //                                    solution,
    //                                    system_rhs,false);
  /*}}}*/
  }

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

  template <int spacedim>
  void LaplaceBeltramiProblem<spacedim>::run () {
  /*{{{*/
    cout << "entered run ()" << endl;
    
    make_grid_and_dofs();
    cout << "made grid and dofs" << endl;

    assemble_system ();
    cout << "assembled system" << endl;
    
    solve ();
    cout << "solved" << endl;
    
    output_results ();
    compute_error ();
  /*}}}*/
  }
}


// @sect3{The main() function}

// The remainder of the program is taken up by the <code>main()</code>
// function. It follows exactly the general layout first introduced in step-6
// and used in all following tutorial programs:
int main ()
{
  try
    {
      using namespace dealii;
      using namespace laplace_on_surface;

      const unsigned int spacedim = 3;
      LaplaceBeltramiProblem<spacedim> laplace_beltrami;
      
      cout << "about to run laplace_beltrami..." << endl;
      laplace_beltrami.run();
      cout << "ran laplace_beltrami..." << endl;
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
