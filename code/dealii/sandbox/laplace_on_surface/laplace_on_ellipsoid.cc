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


#include <deal.II/base/utilities.h>
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

#include <stdio.h>
#include <fstream>
#include <iostream>



using namespace dealii;

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
Ellipsoid<dim,spacedim>::Ellipsoid(double a, double b, double c) : center(0,0,0), SphericalManifold<dim,spacedim>(center), a(a), b(b),c(c)        
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


/* ------------------------ */

template <int spacedim>
class LaplaceBeltramiProblem
{
  /*{{{*/
public:
  
  LaplaceBeltramiProblem (const unsigned degree = 2);
  
  void run ();
  
  void set_timestep(double &timestep);
  void set_theta(double theta);
  void set_end_time(double end_time);
  
  double timestep;
  double theta;
  double end_time;

private:
  static const unsigned int dim = spacedim-1;

  void make_grid_and_dofs ();
  void assemble_mesh_and_manifold();
  void assemble_system ();
  void solve ();
  void output_results (int &step) const;
  void compute_error () const;


  Triangulation<dim,spacedim>   triangulation;
  FE_Q<dim,spacedim>            fe;
  DoFHandler<dim,spacedim>      dof_handler;
  MappingQ<dim, spacedim>       mapping;

  SparsityPattern               sparsity_pattern;
  SparseMatrix<double>          mass_matrix;
  SparseMatrix<double>          laplace_matrix;
  SparseMatrix<double>          system_matrix;
  SparseMatrix<double>          rhs_matrix;

  Vector<double>                system_rhs;
  Vector<double>                solution;
  Vector<double>                old_solution;

  /*}}}*/
};

template <int spacedim>
LaplaceBeltramiProblem<spacedim>::LaplaceBeltramiProblem (const unsigned degree) :
  fe (degree),
  dof_handler(triangulation),
  mapping (degree)
{}

template<int spacedim>
void LaplaceBeltramiProblem<spacedim>::set_timestep(double &timestep) 
{
  this->timestep = timestep;
}

template<int spacedim>
void LaplaceBeltramiProblem<spacedim>::set_theta(double theta) 
{
  this->theta = theta;
}

  template<int spacedim>
void LaplaceBeltramiProblem<spacedim>::set_end_time(double end_time) 
{
  this->end_time = end_time;
}



template <int spacedim>
class InitialCondition : public Function<spacedim>
{
/*{{{*/
  public:
    InitialCondition() : Function<spacedim>() {}
    
    virtual double value (const Point<spacedim> &p, const unsigned int component = 0) const;
/*}}}*/
};

template<int spacedim>
double InitialCondition<spacedim>::value(const Point<spacedim> &p, const unsigned int /*component*/) const
{
  /*{{{*/
  return exp(p(2));
  /*}}}*/
}

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

  /*}}}*/
}

template <int spacedim>
void LaplaceBeltramiProblem<spacedim>::assemble_system ()
{
  /*{{{*/

  const QGauss<dim>  quadrature_formula(2*fe.degree);
  FEValues<dim,spacedim> fe_values (mapping, fe, quadrature_formula,
                                    update_values              |
                                    update_gradients           |
                                    update_quadrature_points   |
                                    update_JxW_values);

  mass_matrix.reinit (sparsity_pattern);
  laplace_matrix.reinit (sparsity_pattern);

  const unsigned int        dofs_per_cell = fe.dofs_per_cell;
  const unsigned int        n_q_points    = quadrature_formula.size();

  FullMatrix<double>        cell_mass_matrix (dofs_per_cell, dofs_per_cell);
  FullMatrix<double>        cell_laplace_matrix (dofs_per_cell, dofs_per_cell);

  std::vector<types::global_dof_index> local_dof_indices (dofs_per_cell);

  for (typename DoFHandler<dim,spacedim>::active_cell_iterator
       cell = dof_handler.begin_active(),
       endc = dof_handler.end();
       cell!=endc; ++cell)
    {
      cell_mass_matrix = 0;
      cell_laplace_matrix = 0;

      fe_values.reinit (cell);

      for (unsigned int i=0; i<dofs_per_cell; ++i)
        for (unsigned int j=0; j<dofs_per_cell; ++j)
          for (unsigned int q_point=0; q_point<n_q_points; ++q_point)
          {
            cell_mass_matrix(i,j) += fe_values.shape_value(i,q_point) *
                                     fe_values.shape_value(j,q_point) *
                                     fe_values.JxW(q_point);
            cell_laplace_matrix(i,j) += fe_values.shape_grad(i,q_point) *
                                        fe_values.shape_grad(j,q_point) *
                                        fe_values.JxW(q_point);
          }

      cell->get_dof_indices (local_dof_indices);
      for (unsigned int i=0; i<dofs_per_cell; ++i)
        {
          for (unsigned int j=0; j<dofs_per_cell; ++j)
          {
            mass_matrix.add (local_dof_indices[i],
                             local_dof_indices[j],
                             cell_mass_matrix(i,j));
            laplace_matrix.add (local_dof_indices[i],
                                local_dof_indices[j],
                                cell_laplace_matrix(i,j));
          }
        }
    }
  
  //MatrixCreator::create_mass_matrix(mapping,dof_handler, quadrature_formula, mass_matrix);
  //MatrixCreator::create_laplace_matrix(mapping,dof_handler, quadrature_formula, laplace_matrix);

  old_solution.reinit (dof_handler.n_dofs());
  solution.reinit (dof_handler.n_dofs());
  system_rhs.reinit (dof_handler.n_dofs());
/*}}}*/
}

template <int spacedim>
void LaplaceBeltramiProblem<spacedim>::output_results (int &step) const
{
  /*{{{*/
  DataOut<dim,DoFHandler<dim,spacedim> > data_out;
  data_out.attach_dof_handler (dof_handler);
  data_out.add_data_vector (solution,
                            "ellipsoid_solution",
                            DataOut<dim,DoFHandler<dim,spacedim> >::type_dof_data);
  data_out.build_patches (mapping,
                          mapping.get_degree());

  std::string filename ("data/ellipsoid_solution-" + Utilities::int_to_string(step, 5));
  filename += ".vtk";
  std::ofstream output (filename.c_str());
  data_out.write_vtk (output);
  /*}}}*/
}


template<int spacedim>                                                                              
void LaplaceBeltramiProblem<spacedim>::run() 
{                                                                
/*{{{*/                                                                                        
//          template <int spacedim>
//           void LaplaceBeltramiProblem<spacedim>::run ()

            make_grid_and_dofs();

            assemble_system ();
            // should have access to: system_matrix, 
            //                        system_rhs, 
            //                        initial_condition
            //                        
            //                        empty old_solution,
            //                        empty solution,
            //
  
  

            // store initial condition in old_solution
            Vector<double> tmp;
            tmp.reinit (solution.size());
            system_matrix.reinit (sparsity_pattern);
            rhs_matrix.reinit (sparsity_pattern);
            
            VectorTools::interpolate(dof_handler, 
                                     InitialCondition<spacedim>(), 
                                     old_solution);
            

            double time = 0.0;
            int step = 0;
            solution = old_solution;
            output_results(step);
            while (time <= end_time)
            {
              time += timestep; step +=1;
              printf("time: %0.8f\n", time);

            
              //rhs_matrix.reinit (dof_handler.n_dofs());
              //system_rhs.reinit(solution.size());

              // assemble: [M + timestep*theta*A]*U^n = [M - timestep(1-theta)*A] * U^(n-1)
              //              system_matrix             rhs_matrix
              mass_matrix.vmult(system_rhs, old_solution);
              laplace_matrix.vmult(tmp, old_solution);
              system_rhs.add( -(1-theta) * timestep, tmp);

              system_matrix.copy_from(mass_matrix);
              system_matrix.add(theta*timestep,laplace_matrix);

              SolverControl solver_control (solution.size(), 1e-7 );
              SolverCG<> cg (solver_control);

              //PreconditionSSOR<> preconditioner;
              //preconditioner.initialize(system_matrix, 1.2);

              // equation: system_matrix*solution = system_rhs
              cg.solve(system_matrix, solution, system_rhs, PreconditionIdentity());
              
              // update solution
              old_solution = solution;
              
              output_results(step);
              
            }                                                                                 
 /*}}}*/
}                                                                                              

int main ()
{
  try
    {
      using namespace dealii;

      LaplaceBeltramiProblem<3> laplace_beltrami;
      double timestep = 0.01;
      double theta    = 0.5;
      double end_time = 20;
      laplace_beltrami.set_timestep(timestep);
      laplace_beltrami.set_theta(theta);
      laplace_beltrami.set_end_time(end_time);
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
