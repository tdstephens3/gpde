/* ---------------------------------------------------------------------
 *
 * ellipsoid_mean_curvature.cc
*/

#include <deal.II/base/utilities.h>
#include <deal.II/base/quadrature_lib.h>
#include <deal.II/base/function.h>
#include <deal.II/base/tensor.h>
#include <deal.II/lac/block_vector.h>
#include <deal.II/grid/tria.h>
#include <deal.II/grid/tria_iterator.h>
#include <deal.II/grid/tria_accessor.h>

#include <deal.II/grid/manifold_lib.h>
#include <deal.II/grid/grid_generator.h>
#include <deal.II/grid/grid_tools.h>
#include <deal.II/grid/grid_out.h>
#include <deal.II/grid/grid_in.h>

#include <deal.II/lac/full_matrix.h>
#include <deal.II/lac/linear_operator.h>
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
#include <deal.II/dofs/dof_renumbering.h>
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
  Tensor<2,spacedim> full_shape_grad = Identity<spacedim>::shape_grad(unit_normal);
  Tensor<1,spacedim> grad_component;

  grad_component[0] = full_shape_grad[component][0];
  grad_component[1] = full_shape_grad[component][1];
  grad_component[2] = full_shape_grad[component][2];

  return grad_component;
  /*}}}*/
}

int main ()
{
  using namespace dealii;
  /*{{{*/
  
  ///////////// mesh and geometry ///////////////
  const int dim = 2; const int spacedim = 3;
  Triangulation<dim,spacedim>    triangulation;
  
  double a = 1.0; double b = 1.0; double c = 1.0;
  Point<spacedim> center(0,0,0);
  static Ellipsoid<dim,spacedim> ellipsoid(a,b,c,center);

  GridGenerator::hyper_sphere(triangulation,Point<spacedim>(0,0,0), 1.0);
  triangulation.set_all_manifold_ids(0);
  
  GridTools::transform(std_cxx11::bind(&Ellipsoid<dim,spacedim>::grid_transform, 
                       &ellipsoid, std_cxx11::_1), triangulation);

  triangulation.set_manifold (0, ellipsoid);
  triangulation.refine_global(2);
  std::cout << "Surface mesh has " << triangulation.n_active_cells()
            << " cells."
            << std::endl;
  
  

  ///////////// finite elements ////////////////////////////////
  const int fe_degree = 2;
  FESystem<dim,spacedim> fe( FE_Q<dim,spacedim>(fe_degree), spacedim);
  DoFHandler<dim,spacedim>  dof_handler(triangulation);
  MappingQGeneric<dim, spacedim> mapping(fe_degree);
  
  // vector-valued function on manifold //
  //Vector<double> identity_on_manifold(dof_handler.n_dofs());
  Identity<spacedim> identity_on_manifold;
  

  dof_handler.distribute_dofs (fe);
  DynamicSparsityPattern dsp (dof_handler.n_dofs(), dof_handler.n_dofs());
  DoFTools::make_sparsity_pattern (dof_handler, dsp);
  SparsityPattern sparsity_pattern;
  sparsity_pattern.copy_from (dsp);

  const QGauss<dim>  quadrature_formula(1+fe.degree);
  FEValues<dim,spacedim> fe_values (mapping, fe, quadrature_formula,
                                    update_quadrature_points   |
                                    update_normal_vectors      |
                                    update_values              |
                                    update_gradients           |
                                    update_JxW_values);
              
  /////////////////////// build system ////////////////////////////
  const unsigned int dofs_per_cell   = fe.dofs_per_cell;
  const unsigned int n_q_points      = quadrature_formula.size();
  std::vector<types::global_dof_index> local_dof_indices (dofs_per_cell);
  
  /// matrices, vectors, and scalars ///
  FullMatrix<double>             cell_mass_matrix(dofs_per_cell, dofs_per_cell);
  Vector<double>                 cell_rhs_x(dofs_per_cell);
  Vector<double>                 cell_rhs_y(dofs_per_cell);
  Vector<double>                 cell_rhs_z(dofs_per_cell);
  
  SparseMatrix<double>           mass_matrix;
  Vector<double>                 system_rhs_x;
  Vector<double>                 system_rhs_y;
  Vector<double>                 system_rhs_z;
  
  Vector<double>                 mean_curvature_x;
  Vector<double>                 mean_curvature_y;
  Vector<double>                 mean_curvature_z;
  Vector<double>                 mean_curvature_squared;

  mass_matrix.reinit (sparsity_pattern);
  system_rhs_x.reinit(dof_handler.n_dofs());
  system_rhs_y.reinit(dof_handler.n_dofs());
  system_rhs_z.reinit(dof_handler.n_dofs());
  
  typename DoFHandler<dim,spacedim>::active_cell_iterator cell = dof_handler.begin_active();
  for (; cell!=dof_handler.end(); ++cell)
  {
    cell_mass_matrix = 0;
    cell_rhs_x = 0;
    cell_rhs_y = 0;
    cell_rhs_z = 0;
    fe_values.reinit (cell);

    // create lhs mass matrix, (k*nu, v)_ij
    for (unsigned int i=0; i<dofs_per_cell; ++i)
      for (unsigned int j=0; j<dofs_per_cell; ++j)
        for (unsigned int q_point=0; q_point<n_q_points; ++q_point)
        {
          cell_mass_matrix(i,j) += fe_values.shape_value(i,q_point) *
                                   fe_values.shape_value(j,q_point) *
                                   fe_values.JxW(q_point);
        }
    // create rhs vector, (nabla_X id_X, nabla_X v)_i := \int nabla_X id_X : nabla_X v
    for (unsigned int i=0; i<dofs_per_cell; ++i)
      for (unsigned int q_point=0; q_point<n_q_points; ++q_point) 
      {
        cell_rhs_x(i) += fe_values.shape_grad_component(i,q_point,0)*
                         identity_on_manifold.shape_grad_component(fe_values.normal_vector(q_point),0)* 
                         fe_values.JxW(q_point);
        cell_rhs_y(i) += fe_values.shape_grad_component(i,q_point,1)*
                         identity_on_manifold.shape_grad_component(fe_values.normal_vector(q_point),1)* 
                         fe_values.JxW(q_point);
        cell_rhs_z(i) += fe_values.shape_grad_component(i,q_point,2)*
                         identity_on_manifold.shape_grad_component(fe_values.normal_vector(q_point),2)* 
                         fe_values.JxW(q_point);
      }
    cell->get_dof_indices (local_dof_indices);
    for (unsigned int i=0; i<dofs_per_cell; ++i)
    {
      system_rhs_x(local_dof_indices[i]) += cell_rhs_x(i);
      system_rhs_y(local_dof_indices[i]) += cell_rhs_y(i);
      system_rhs_z(local_dof_indices[i]) += cell_rhs_z(i);
      
      for (unsigned int j=0; j<dofs_per_cell; ++j)
        mass_matrix.add (local_dof_indices[i],
                         local_dof_indices[j],
                         cell_mass_matrix(i,j));
    }
  }

  system_rhs_x.equ(-1.0,system_rhs_x);
  system_rhs_y.equ(-1.0,system_rhs_y);
  system_rhs_z.equ(-1.0,system_rhs_z);
  /// solve   mass_matrix * mean_curvature_vector = - system_rhs_vector   for mean_curvature_vector
  
  SolverControl solver_control (mean_curvature_x.size(), 1e-7*system_rhs_x.l2_norm(),true);
  SolverCG<> cg_solver (solver_control);
  
  mean_curvature_x.reinit(dof_handler.n_dofs());
  mean_curvature_y.reinit(dof_handler.n_dofs());
  mean_curvature_z.reinit(dof_handler.n_dofs());
  
  cg_solver.solve(mass_matrix, 
                  mean_curvature_x, 
                  system_rhs_x, 
                  PreconditionIdentity());
  cg_solver.solve(mass_matrix, 
                  mean_curvature_y, 
                  system_rhs_y, 
                  PreconditionIdentity());
  cg_solver.solve(mass_matrix, 
                  mean_curvature_z, 
                  system_rhs_z, 
                  PreconditionIdentity());
  
  double summ = 0;
  double comp_squared = 0;
  mean_curvature_squared.reinit(dof_handler.n_dofs());
  mean_curvature_squared = 0;
  for (size_t i=0; i< dof_handler.n_dofs(); ++i) 
  {
    mean_curvature_squared(i) = pow(mean_curvature_x(i),2) + pow(mean_curvature_y(i),2) + pow(mean_curvature_z(i),2);
    summ += mean_curvature_squared(i);
    //printf("mean_curv_x,y,z: %0.12f, %0.12f, %0.12f\n",mean_curvature_x(i),mean_curvature_y(i), mean_curvature_z(i));
    //printf("%0.12f\n",mean_curvature_squared(i));
    //printf("%0.12f\n",mean_curvature_squared(i));
  }
  
  double avg = summ/dof_handler.n_dofs();
  std::cout << "avg curvature: " << avg << std::endl;
  
  DataOut<dim,DoFHandler<dim,spacedim> > data_out;
  data_out.attach_dof_handler (dof_handler);
  data_out.add_data_vector (mean_curvature_squared,
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
  return 0;
}

