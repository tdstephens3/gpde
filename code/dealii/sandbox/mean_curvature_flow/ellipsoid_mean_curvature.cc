/* ---------------------------------------------------------------------
 *
 * ellipsoid_mean_curvature.cc
*/

#include <deal.II/base/utilities.h>
#include <deal.II/base/quadrature_lib.h>
#include <deal.II/base/function.h>
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
#include <deal.II/lac/vector.h>
#include <deal.II/lac/solver_control.h>
#include <deal.II/lac/solver_cg.h>
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
    
    virtual void vector_value (const Point<spacedim> &p, Vector<double> &value) const;
    virtual double value (const Point<spacedim> &p, const unsigned int component = 0) const;
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

int main ()
{
  using namespace dealii;
  /*{{{*/
  
  //BlockVector<double> mean_curvature;


  ///////////// mesh and geometry ///////////////
  const int dim = 2; const int spacedim = 3;
  Triangulation<dim,spacedim>    triangulation;
  
  double a = 1.0; double b = 2.0; double c = 3.0;
  Point<spacedim> center(0,0,0);
  static Ellipsoid<dim,spacedim> ellipsoid(a,b,c,center);

  GridGenerator::hyper_sphere(triangulation,Point<spacedim>(0,0,0), 1.0);
  triangulation.set_all_manifold_ids(0);
  
  GridTools::transform(std_cxx11::bind(&Ellipsoid<dim,spacedim>::grid_transform, &ellipsoid, std_cxx11::_1),
                                       triangulation);

  triangulation.set_manifold (0, ellipsoid);
  triangulation.refine_global(2);
  std::cout << "Surface mesh has " << triangulation.n_active_cells()
            << " cells."
            << std::endl;
  /////////////////////////////////////////////
  


  ///////////// fem ////////////////////////////////
  const int fe_degree = 2;
  DoFHandler<dim,spacedim>       dof_handler(triangulation);
  FESystem<dim,spacedim> fe( FE_Q<dim,spacedim>(fe_degree), spacedim);
  MappingQGeneric<dim, spacedim> mapping(fe_degree);
  
  
  
  std::cout << "about to distribute dofs " << std::endl;
  dof_handler.distribute_dofs (fe);
  std::cout << "distributed dofs " << std::endl;
          


  
  BlockVector<double> identity_on_manifold(3, dof_handler.n_dofs());
  //DoFRenumbering::component_wise (dof_handler);
  std::cout << "block vector vec.size():  " << identity_on_manifold.size() << std::endl;
  std::cout << "dof.n_dofs():  " << dof_handler.n_dofs() << std::endl;
  VectorTools::interpolate(mapping, dof_handler, 
                           Identity<spacedim>(), 
                           identity_on_manifold);
  std::cout << "interpolated function onto dofs " << std::endl;




  //DynamicSparsityPattern dsp (dof_handler.n_dofs(), dof_handler.n_dofs());
  //DoFTools::make_sparsity_pattern (dof_handler, dsp);
  //sparsity_pattern.copy_from (dsp);

  const QGauss<dim>  quadrature_formula(2*fe.degree);
  FEValues<dim,spacedim> fe_values (mapping, fe, quadrature_formula,
                                    update_values              |
                                    update_gradients           |
                                    update_normal_vectors      |
                                    update_quadrature_points   |
                                    update_JxW_values);
              

  const unsigned int n_q_points      = quadrature_formula.size();
  const unsigned int dofs_per_cell   = fe.dofs_per_cell;
  
  for (typename DoFHandler<dim,spacedim>::active_cell_iterator
       cell = dof_handler.begin_active(),
       endc = dof_handler.end();
       cell!=endc; ++cell)
    {
      fe_values.reinit (cell);

      for (unsigned int q_point=0; q_point<n_q_points; ++q_point)
        for (unsigned int i=0; i<dofs_per_cell; ++i) 
        {
          Tensor<1,spacedim> shape_grad = fe_values.shape_grad(i,q_point);
          std::cout << shape_grad << std::endl;
        }
    }
  
  
  
  //const unsigned int        dofs_per_cell = fe.dofs_per_cell;
  //const unsigned int        n_q_points    = quadrature_formula.size();

  //std::vector<types::global_dof_index> local_dof_indices (dofs_per_cell);

  //for (typename DoFHandler<dim,spacedim>::active_cell_iterator
  //     cell = dof_handler.begin_active(),
  //     endc = dof_handler.end();
  //     cell!=endc; ++cell)
  //  {
  //    cell_mass_matrix = 0;
  //    cell_laplace_matrix = 0;

  //    fe_values.reinit (cell);

  //    for (unsigned int i=0; i<dofs_per_cell; ++i)
  //      for (unsigned int j=0; j<dofs_per_cell; ++j)
  //        for (unsigned int q_point=0; q_point<n_q_points; ++q_point)
  //        {
  //          cell_mass_matrix(i,j) += fe_values.shape_value(i,q_point) *
  //                                   fe_values.shape_value(j,q_point) *
  //                                   fe_values.JxW(q_point);
  //          cell_laplace_matrix(i,j) += fe_values.shape_grad(i,q_point) *
  //                                      fe_values.shape_grad(j,q_point) *
  //                                      fe_values.JxW(q_point);
  //        }

  //    cell->get_dof_indices (local_dof_indices);
  //    for (unsigned int i=0; i<dofs_per_cell; ++i)
  //      {
  //        for (unsigned int j=0; j<dofs_per_cell; ++j)
  //        {
  //          mass_matrix.add (local_dof_indices[i],
  //                           local_dof_indices[j],
  //                           cell_mass_matrix(i,j));
  //          laplace_matrix.add (local_dof_indices[i],
  //                              local_dof_indices[j],
  //                              cell_laplace_matrix(i,j));
  //        }
  //      }
  //  }
  //
  ////MatrixCreator::create_mass_matrix(mapping,dof_handler, quadrature_formula, mass_matrix);
  ////MatrixCreator::create_laplace_matrix(mapping,dof_handler, quadrature_formula, laplace_matrix);

  //old_solution.reinit (dof_handler.n_dofs());
  //solution.reinit (dof_handler.n_dofs());
  //system_rhs.reinit (dof_handler.n_dofs());


  /*}}}*/


  return 0;
}
