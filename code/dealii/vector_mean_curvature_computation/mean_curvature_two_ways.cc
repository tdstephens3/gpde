/* ---------------------------------------------------------------------
 *
 * refinement_and_smoothing.cc      Nov 10, 2016
 *
 *  Author: TOM STEPHENS, August 2016
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
 */

#include <deal.II/base/quadrature_lib.h>
#include <deal.II/base/function.h>
#include <deal.II/grid/tria.h>
#include <deal.II/grid/tria_iterator.h>
#include <deal.II/grid/tria_accessor.h>
#include <deal.II/grid/manifold_lib.h>
#include <deal.II/grid/grid_tools.h>
#include <deal.II/grid/grid_refinement.h>
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
#include <deal.II/numerics/error_estimator.h>
#include <deal.II/numerics/solution_transfer.h>

#include <stdio.h>
#include <fstream>
#include <iostream>
#include <typeinfo>

namespace VectorHelfrich
{
  using namespace dealii;


/* ********************************** 
 *
 * VectorHelfrichFlow<spacedim> class definition
 *
 * */
template <int spacedim>
class VectorHelfrichFlow
{
  /*{{{*/
  public:
    VectorHelfrichFlow () ;
    void run ();

    
  
  private:
    static const unsigned int dim = spacedim-1;
    const double c0 = 0;
  
    void make_grid_and_global_refine (const unsigned int initial_global_refinement);
    void setup_dofs ();
    void initialize_global_euler_vector(const double &,const double &,const double &);
    void apply_initial_values ();
    void refine_mesh (const unsigned int min_grid_level,
                      const unsigned int max_grid_level);
    
    void compute_initial_Hn (const double &);
    void compute_scalar_H(); 
    
    
    void assemble_bending_system (const double &, const bool initial_assembly=false);
    
    //void assemble_curvature_system ();
    void update_mapping (const double &);
    void output_results (const double &, const int &);
    //void solve_using_gmres(); 
    void solve_using_umfpack(const double &time_step, const bool apply_surface_area_constraints, const bool apply_volume_constraints); 
    //void compute_error (double, double, double, Point<3>) const;
    //double compute_max_norm_wrt_cell(Vector<double>); 
    double compute_min_mesh_diam();
    double compute_surface_area(const Vector<double> &euler_vector);
    double compute_volume(const Vector<double> &euler_vector);
    double compute_helfrich_energy (const double &lambda, const double &rho, 
                                    const double &current_surface_area, 
                                    const double &original_surface_area, 
                                    const double &current_volume, 
                                    const double &original_volume);

    double integrate_div_func_on_surface (const Vector<double> &euler_vector, const Vector<double> &func);
    double integrate_normal_dot_func_on_surface (const Vector<double> &euler_vector, const Vector<double> &func);
    Tensor<1,2> eff (const Vector<double> &weighted_euler_vector);
    void update_global_euler_vector(const double &time_step, const Vector<double> &total_displacement);
    void gc_euler_vector_update();
    void compute_weighted_euler_vector(Vector<double>                     &weighted_euler_vector,
                                       const double                       &time_step, 
                                       const std::vector<Vector<double> > &displacements, 
                                       const std::vector<double >         &displacement_weights);

    Triangulation<dim,spacedim>     triangulation;
    const unsigned int              global_refinements = 3;
    
    
    /* - data structures for bending energy - */
    const unsigned int              bending_fe_degree = 2;
    FESystem<dim,spacedim>          bending_fe; 
    DoFHandler<dim,spacedim>        bending_dof_handler;
    
    ConstraintMatrix                bending_constraints;
    BlockSparsityPattern            bending_block_sparsity_pattern;
    BlockSparseMatrix<double>       bending_matrix;
    
    BlockVector<double>             bending_rhs;  // bending velocity and vector mean curvature rhs
    BlockVector<double>             V_bending_Hn; // bending velocity and vector mean curvature
    Vector<double>                  Hn;           // vector mean curvature
    
    /* - data structures for geometrically consistent mesh modification - */
    SparsityPattern                 gc_sparsity_pattern;
    SparseMatrix<double>            gc_matrix;
    Vector<double>                  gc_rhs;   
    Vector<double>                  interpolated_X;   // identity on refined meshes
    Vector<double>                  gc_X;             // geometically consistent X
    Vector<double>                  global_euler_vector;
    Vector<double>                  gc_euler_vector;
    Vector<double>                  V_constrained;
    
    /* - data structures for scalar parameters - */
    const unsigned int              scalar_fe_degree = 2;
    FE_Q<dim,spacedim>              scalar_fe; 
    DoFHandler<dim,spacedim>        scalar_dof_handler;
    
    ConstraintMatrix                scalar_constraints;
    BlockSparsityPattern            scalar_block_sparsity_pattern;
    BlockSparseMatrix<double>       scalar_block_matrix;
    
    SparsityPattern                 scalar_sparsity_pattern;
    SparseMatrix<double>            scalar_matrix;
    
    BlockVector<double>             scalar_block_rhs;       // block rhs for scalar parameters
    Vector<double>                  bending_modulus;        // function values for bending modulus   
    Vector<double>                  spontaneous_curvature;  // function values for spontaneous curvature     
    Vector<double>                  scalar_rhs;
    
    Vector<double>                  scalar_H;               // scalar mean curvature
    Vector<double>                  deviation;              // scalar_H - c0
    
    /* - data structures for surface_area and volume constraints - */
    Vector<double>                  surface_area_rhs;
    Vector<double>                  volume_rhs;
    
    Vector<double>                  V_bending;         // bending velocity       (See Bonito, et al Parametric FEM paper)
    Vector<double>                  V_surface_area;    // surface area constraint velocity
    Vector<double>                  V_volume;          // volume constraint velocity       (See Bonito, et al Parametric FEM paper)

    double lambda,rho;
                 
    /*}}}*/
};


/* ********************************** 
 *
 * Function<spacedim> classes 
 *
 * */
/*{{{*/

template <int spacedim>
class Identity : public Function<spacedim>
{
/*{{{*/
  public:
    Identity() : Function<spacedim>(3) {}
    
    virtual void vector_value_list (const std::vector<Point<spacedim> > &points,
                                            std::vector<Vector<double> >   &value_list) const;
    virtual void vector_value (const Point<spacedim> &p, Vector<double> &value) const;
    virtual double value (const Point<spacedim> &p, const unsigned int component) const;
    
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
    vector_value (points[p], value_list[p]);
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
class Initial_map_sphere_to_ellipsoid: public Function<spacedim>
{
/*{{{*/
  public:
    Initial_map_sphere_to_ellipsoid(double a, double b, double c) : Function<spacedim>(3), abc_coeffs{a,b,c}  {}
    
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
  
  double norm_p = p.distance(Point<spacedim>(0,0,0));
  return abc_coeffs[component]*p(component)/norm_p - p(component);   

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
class BendingModulus : public Function<spacedim>
{
/*{{{*/
  public:
    BendingModulus() : Function<spacedim>() {}
    
    //virtual void vector_value_list (const std::vector<Point<spacedim> > &points,
                                            //std::vector<Vector<double> >   &value_list) const;
    virtual double value (const Point<spacedim> &p, const unsigned int=0) const;
    virtual void value_list (const std::vector<Point<spacedim> > &points,
                             std::vector<double> &value_list, const unsigned int component=0) const;
    virtual Tensor<1,spacedim> gradient(const Point<spacedim> &p, const unsigned int component=0)  const;
    virtual Tensor<1,spacedim> shape_grad(const Point<spacedim> &p, const Tensor<1,spacedim> unit_normal) const;
    
  private:
    double k = 1.0;
/*}}}*/
};

template<int spacedim>
double BendingModulus<spacedim>::value(const Point<spacedim> &p, const unsigned int)  const
{
  /*{{{*/
  (void)p; // while this function is constant, want to supress unused variable compiler warnings
  double bend_mod = k;
  
  return bend_mod; 
  /*}}}*/
}

template <int spacedim>
void BendingModulus<spacedim>::value_list (const std::vector<Point<spacedim> > &space_points,
                                       std::vector<double> &value_list, const unsigned int) const
{
  /*{{{*/
  Assert (value_list.size() == space_points.size(),
          ExcDimensionMismatch (value_list.size(), space_points.size()));
  const unsigned int n_points = space_points.size();
  for (unsigned int p=0; p<n_points; ++p)
    value_list[p] = value (space_points[p]);
  /*}}}*/
}

template <int spacedim>
Tensor<1,spacedim> BendingModulus<spacedim>::gradient(const Point<spacedim> &p, const unsigned int) const
{
  /*{{{*/
  (void)p; // while this function is constant, want to supress unused variable compiler warnings
  Tensor<1,spacedim> grad;

  grad[0] = 0; 
  grad[1] = 0; 
  grad[2] = 0; 
  
  return grad;
  /*}}}*/
}

template <int spacedim>
Tensor<1,spacedim> BendingModulus<spacedim>::shape_grad(const Point<spacedim> &p, const Tensor<1,spacedim> unit_normal) const
{
  /*{{{*/
  Tensor<1,spacedim> grad; //, grad_n, grad_n_nT;
  
  grad      = gradient(p); 
  //grad_n    = grad*unit_normal;
  //grad_n_nT = outer_product(grad_n,unit_normal);
  
  return grad - (grad*unit_normal)*unit_normal;
  /*}}}*/
}


template <int spacedim>
class SpontaneousCurvature : public Function<spacedim>
{
/*{{{*/
  public:
    SpontaneousCurvature(const double c0) : Function<spacedim>(), c0(c0) {}
    
    //virtual void vector_value_list (const std::vector<Point<spacedim> > &points,
                                            //std::vector<Vector<double> >   &value_list) const;
    virtual double value (const Point<spacedim> &p, const unsigned int=0) const;
    virtual void value_list (const std::vector<Point<spacedim> > &points,
                             std::vector<double> &value_list, const unsigned int component=0) const;
    virtual Tensor<1,spacedim> gradient(const Point<spacedim> &p, const unsigned int component=0)  const;
    virtual Tensor<1,spacedim> shape_grad(const Point<spacedim> &p, const Tensor<1,spacedim> unit_normal) const;
    
  private:
    const double sigma2 =  0.2;
    const double c0;
/*}}}*/
};

template<int spacedim>
double SpontaneousCurvature<spacedim>::value(const Point<spacedim> &p, const unsigned int)  const
{
  /*{{{*/
  double spont_curv = 0;

  //spont_curv = c0*exp(-(p(0)*p(0))/sigma2);
  if (p(2) > 0) 
    spont_curv = c0*exp(-(p(0)*p(0) + p(1)*p(1))/sigma2);
  return spont_curv; 
  /*}}}*/
}

template <int spacedim>
void SpontaneousCurvature<spacedim>::value_list (const std::vector<Point<spacedim> > &space_points,
                                       std::vector<double> &value_list, const unsigned int) const
{
  /*{{{*/
  Assert (value_list.size() == space_points.size(),
          ExcDimensionMismatch (value_list.size(), space_points.size()));
  const unsigned int n_points = space_points.size();
  for (unsigned int p=0; p<n_points; ++p)
    value_list[p] = value (space_points[p]);
  /*}}}*/
}

template <int spacedim>
Tensor<1,spacedim> SpontaneousCurvature<spacedim>::gradient(const Point<spacedim> &p, const unsigned int) const
{
  /*{{{*/
  Tensor<1,spacedim> grad;

  grad[0] =  -2*p(0)/sigma2*value(p); 
  grad[1] =  -2*p(1)/sigma2*value(p); 
  grad[2] =   0;//-2*p(2)/sigma2*value(p); 
  return grad;
  /*}}}*/
}

template <int spacedim>
Tensor<1,spacedim> SpontaneousCurvature<spacedim>::shape_grad(const Point<spacedim> &p, const Tensor<1,spacedim> unit_normal) const
{
  /*{{{*/
  Tensor<1,spacedim> grad; //, grad_n, grad_n_nT;
  
  grad      = gradient(p); 
  //grad_n    = grad*unit_normal;
  //grad_n_nT = outer_product(grad_n,unit_normal);

  return grad - (grad*unit_normal)*unit_normal;
  /*}}}*/
}
/*}}}*/

/* ********************************** 
 *
 * DataPostProcessor<spacedim> classes and functions
 *
 * */
/*{{{*/
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
class VectorValuedSolutionNormed : public DataPostprocessorScalar<spacedim>
{
/*{{{*/
public:
  VectorValuedSolutionNormed (std::string = "dummy");
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
VectorValuedSolutionNormed<spacedim>::VectorValuedSolutionNormed (std::string data_name) : DataPostprocessorScalar<spacedim> (data_name, update_values) {}

template <int spacedim> 
void VectorValuedSolutionNormed<spacedim>::compute_derived_quantities_vector (const std::vector<Vector<double> >     &uh,
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
      computed_quantities[i](0) = sqrt(uh[i](0)*uh[i](0) + uh[i](1)*uh[i](1) + uh[i](2)*uh[i](2)) ;
    }
/*}}}*/
}



/*}}}*/


/* ********************************** 
 *
 * VectorHelfrichFlow<spacedim> functions
 *
 * */
/*{{{*/

template <int spacedim>
VectorHelfrichFlow<spacedim>::VectorHelfrichFlow ()
  :
  bending_fe(FE_Q<dim,spacedim>(bending_fe_degree),spacedim),
  bending_dof_handler(triangulation),
  scalar_fe(scalar_fe_degree),
  scalar_dof_handler(triangulation)
  {}


template <int spacedim>
double VectorHelfrichFlow<spacedim>::compute_min_mesh_diam ()
{
  /*{{{*/
  double diam = 0;
  double min_diam = 1E9;
  for (typename DoFHandler<dim,spacedim>::active_cell_iterator
       cell = bending_dof_handler.begin_active(),
       endc = bending_dof_handler.end();
       cell!=endc; ++cell)
  { 
    diam = cell->diameter();
    if (diam < min_diam)
      min_diam = diam;
  }
  return min_diam;
  /*}}}*/
}

template<int spacedim>
double VectorHelfrichFlow<spacedim>::integrate_div_func_on_surface (const Vector<double> &local_euler_vector, const Vector<double> &func)
{
  /*{{{*/
  const MappingQEulerian<dim,Vector<double>,spacedim> mapping(2, bending_dof_handler, local_euler_vector);
  
  const QGauss<dim> quadrature_formula (2*bending_fe.degree);
  FEValues<dim,spacedim> fe_values (mapping, bending_fe, quadrature_formula,
                                    update_values              |
                                    update_gradients           |
                                    update_quadrature_points   |
                                    update_JxW_values);

  const unsigned int n_q_points = quadrature_formula.size();
  const FEValuesExtractors::Vector W (0);
  std::vector<Tensor<2,spacedim> > local_func_gradients(n_q_points, Tensor<2,spacedim>());
  
  double integral = 0;
  for (typename DoFHandler<dim,spacedim>::active_cell_iterator
       cell = bending_dof_handler.begin_active(),
       endc = bending_dof_handler.end();
       cell!=endc; ++cell)
  { 
    fe_values.reinit (cell);
    fe_values[W].get_function_gradients(func,local_func_gradients);
    
    for (unsigned int q_point=0; q_point<n_q_points; ++q_point)
    {
      integral += (local_func_gradients[q_point][0][0] 
                 + local_func_gradients[q_point][1][1]        
                 + local_func_gradients[q_point][2][2] // divergence
                  )*fe_values.JxW(q_point);
    }
  }
  return integral;
  /*}}}*/
}

template<int spacedim>
double VectorHelfrichFlow<spacedim>::integrate_normal_dot_func_on_surface (const Vector<double> &local_euler_vector, const Vector<double> &func)
{
  /*{{{*/
  const MappingQEulerian<dim,Vector<double>,spacedim> mapping(2, bending_dof_handler, local_euler_vector);
  
  const QGauss<dim> quadrature_formula (2*bending_fe.degree);
  FEValues<dim,spacedim> fe_values (mapping, bending_fe, quadrature_formula,
                                    update_values              |
                                    update_normal_vectors      |
                                    update_gradients           |
                                    update_quadrature_points   |
                                    update_JxW_values);

  const unsigned int  n_q_points    = quadrature_formula.size();
  const FEValuesExtractors::Vector W (0);
                                           
  std::vector<Tensor<1,spacedim> > local_func_values(n_q_points, Tensor<1,spacedim>());
  double integral = 0;
  for (typename DoFHandler<dim,spacedim>::active_cell_iterator
       cell = bending_dof_handler.begin_active(),
       endc = bending_dof_handler.end();
       cell!=endc; ++cell)
  { 
    fe_values.reinit (cell);
    fe_values[W].get_function_values(func,local_func_values);
    
    for (unsigned int q_point=0; q_point<n_q_points; ++q_point)
    {
      integral -= fe_values.normal_vector(q_point)*  /* -= to account for direction of normal vector */ 
                  local_func_values[q_point]* 
                  fe_values.JxW(q_point);
    }
  }
  return integral;
  /*}}}*/
}

template <int spacedim>
Tensor<1,2> VectorHelfrichFlow<spacedim>::eff (const Vector<double> &local_euler_vector)   
{
  /*{{{*/
  Tensor<1,2> f;
  double original_surface_area = compute_surface_area(global_euler_vector);
  double original_volume       = compute_volume(global_euler_vector);

  f[0] = compute_surface_area(local_euler_vector) - original_surface_area;
  f[1] = compute_volume(local_euler_vector)       - original_volume;

  return f;
  /*}}}*/
}

template <int spacedim>
double VectorHelfrichFlow<spacedim>::compute_surface_area (const Vector<double> &local_euler_vector)
{
  /*{{{*/
  const MappingQEulerian<dim,Vector<double>,spacedim> mapping(2, bending_dof_handler, local_euler_vector);
  
  const QGauss<dim> quadrature_formula (2*bending_fe.degree);
  FEValues<dim,spacedim> fe_values (mapping, bending_fe, quadrature_formula,
                                    update_values              |
                                    update_normal_vectors      |
                                    update_gradients           |
                                    update_quadrature_points   |
                                    update_JxW_values);

  const unsigned int  n_q_points    = quadrature_formula.size();

  double surface_area = 0;
  for (typename DoFHandler<dim,spacedim>::active_cell_iterator
       cell = bending_dof_handler.begin_active(),
       endc = bending_dof_handler.end();
       cell!=endc; ++cell)
  { 
    fe_values.reinit (cell);
    
    for (unsigned int q_point=0; q_point<n_q_points; ++q_point)
      surface_area += fe_values.JxW(q_point);
  }
  return surface_area;
  /*}}}*/
}

template<int spacedim>
double VectorHelfrichFlow<spacedim>::compute_volume (const Vector<double> &local_euler_vector)
{
  /*{{{*/
  const MappingQEulerian<dim,Vector<double>,spacedim> mapping(2, bending_dof_handler, local_euler_vector);
  
  const QGauss<dim> quadrature_formula (2*bending_fe.degree);
  FEValues<dim,spacedim> fe_values (mapping, bending_fe, quadrature_formula,
                                    update_values              |
                                    update_normal_vectors      |
                                    update_gradients           |
                                    update_quadrature_points   |
                                    update_JxW_values);

  const unsigned int  n_q_points    = quadrature_formula.size();

  double volume = 0;
  for (typename DoFHandler<dim,spacedim>::active_cell_iterator
       cell = bending_dof_handler.begin_active(),
       endc = bending_dof_handler.end();
       cell!=endc; ++cell)
  { 
    fe_values.reinit (cell);
    
    for (unsigned int q_point=0; q_point<n_q_points; ++q_point) /* -= to account for direction of normal vector */
      volume -= fe_values.quadrature_point(q_point)*fe_values.normal_vector(q_point)*fe_values.JxW(q_point);
  }
  return (1.0/3.0)*volume;  
  /*}}}*/
}

template<int spacedim>
double VectorHelfrichFlow<spacedim>::compute_helfrich_energy (const double &lambda, const double &rho, 
                                                              const double &current_surface_area, 
                                                              const double &original_surface_area, 
                                                              const double &current_volume, 
                                                              const double &original_volume)
{
  /*{{{*/
  const MappingQEulerian<dim,Vector<double>,spacedim> mapping(2, bending_dof_handler, global_euler_vector);
  
  const QGauss<dim> quadrature_formula (2*scalar_fe.degree);
  FEValues<dim,spacedim> fe_values (mapping, scalar_fe, quadrature_formula,
                                    update_values              |
                                    update_normal_vectors      |
                                    update_gradients           |
                                    update_quadrature_points   |
                                    update_JxW_values);

  const unsigned int  n_q_points = quadrature_formula.size();
  std::vector<double> local_scalar_H_values(n_q_points, 0);
  std::vector<double> local_c0_values(n_q_points, 0);

  double bending_energy      = 0;
  double surface_area_energy = 0;
  double volume_energy       = 0;
  for (typename DoFHandler<dim,spacedim>::active_cell_iterator
       cell = scalar_dof_handler.begin_active(),
       endc = scalar_dof_handler.end();
       cell!=endc; ++cell)
  { 
    fe_values.reinit (cell);
    fe_values.get_function_values(scalar_H, local_scalar_H_values);
    fe_values.get_function_values(spontaneous_curvature, local_c0_values);
    
    for (unsigned int q_point=0; q_point<n_q_points; ++q_point) 
    {
      //bending_energy += local_Hn_values[q_point]*local_Hn_values[q_point]*fe_values.JxW(q_point);
      bending_energy += pow(local_scalar_H_values[q_point] - local_c0_values[q_point],2)*fe_values.JxW(q_point);
    }
  }

  surface_area_energy = lambda*(current_surface_area - original_surface_area);
  volume_energy       = rho*(current_volume - original_volume);
  
  return bending_energy + surface_area_energy + volume_energy;  
  /*}}}*/
}

template <int spacedim>
void VectorHelfrichFlow<spacedim>::compute_weighted_euler_vector(Vector<double>                     &weighted_euler_vector,
                                                                 const double                       &time_step, 
                                                                 const std::vector<Vector<double> > &displacements, 
                                                                 const std::vector<double >         &displacement_weights) 
{
  /*{{{*/
  Assert (displacements.size() == displacement_weights.size(),
          ExcDimensionMismatch (displacements.size(), displacement_weights.size()));

  weighted_euler_vector = Vector<double>(global_euler_vector);
  double weighted_displacement;
  for (unsigned int i=0;i<displacements[0].size(); ++i)
  {
    weighted_displacement = 0;
    for (unsigned int j=0;j<displacements.size(); ++j)
      weighted_displacement += displacement_weights[j]*displacements[j](i);  
    
    weighted_euler_vector(i) += time_step*weighted_displacement;
  }
  /*}}}*/
}

template <int spacedim>
void VectorHelfrichFlow<spacedim>::update_global_euler_vector(const double &time_step, const Vector<double> &total_displacement)
{
  /*{{{*/
  //Vector<double> disp  = Vector<double>(total_displacement);
  Vector<double> disp = total_displacement;
  disp *= time_step; 
  global_euler_vector += disp;
  //for (unsigned int i=0;i<total_displacement.size(); ++i)
  //  global_euler_vector(i) += time_step*total_displacement(i);
  /*}}}*/
}

template <int spacedim>
void VectorHelfrichFlow<spacedim>::make_grid_and_global_refine (const unsigned int initial_global_refinement)
{
  /*{{{*/
  /*  --- build geometry --- */
  GridGenerator::hyper_sphere(triangulation,Point<spacedim>(0,0,0),1.0);

  triangulation.set_all_manifold_ids(0);
  
  triangulation.refine_global(initial_global_refinement);
  /*}}}*/
}

template <int spacedim>
void VectorHelfrichFlow<spacedim>::setup_dofs ()
{
  /*{{{*/                     
  
  bending_dof_handler.distribute_dofs (bending_fe);
  scalar_dof_handler.distribute_dofs  (scalar_fe);
  
  bending_constraints.clear();
  scalar_constraints.clear();
  
  DoFTools::make_hanging_node_constraints(bending_dof_handler,
                                          bending_constraints);
  
  DoFTools::make_hanging_node_constraints(scalar_dof_handler,
                                          scalar_constraints);

  bending_constraints.close();
  scalar_constraints.close();

  /*---- build system block matrix for bending equations ------*/
  bending_matrix.clear();
  const unsigned int bending_dofs = bending_dof_handler.n_dofs();
  BlockDynamicSparsityPattern bending_block_dsp (2,2);
  bending_block_dsp.block(0,0).reinit (bending_dofs,bending_dofs);
  bending_block_dsp.block(0,1).reinit (bending_dofs,bending_dofs);
  bending_block_dsp.block(1,0).reinit (bending_dofs,bending_dofs);
  bending_block_dsp.block(1,1).reinit (bending_dofs,bending_dofs);
  bending_block_dsp.collect_sizes();
  
  DoFTools::make_sparsity_pattern (bending_dof_handler, 
                                   bending_block_dsp.block(0,0),
                                   bending_constraints,false); 
  DoFTools::make_sparsity_pattern (bending_dof_handler, 
                                   bending_block_dsp.block(0,1),
                                   bending_constraints,false); 
  DoFTools::make_sparsity_pattern (bending_dof_handler, 
                                   bending_block_dsp.block(1,0),
                                   bending_constraints,false); 
  DoFTools::make_sparsity_pattern (bending_dof_handler, 
                                   bending_block_dsp.block(1,1),
                                   bending_constraints,false); 
  
  //bending_block_sparsity_pattern.reinit(2,2);
  bending_block_sparsity_pattern.copy_from (bending_block_dsp);
 
  /*---- build system matrix for gcmm ------*/
  gc_matrix.clear();
  DynamicSparsityPattern gc_dsp(bending_dofs, bending_dofs);
  DoFTools::make_sparsity_pattern(bending_dof_handler, gc_dsp, bending_constraints, false);
  gc_sparsity_pattern.copy_from(gc_dsp);
  gc_matrix.reinit(gc_sparsity_pattern);
  gc_rhs.reinit(bending_dofs);
  

  /*---- build block system matrix for scalar equations ------*/
  scalar_block_matrix.clear();
  BlockDynamicSparsityPattern scalar_block_dsp (2,2);
  const unsigned int scalar_dofs = scalar_dof_handler.n_dofs();
  scalar_block_dsp.block(0,0).reinit (scalar_dofs,scalar_dofs);
  scalar_block_dsp.block(0,1).reinit (scalar_dofs,scalar_dofs);
  scalar_block_dsp.block(1,0).reinit (scalar_dofs,scalar_dofs);
  scalar_block_dsp.block(1,1).reinit (scalar_dofs,scalar_dofs);
  scalar_block_dsp.collect_sizes();
  
  DoFTools::make_sparsity_pattern (scalar_dof_handler, 
                                   scalar_block_dsp.block(0,0),
                                   scalar_constraints,false); 
  DoFTools::make_sparsity_pattern (scalar_dof_handler, 
                                   scalar_block_dsp.block(0,1),
                                   scalar_constraints,false); 
  DoFTools::make_sparsity_pattern (scalar_dof_handler, 
                                   scalar_block_dsp.block(1,0),
                                   scalar_constraints,false); 
  DoFTools::make_sparsity_pattern (scalar_dof_handler, 
                                   scalar_block_dsp.block(1,1),
                                   scalar_constraints,false); 
  
  
  //scalar_block_sparsity_pattern.reinit(2,2);
  scalar_block_sparsity_pattern.copy_from (scalar_block_dsp);
  /*-------------------------*/

  /*---- build NON-BLOCK system matrix for scalar equations ------*/
  scalar_matrix.clear();
  DynamicSparsityPattern scalar_dsp(scalar_dofs);
  DoFTools::make_sparsity_pattern(scalar_dof_handler,scalar_dsp, scalar_constraints,false);
  scalar_sparsity_pattern.copy_from(scalar_dsp);
  
  bending_matrix.reinit(bending_block_sparsity_pattern);
  bending_rhs.reinit (2);
  bending_rhs.block(0).reinit (bending_dofs);
  bending_rhs.block(1).reinit (bending_dofs);
  bending_rhs.collect_sizes();
  
  scalar_block_matrix.reinit (scalar_block_sparsity_pattern);
  scalar_block_rhs.reinit(2);
  scalar_block_rhs.block(0).reinit(scalar_dofs);
  scalar_block_rhs.block(1).reinit(scalar_dofs);
  scalar_block_rhs.collect_sizes();
  
  scalar_matrix.reinit(scalar_sparsity_pattern);
  scalar_rhs.reinit(scalar_dofs);
 
  surface_area_rhs.reinit (bending_dofs);
  volume_rhs.reinit (bending_dofs);
  
  V_bending_Hn.reinit(2);
  V_bending_Hn.block(0).reinit (bending_dofs);
  V_bending_Hn.block(1).reinit (bending_dofs);
  V_bending_Hn.collect_sizes();

  Hn.reinit                    (bending_dofs);
  V_bending.reinit             (bending_dofs);
  V_volume.reinit              (bending_dofs);
  V_surface_area.reinit        (bending_dofs);
  V_constrained.reinit         (bending_dofs);
  global_euler_vector.reinit   (bending_dofs);
  gc_euler_vector.reinit       (bending_dofs);
  interpolated_X.reinit        (bending_dofs);
  gc_X.reinit                  (bending_dofs);
  
  scalar_H.reinit              (scalar_dofs);
  deviation.reinit             (scalar_dofs);
  bending_modulus.reinit       (scalar_dofs);
  spontaneous_curvature.reinit (scalar_dofs);
  ///*-------------------------*/
  
  std::cout << "in setup_dofs(), Total number of degrees of freedom: "
            << bending_dofs + scalar_dofs
            << "\n( = dofs for bending equation vector elements + dofs for dynamic parameters scalar elements )"
            << "\n( = " << bending_dofs << " + " << scalar_dofs <<" )"
            << std::endl;
  ///* --- compute some triangulation and dof statistics ---- */
  //std::vector<unsigned int> bending_sub_blocks (spacedim,0);
  //std::vector<types::global_dof_index> bending_dofs_per_block (1);
  //DoFTools::count_dofs_per_block (bending_dof_handler, bending_dofs_per_block,
  //                                bending_sub_blocks);
  //const unsigned int n_dofs_bending = bending_dofs_per_block[0],
  //                    n_dofs_scalar = scalar_dof_handler.n_dofs();
  //std::cout << "Number of active cells: "
  //          << triangulation.n_active_cells()
  //          << " (on "
  //          << triangulation.n_levels()
  //          << " levels)\n"
  //          << "having:\n" 
  //          << triangulation.n_active_quads() 
  //          << "\t active quads,\n"
  //          << triangulation.n_active_lines() 
  //          << "\t active lines,\n"
  //          << triangulation.n_used_vertices() 
  //          << "\t used vertices"
  //          << std::endl;
  /* ------------------------------------------------------ */
  std::cout << "completed setup_dofs() " << std::endl;
  /*}}}*/
}

  
template <int spacedim>
void VectorHelfrichFlow<spacedim>::initialize_global_euler_vector (const double &a, 
                                                                   const double &b, 
                                                                   const double &c)
{
/*{{{*/
/* interpolate the map from triangulation to desired geometry for the first time */
  VectorTools::interpolate(bending_dof_handler, 
                           Initial_map_sphere_to_ellipsoid<spacedim>(a,b,c),
                           global_euler_vector);

/*}}}*/
}

template <int spacedim>
void VectorHelfrichFlow<spacedim>::apply_initial_values ()
{
/*{{{*/
  const MappingQEulerian<dim,Vector<double>,spacedim> mapping (2, bending_dof_handler, global_euler_vector);
  /* project initial values onto scalar fe space */
  //VectorTools::project(scalar_dof_handler,
  //                     scalar_constraints,
  //                     QGauss<dim>(scalar_fe_degree+2),
  //                     Initial_bending_modulus<spacedim>(),
  //                     bending_modulus);
  VectorTools::interpolate(mapping, scalar_dof_handler,
                           BendingModulus<spacedim>(),
                           bending_modulus);

  //VectorTools::project(scalar_dof_handler,
  //                     scalar_constraints,
  //                     QGauss<dim>(scalar_fe_degree+2),
  //                     Initial_spontaneous_curvature<spacedim>(),
  //                     spontaneous_curvature);
  VectorTools::interpolate(mapping, scalar_dof_handler,
                           SpontaneousCurvature<spacedim>(c0),
                           spontaneous_curvature);
  
  //VectorTools::interpolate(mapping, bending_dof_handler,
  //                         Identity<spacedim>(),
  //                         interpolated_X);
/*}}}*/
}

template <int spacedim>
void VectorHelfrichFlow<spacedim>::assemble_bending_system (const double &time_step, const bool initial_assembly)
{
/*{{{*/
  
  /*
   *
   *    Spontaneous curvature model, see Eqn 5.14 in Dogan Nochetto ESIAM 2012 paper
   *
   *
   *    [   M      L - hL + 0.5*d - c0_div - c0_grad ][ V_n+1 ]   [  rhs_c0_L - rhs_c0_hL - 1.5*rhs_c0_div - rhs_c0_grad ]
   *    |                                            ||       | = |                                                      |
   *    [ -time_step*L                     M         ][ H_n+1 ]   [                      rhs_H                           ]
   *
   *    bending_matrix*Vb_H = bending_rhs
   *    
   *    M*Vs = a_rhs
   *    M*Vv = v_rhs
   *    
   *    system_matrix has size 2*n_dofs x 2*n_dofs, 
   *    each block has size n_dofs x n_dofs, and
   *    each of the rhs_X vectors have size n_dofs
   *
   *    also build the n_dofs x n_dofs matrix W_nu which holds the shape
   *    derivative of the volume, so  delta V(Gamma;W), see Discrete Helfrich
   *    Model I in GPDE Workshop Notes 
   */
  
  /* build a MappingQEulerian that initially approximates
   * sphere --> ellipsoid. This will be updated by the
   * global_euler_vector induced by the velocity field computed at each time step, see
   * update_mapping() 
   */                               

  const MappingQEulerian<dim,Vector<double>,spacedim> mapping (2, bending_dof_handler, global_euler_vector);
   
  bending_matrix   = 0;
  bending_rhs      = 0;
  
  if (!initial_assembly)
  {
    surface_area_rhs = 0;
    volume_rhs       = 0;
  }

  const QGauss<dim> quadrature_formula (2*bending_fe.degree);
  FEValues<dim,spacedim> bending_fe_values (mapping, bending_fe, quadrature_formula,
                                            update_quadrature_points |
                                            update_values            |
                                            update_normal_vectors    |
                                            update_gradients         |
                                            update_JxW_values);
  
  FEValues<dim,spacedim> scalar_fe_values (mapping, scalar_fe, quadrature_formula,
                                                       update_quadrature_points |
                                                       update_values            | 
                                                       update_gradients         |
                                                       update_JxW_values);

  const unsigned int  bending_dofs_per_cell = bending_fe.dofs_per_cell;
  const unsigned int  n_q_points = quadrature_formula.size();

  double local_M       = 0; //    (bending_dofs_per_cell, bending_dofs_per_cell);
  double local_L       = 0; //    (bending_dofs_per_cell, bending_dofs_per_cell);
  double local_hL      = 0; //    (bending_dofs_per_cell, bending_dofs_per_cell);
  double local_d       = 0; //    (bending_dofs_per_cell, bending_dofs_per_cell);
  double local_c0_div  = 0; //    (bending_dofs_per_cell, bending_dofs_per_cell);
  double local_c0_grad = 0; //    (bending_dofs_per_cell, bending_dofs_per_cell);
 
  FullMatrix<double>  local_matrix_block_00 (bending_dofs_per_cell, bending_dofs_per_cell);
  FullMatrix<double>  local_matrix_block_01 (bending_dofs_per_cell, bending_dofs_per_cell);
  FullMatrix<double>  local_matrix_block_10 (bending_dofs_per_cell, bending_dofs_per_cell);
  FullMatrix<double>  local_matrix_block_11 (bending_dofs_per_cell, bending_dofs_per_cell);
  
  double local_rhs_c0_L    = 0; //(bending_dofs_per_cell);
  double local_rhs_c0_hL   = 0; //(bending_dofs_per_cell);
  double local_rhs_c0_div  = 0; //(bending_dofs_per_cell);
  double local_rhs_c0_grad = 0; //(bending_dofs_per_cell);
  double local_rhs_H       = 0; //(bending_dofs_per_cell);
  
  Vector<double>      local_rhs_block_0 (bending_dofs_per_cell);
  Vector<double>      local_rhs_block_1 (bending_dofs_per_cell);
  
  Vector<double>      local_s_rhs       (bending_dofs_per_cell);
  Vector<double>      local_v_rhs       (bending_dofs_per_cell);

  Tensor<2,spacedim> temp_rank2_tensor;
  Point<spacedim> space_point;
  
  Identity<spacedim> identity_on_manifold;
  
  const FEValuesExtractors::Vector W (0);
  std::vector<types::global_dof_index> local_dof_indices (bending_dofs_per_cell);
 
  std::vector<Tensor<1,spacedim> > local_Hn_values(n_q_points, Tensor<1,spacedim>());
  
  std::vector<double> local_bend_mod_values(n_q_points, 0);
  std::vector<Tensor<1,spacedim> > local_bend_mod_gradients(n_q_points,Tensor<1,spacedim>());
  
  std::vector<double> local_spont_curv_values(n_q_points, 0);
  std::vector<Tensor<1,spacedim> > local_spont_curv_gradients(n_q_points,Tensor<1,spacedim>());
  
  typename DoFHandler<dim,spacedim>::active_cell_iterator 
    scalar_cell = scalar_dof_handler.begin_active();
  typename DoFHandler<dim,spacedim>::active_cell_iterator
    cell = bending_dof_handler.begin_active(),
    endc = bending_dof_handler.end();
  
  for ( ; cell!=endc; ++cell, ++scalar_cell)
  { 

    local_matrix_block_00 = 0; 
    local_matrix_block_01 = 0;
    local_matrix_block_10 = 0;
    local_matrix_block_11 = 0;

    local_rhs_block_0 = 0;
    local_rhs_block_1 = 0;  
    
    local_s_rhs       = 0;
    local_v_rhs       = 0;

    bending_fe_values.reinit (cell);
    scalar_fe_values.reinit (scalar_cell);
    
    if (!initial_assembly)
      bending_fe_values[W].get_function_values(Hn,local_Hn_values);

    scalar_fe_values.get_function_values(bending_modulus,local_bend_mod_values);
    scalar_fe_values.get_function_gradients(bending_modulus,
                                                        local_bend_mod_gradients);
    
    scalar_fe_values.get_function_values(spontaneous_curvature,
                                                     local_spont_curv_values);
    scalar_fe_values.get_function_gradients(spontaneous_curvature,
                                                        local_spont_curv_gradients);

    for (unsigned int i=0; i<bending_dofs_per_cell; ++i)
      for (unsigned int j=0; j<bending_dofs_per_cell; ++j)
        for (unsigned int q_point=0; q_point<n_q_points; ++q_point)
        {
          space_point = bending_fe_values.quadrature_point(q_point);

          local_M       =  bending_fe_values[W].value(i,q_point)*
                           bending_fe_values[W].value(j,q_point)*
                           bending_fe_values.JxW(q_point);

          local_L       =  scalar_product(bending_fe_values[W].gradient(i,q_point),
                                          bending_fe_values[W].gradient(j,q_point)
                                         )* bending_fe_values.JxW(q_point);
          
          local_hL      =  scalar_product(identity_on_manifold.symmetric_grad(bending_fe_values.normal_vector(q_point))*
                                               bending_fe_values[W].gradient(i,q_point),
                                               bending_fe_values[W].gradient(j,q_point)
                                              )* bending_fe_values.JxW(q_point);
          
          local_d       =  bending_fe_values[W].divergence(i,q_point)*
                           bending_fe_values[W].divergence(j,q_point)*
                           bending_fe_values.JxW(q_point);
          
          local_c0_div  =  bending_fe_values[W].divergence(i,q_point)*
                           local_spont_curv_values[q_point]*
                           bending_fe_values.normal_vector(q_point)*
                           bending_fe_values[W].value(j,q_point)*
                           bending_fe_values.JxW(q_point); 
          
          local_c0_grad = (local_spont_curv_gradients[q_point]*
                            bending_fe_values[W].value(i,q_point))*
                            bending_fe_values.normal_vector(q_point)*
                            bending_fe_values[W].value(j,q_point)*
                            bending_fe_values.JxW(q_point); 

          local_matrix_block_00(i,j) += local_M;
          
          local_matrix_block_01(i,j) += local_L
                                      - local_hL
                                      + 0.5*local_d
                                      - local_c0_div
                                      - local_c0_grad;
         
          local_matrix_block_10(i,j) -= time_step*local_L;
        
          local_matrix_block_11(i,j) += local_M;
        }

    
    for (unsigned int i=0; i<bending_dofs_per_cell; ++i)
      for (unsigned int q_point=0; q_point<n_q_points; ++q_point)
      {                                                          
        space_point = bending_fe_values.quadrature_point(q_point);

        Tensor<1,spacedim> local_spont_curv_grad = local_spont_curv_gradients[q_point];
        Tensor<1,spacedim> spont_curv_shape_grad = 
          local_spont_curv_grad - (local_spont_curv_grad*bending_fe_values.normal_vector(q_point))*
                            bending_fe_values.normal_vector(q_point);

        temp_rank2_tensor = outer_product(bending_fe_values.normal_vector(q_point), 
                                          spont_curv_shape_grad);
        

        local_rhs_c0_L    = scalar_product(bending_fe_values[W].gradient(i,q_point), temp_rank2_tensor)*
                            bending_fe_values.JxW(q_point);          
        
        local_rhs_c0_hL   = scalar_product(identity_on_manifold.symmetric_grad(bending_fe_values.normal_vector(q_point))*
                                              bending_fe_values[W].gradient(i,q_point), temp_rank2_tensor)*
                               bending_fe_values.JxW(q_point);
        
        local_rhs_c0_div  = local_spont_curv_values[q_point]*
                            local_spont_curv_values[q_point]*
                            bending_fe_values[W].divergence(i,q_point)*
                            bending_fe_values.JxW(q_point);

        local_rhs_c0_grad = local_spont_curv_values[q_point]*
                            (local_spont_curv_grad*
                             bending_fe_values[W].value(i,q_point))*
                            bending_fe_values.JxW(q_point);

        local_rhs_H       = scalar_product(bending_fe_values[W].gradient(i,q_point),
                                           0.5*identity_on_manifold.symmetric_grad(bending_fe_values.normal_vector(q_point)))*
                            bending_fe_values.JxW(q_point);
          
        if (!initial_assembly)
        {
          local_s_rhs(i) += bending_fe_values[W].value(i,q_point)*
                            local_Hn_values[q_point]* 
                            bending_fe_values.JxW(q_point); 

          local_v_rhs(i) += bending_fe_values[W].value(i,q_point)*
                            bending_fe_values.normal_vector(q_point)*
                            bending_fe_values.JxW(q_point); 
        }

        local_rhs_block_0 (i) += local_rhs_c0_L          
                               - local_rhs_c0_hL   
                           - 1.5*local_rhs_c0_div 
                               - local_rhs_c0_grad;
        local_rhs_block_1 (i) += local_rhs_H;
      }

    cell->get_dof_indices (local_dof_indices);
    
    bending_constraints.distribute_local_to_global (local_matrix_block_00,
                                                    local_dof_indices,
                                                    bending_matrix.block(0,0));
    bending_constraints.distribute_local_to_global (local_matrix_block_01,
                                                    local_dof_indices,
                                                    bending_matrix.block(0,1));
    bending_constraints.distribute_local_to_global (local_matrix_block_10,
                                                    local_dof_indices,
                                                    bending_matrix.block(1,0));
    bending_constraints.distribute_local_to_global (local_matrix_block_11,
                                                    local_dof_indices,
                                                    bending_matrix.block(1,1));
    
    bending_constraints.distribute_local_to_global (local_rhs_block_0,
                                                    local_dof_indices,
                                                    bending_rhs.block(0));
    bending_constraints.distribute_local_to_global (local_rhs_block_1,
                                                    local_dof_indices,
                                                    bending_rhs.block(1));

    if (!initial_assembly)
    {
      bending_constraints.distribute_local_to_global (local_s_rhs,
                                                      local_dof_indices,
                                                      surface_area_rhs);
      bending_constraints.distribute_local_to_global (local_v_rhs,
                                                      local_dof_indices,
                                                      volume_rhs);
    }
  }
  /*}}}*/
}


template <int spacedim>
void VectorHelfrichFlow<spacedim>::compute_initial_Hn(const double &time_step)
{
  /*{{{*/
  /* compute initial curvature        Hn_initial and H_initial */  
  assemble_bending_system(time_step,true);
  SparseDirectUMFPACK M;
  M.initialize(bending_matrix.block(1,1));   // get mass matrix 
  M.vmult(Hn, bending_rhs.block(1));         
  bending_constraints.distribute(Hn);
  /*}}}*/
}

template <int spacedim>
void VectorHelfrichFlow<spacedim>::output_results (const double &current_time, const int &write_step) 
{
  /*{{{*/
  (void)current_time; // turn off warnings since this is not in use

  VectorValuedSolutionSquared<spacedim> computed_mean_curvature_squared("H2");
  VectorValuedSolutionNormed<spacedim>  computed_velocity_normed("norm_of_velocity");
  VectorValuedSolutionNormed<spacedim>  computed_bending_velocity_normed("norm_of_bending_velocity");
  VectorValuedSolutionNormed<spacedim>  computed_surface_area_velocity_normed("norm_of_surface_area_velocity");
  VectorValuedSolutionNormed<spacedim>  computed_volume_velocity_normed("norm_of_volume_velocity");
  DataOut<dim,DoFHandler<dim,spacedim> > data_out;
  
  data_out.add_data_vector (scalar_dof_handler, spontaneous_curvature, "spontaneous_curvature");
  data_out.add_data_vector (scalar_dof_handler, bending_modulus, "bending_modulus");
  data_out.add_data_vector (scalar_dof_handler, scalar_H, "scalar_H");
  data_out.add_data_vector (scalar_dof_handler, deviation, "scalar_H_minus_c0");
  
  //data_out.add_data_vector (scalar_dof_handler, estimated_error_per_cell, "error_estimate");
   
  std::vector<DataComponentInterpretation::DataComponentInterpretation> 
      interp_X_ci(spacedim, DataComponentInterpretation::component_is_part_of_vector);
  
  std::vector<DataComponentInterpretation::DataComponentInterpretation> 
      gc_X_ci(spacedim, DataComponentInterpretation::component_is_part_of_vector);
  
  std::vector<DataComponentInterpretation::DataComponentInterpretation> 
      global_euler_vector_ci(spacedim, DataComponentInterpretation::component_is_part_of_vector);
  
  std::vector<DataComponentInterpretation::DataComponentInterpretation> 
      gc_euler_vector_ci(spacedim, DataComponentInterpretation::component_is_part_of_vector);
  
  std::vector<DataComponentInterpretation::DataComponentInterpretation> 
      helfrich_vector_ci(spacedim, DataComponentInterpretation::component_is_part_of_vector);
  
  std::vector<DataComponentInterpretation::DataComponentInterpretation> 
      surface_area_vector_ci(spacedim, DataComponentInterpretation::component_is_part_of_vector);
  
  std::vector<DataComponentInterpretation::DataComponentInterpretation> 
      volume_vector_ci(spacedim, DataComponentInterpretation::component_is_part_of_vector);
  
  std::vector<std::string> interp_X_names            (spacedim,"interpolated_identity");
  std::vector<std::string> gc_X_names                (spacedim,"gc_X");
  std::vector<std::string> global_euler_vector_names (spacedim,"global_euler_vector");
  std::vector<std::string> gc_euler_vector_names     (spacedim,"gc_euler_vector");
  std::vector<std::string> helfrich_names            (spacedim,"helfrich_vector");
  std::vector<std::string> surface_area_names        (spacedim,"surface_area_vector");
  std::vector<std::string> volume_names              (spacedim,"volume_vector");
  
  data_out.add_data_vector (bending_dof_handler, interpolated_X, interp_X_names,
                            interp_X_ci);
 
  data_out.add_data_vector (bending_dof_handler, gc_X, gc_X_names,
                            gc_X_ci);
  
  data_out.add_data_vector (bending_dof_handler, global_euler_vector, global_euler_vector_names,
                            global_euler_vector_ci);
  
  data_out.add_data_vector (bending_dof_handler, gc_euler_vector, gc_euler_vector_names,
                            gc_euler_vector_ci);

  data_out.add_data_vector (bending_dof_handler, V_bending, helfrich_names,
                            helfrich_vector_ci);

  data_out.add_data_vector (bending_dof_handler, V_surface_area, surface_area_names,
                            surface_area_vector_ci);

  data_out.add_data_vector (bending_dof_handler, V_volume, volume_names,
                            volume_vector_ci);


  data_out.add_data_vector (bending_dof_handler, Hn, computed_mean_curvature_squared);
  data_out.add_data_vector (bending_dof_handler, V_constrained, computed_velocity_normed);
  
  data_out.add_data_vector (bending_dof_handler, V_bending,      computed_bending_velocity_normed);
  data_out.add_data_vector (bending_dof_handler, V_surface_area, computed_surface_area_velocity_normed);
  data_out.add_data_vector (bending_dof_handler, V_volume,       computed_volume_velocity_normed);

  //const MappingQEulerian<dim,Vector<double>,spacedim> mapping(2, bending_dof_handler, global_euler_vector);
  //data_out.build_patches (mapping,2);
  data_out.build_patches ();

  char filename[80];
  sprintf(filename,"./data/refine_and_smooth_c0_%0.2f_write_step_%07d.vtk", c0, write_step);
  std::ofstream output (filename);
  data_out.write_vtk (output);
  std::cout << "data written to " << filename << std::endl;
  /*}}}*/
}

template <int spacedim>
void VectorHelfrichFlow<spacedim>::compute_scalar_H() 
{
/*{{{*/
  /*
   *
   *    scalar H from vector Hn
   *
   *    use scalar fevalues from scalar_fe to build a mass matrix
   *    M, and pull Hn values from bending_fe, then solve
   *
   *    (phi,H) = (phi, Hn*n)
   *    
   *    M*scalar_H = Mn*Hn  
   *
   */
  
  const MappingQEulerian<dim,Vector<double>,spacedim> mapping (2, bending_dof_handler, global_euler_vector);
   
  scalar_matrix = 0;
  scalar_rhs    = 0;

  const QGauss<dim> quadrature_formula (2*bending_fe.degree);
  FEValues<dim,spacedim> bending_fe_values (mapping, bending_fe, quadrature_formula,
                                            update_quadrature_points |
                                            update_values            |
                                            update_normal_vectors    |
                                            update_gradients         |
                                            update_JxW_values);
  
  FEValues<dim,spacedim> scalar_fe_values (mapping, scalar_fe, quadrature_formula,
                                           update_quadrature_points |
                                           update_values            | 
                                           update_normal_vectors    |
                                           update_gradients         |
                                           update_JxW_values);

  //const unsigned int  bending_dofs_per_cell = bending_fe.dofs_per_cell;
  const unsigned int  scalar_dofs_per_cell  = scalar_fe.dofs_per_cell;
  const unsigned int  n_q_points = quadrature_formula.size();

  FullMatrix<double>  local_scalar_matrix     (scalar_dofs_per_cell, scalar_dofs_per_cell);
  Vector<double>      local_scalar_rhs (scalar_dofs_per_cell);
  
  const FEValuesExtractors::Vector W (0);
  std::vector<types::global_dof_index> local_dof_indices (scalar_dofs_per_cell);
  std::vector<Tensor<1,spacedim> > local_Hn_values(n_q_points, Tensor<1,spacedim>());
  
  typename DoFHandler<dim,spacedim>::active_cell_iterator 
    bending_cell = bending_dof_handler.begin_active();
  typename DoFHandler<dim,spacedim>::active_cell_iterator
    scalar_cell = scalar_dof_handler.begin_active(),
    endc = scalar_dof_handler.end();
  
  for ( ; scalar_cell!=endc; ++scalar_cell, ++bending_cell)
  { 
  
    local_scalar_matrix = 0;
    local_scalar_rhs    = 0;

    bending_fe_values.reinit (bending_cell);
    scalar_fe_values.reinit (scalar_cell);
    

    bending_fe_values[W].get_function_values(Hn,local_Hn_values);

    for (unsigned int i=0; i<scalar_dofs_per_cell; ++i)
      for (unsigned int j=0; j<scalar_dofs_per_cell; ++j)
        for (unsigned int q_point=0; q_point<n_q_points; ++q_point)
        {

          local_scalar_matrix(i,j) += scalar_fe_values.shape_value(i,q_point)*
                                      scalar_fe_values.shape_value(j,q_point)*
                                      scalar_fe_values.JxW(q_point);
        }

    for (unsigned int i=0; i<scalar_dofs_per_cell; ++i)
      for (unsigned int q_point=0; q_point<n_q_points; ++q_point)
      {                          
        local_scalar_rhs(i) += scalar_fe_values.shape_value(i,q_point)*
                               (scalar_fe_values.normal_vector(q_point)*local_Hn_values[q_point])*
                               scalar_fe_values.JxW(q_point);
      }

    scalar_cell->get_dof_indices (local_dof_indices);

    scalar_constraints.distribute_local_to_global(local_scalar_matrix,
                                                  local_scalar_rhs,
                                                  local_dof_indices,
                                                  scalar_matrix,
                                                  scalar_rhs);
  }

  SparseDirectUMFPACK scalar_matrix_direct;
  scalar_matrix_direct.initialize(scalar_matrix);
  scalar_matrix_direct.vmult(scalar_H,scalar_rhs);
  
  deviation  = scalar_H;
  deviation -= spontaneous_curvature;
  
  scalar_constraints.distribute(scalar_H);
  scalar_constraints.distribute(deviation);
/*}}}*/
}


template <int spacedim>
void VectorHelfrichFlow<spacedim>::solve_using_umfpack(const double &time_step,  
                                                       const bool apply_surface_area_constraints,
                                                       const bool apply_volume_constraints) 
{
/*{{{*/
  // equation: system_matrix*Vb_H = system_rhs
  std::cout << "about to solve using SparseDirectUMFPACK" << std::endl;
  
  /* solve bending equation */
  V_bending_Hn     = 0;
  SparseDirectUMFPACK A_direct;
  A_direct.initialize(bending_matrix);
  A_direct.vmult(V_bending_Hn, bending_rhs);
  
  V_bending      = V_bending_Hn.block(0);
  Hn             = V_bending_Hn.block(1);
  
  bending_constraints.distribute(V_bending);
  bending_constraints.distribute(Hn);


  /* solve surface area and volume constraint velocities */
  V_surface_area = 0;
  V_volume       = 0;
  
  SparseDirectUMFPACK M_direct;
  M_direct.initialize(bending_matrix.block(0,0));
  M_direct.vmult(V_surface_area,surface_area_rhs);
  M_direct.vmult(V_volume,volume_rhs);
  
  bending_constraints.distribute(V_surface_area);
  bending_constraints.distribute(V_volume);
  
  /* obtain initial guess for Lagrange multipliers lambda and rho */
  double det_D, alpha_s, alpha_v, alpha_b, beta_s, beta_v, beta_b;

  alpha_b = integrate_div_func_on_surface (global_euler_vector, V_bending); 
  alpha_s = integrate_div_func_on_surface (global_euler_vector, V_surface_area); 
  alpha_v = integrate_div_func_on_surface (global_euler_vector, V_volume); 
  
  beta_b  = integrate_normal_dot_func_on_surface (global_euler_vector, V_bending);
  beta_s  = integrate_normal_dot_func_on_surface (global_euler_vector, V_surface_area); 
  beta_v  = integrate_normal_dot_func_on_surface (global_euler_vector, V_volume);
  // TODO: put the above six calls in the same loop
 
  Tensor<1,2> lambda_rho_0, rhs;
  Tensor<2,2> D_inv;
  D_inv[0][0]  =  beta_v; D_inv[0][1] = -alpha_v;
  D_inv[1][0]  = -beta_s; D_inv[1][1] =  alpha_s;
  det_D = alpha_s*beta_v - alpha_v*beta_s;
  D_inv *= 1.0/det_D;
  rhs[0] = -alpha_b; 
  rhs[1] = -beta_b; 
  lambda_rho_0 = D_inv*rhs;
    
  /* Perform Newton Iteration to obtain lambda and rho */
  std::vector<Vector<double> > displacements(3);
  std::vector<double> displacement_weights(3);
  displacements[0] = V_bending;      displacement_weights[0] = 1.0; 
  displacements[1] = V_surface_area; //displacement_weights[1] = lambda_rho[0];       
  displacements[2] = V_volume;       //displacement_weights[2] = lambda_rho[1];       
  Vector<double> local_euler_vector(bending_dof_handler.n_dofs());
  
  Tensor<1,2> lambda_rho_old, lambda_rho_new, current_eff; // = lambda_rho_0;
  lambda_rho_old = lambda_rho_0;
  double norm_eff = 1e9; double tol = 1e-9; 
  int newton_iterations = 0; const int max_newton_iterations = 10;
  std::cout << "*\n====== Newton for constraints ======" << std::endl;
  printf("(lambda_old,rho_old):  (%0.12f, %0.12f)\n",   lambda_rho_old[0],lambda_rho_old[1]);
  while (norm_eff > tol && newton_iterations < max_newton_iterations)
  {
    displacement_weights[1] = lambda_rho_old[0]; 
    displacement_weights[2] = lambda_rho_old[1]; // deformations and deformation_weights[0] remain constant in this loop
    compute_weighted_euler_vector(local_euler_vector, 
                                  time_step, 
                                  displacements, 
                                  displacement_weights);
    
    current_eff = eff (local_euler_vector);
    norm_eff    = current_eff.norm();
    
    //printf("f:        (%0.12f, %0.12f)\n", current_eff[0], current_eff[1]);
    printf("f_norm:    %0.12f\n", norm_eff);
    
    alpha_s = integrate_div_func_on_surface (local_euler_vector, V_surface_area); 
    alpha_v = integrate_div_func_on_surface (local_euler_vector, V_volume); 
    
    beta_s  = integrate_normal_dot_func_on_surface (local_euler_vector, V_surface_area); 
    beta_v  = integrate_normal_dot_func_on_surface (local_euler_vector, V_volume);
    
    //Tensor<2,2> D;
    //D[0][0] = alpha_s; D[0][1] = alpha_v;
    //D[1][0] = beta_s;  D[1][1] = beta_v;
    D_inv[0][0]  =  beta_v; D_inv[0][1] = -alpha_v;
    D_inv[1][0]  = -beta_s; D_inv[1][1] =  alpha_s;
    det_D = alpha_s*beta_v - alpha_v*beta_s;
    D_inv *= 1.0/(time_step*det_D);
    //std::cout << "|D|     = " << D.norm() << std::endl;
    //std::cout << "|D_inv| = " << D_inv.norm() << std::endl;
    //
    lambda_rho_new = lambda_rho_old - D_inv*current_eff; 
    //
    //
    lambda_rho_old = lambda_rho_new; 
    newton_iterations+=1;
  }
  printf("(lambda_new,rho_new):  (%0.12f, %0.12f)\n",   lambda_rho_new[0],lambda_rho_new[1]);

  printf("Newton iterations used/allowed:  %d / %d \n", newton_iterations,
                                                        max_newton_iterations);
  std::cout << "==============================\n*" << std::endl;

  /* build constrained_velocity vector V_constrained
   *
   *      V_constrained = V_bending + lambda*V_surface_area + rho*V_volume 
   *
   * */
  lambda = lambda_rho_new[0];
  rho    = lambda_rho_new[1];
  V_surface_area *= lambda;
  V_volume       *= rho;
  
  V_constrained   = V_bending;
  if (apply_surface_area_constraints)
    V_constrained  += V_surface_area;
  if (apply_volume_constraints)
    V_constrained  += V_volume;
/*}}}*/
}

template <int spacedim>
void VectorHelfrichFlow<spacedim>::refine_mesh(const unsigned int min_grid_level,
                                               const unsigned int max_grid_level) 
{
/*{{{*/
  std::cout << ".. entering refine_mesh() .." << std::endl;
  
  Vector<float> estimated_error_per_cell (triangulation.n_active_cells());

  KellyErrorEstimator<dim,spacedim>::estimate (scalar_dof_handler,
                                               QGauss<dim-1>(2),
                                               typename FunctionMap<spacedim>::type(),
                                               deviation,
                                               estimated_error_per_cell);
  //KellyErrorEstimator<dim,spacedim>::estimate (scalar_dof_handler,
  //                                             QGauss<dim-1>(2),
  //                                             typename FunctionMap<spacedim>::type(),
  //                                             spontaneous_curvature,
  //                                             estimated_error_per_cell);
  //KellyErrorEstimator<dim,spacedim>::estimate (bending_dof_handler,
  //                                             QGauss<dim-1>(2),
  //                                             typename FunctionMap<spacedim>::type(),
  //                                             Hn,
  //                                             estimated_error_per_cell);


  //GridRefinement::refine_and_coarsen_optimize (triangulation,
  //                                             estimated_error_per_cell);
  GridRefinement::refine_and_coarsen_fixed_fraction (triangulation,
                                                     estimated_error_per_cell,
                                                     0.5, 0.5);
  if (triangulation.n_levels() > max_grid_level)
    for (typename Triangulation<dim,spacedim>::active_cell_iterator
         cell = triangulation.begin_active(max_grid_level);
         cell != triangulation.end(); ++cell)
      cell->clear_refine_flag ();
  for (typename Triangulation<dim,spacedim>::active_cell_iterator
       cell = triangulation.begin_active(min_grid_level);
       cell != triangulation.end_active(min_grid_level); ++cell)
    cell->clear_coarsen_flag ();

  
  std::vector<Vector<double> > previous_bending_solutions(3);
  previous_bending_solutions[0] = global_euler_vector;
  previous_bending_solutions[1] = interpolated_X;
  previous_bending_solutions[2] = Hn;
  
  std::vector<Vector<double> > previous_scalar_solutions(4);
  previous_scalar_solutions[0] = bending_modulus;
  previous_scalar_solutions[1] = spontaneous_curvature;
  previous_scalar_solutions[2] = scalar_H;
  previous_scalar_solutions[3] = deviation;

  SolutionTransfer<dim, Vector<double>, DoFHandler<dim,spacedim> > bending_transfer(bending_dof_handler);
  SolutionTransfer<dim, Vector<double>, DoFHandler<dim,spacedim> > scalar_transfer(scalar_dof_handler);

  triangulation.prepare_coarsening_and_refinement();
  bending_transfer.prepare_for_coarsening_and_refinement(previous_bending_solutions);
  scalar_transfer.prepare_for_coarsening_and_refinement(previous_scalar_solutions);
  
  std::cout << ".. triangulation prepared for coarsening .." << std::endl;
 
  /* perform the mesh refinement/coarsening */
  triangulation.execute_coarsening_and_refinement ();
  /* rebuild dofs based on new mesh */
  setup_dofs ();
  
  /* initialize new solution vectors based on new dofs */
  std::vector<Vector<double> > new_bending_solutions(3);
  new_bending_solutions[0].reinit(global_euler_vector);
  new_bending_solutions[1].reinit(interpolated_X);
  new_bending_solutions[2].reinit(Hn);
  
  std::vector<Vector<double> > new_scalar_solutions(4);
  new_scalar_solutions[0].reinit(bending_modulus);
  new_scalar_solutions[1].reinit(spontaneous_curvature);
  new_scalar_solutions[2].reinit(scalar_H);
  new_scalar_solutions[3].reinit(deviation);

  /* interpolate old solutions over new mesh */
  bending_transfer.interpolate(previous_bending_solutions, new_bending_solutions);
  scalar_transfer.interpolate(previous_scalar_solutions, new_scalar_solutions);

  global_euler_vector   = new_bending_solutions[0];
  interpolated_X        = new_bending_solutions[1];
  Hn                    = new_bending_solutions[2];
  
  bending_modulus       = new_scalar_solutions[0];
  spontaneous_curvature = new_scalar_solutions[1];
  scalar_H              = new_scalar_solutions[2];
  deviation             = new_scalar_solutions[3];
  
  /* apply hanging node constraints to new solutions */
  bending_constraints.distribute(global_euler_vector); 
  bending_constraints.distribute(interpolated_X); 
  bending_constraints.distribute(Hn); 
  scalar_constraints.distribute(bending_modulus); 
  scalar_constraints.distribute(spontaneous_curvature); 
  scalar_constraints.distribute(scalar_H); 
  scalar_constraints.distribute(deviation);

  std::cout << ".. solutions interpolated over refined mesh .." << std::endl;
/*}}}*/
}

template <int spacedim>
void VectorHelfrichFlow<spacedim>::gc_euler_vector_update ()
{
/*{{{*/
  /*
   *
   */
  //compute_initial_Hn(0);
  const MappingQEulerian<dim,Vector<double>,spacedim> mapping (2, bending_dof_handler, global_euler_vector);
   
  gc_matrix = 0;
  gc_rhs    = 0;

  const QGauss<dim> quadrature_formula (2*bending_fe.degree);
  FEValues<dim,spacedim> fe_values (mapping, bending_fe, quadrature_formula,
                                    update_quadrature_points |
                                    update_values            |
                                    update_normal_vectors    |
                                    update_gradients         |
                                    update_JxW_values);
  
  const unsigned int  dofs_per_cell = bending_fe.dofs_per_cell;
  const unsigned int  n_q_points    = quadrature_formula.size();

  FullMatrix<double>  local_gc_matrix (dofs_per_cell, dofs_per_cell);
  Vector<double>      local_gc_rhs    (dofs_per_cell);
  
  const FEValuesExtractors::Vector W (0);
  std::vector<types::global_dof_index> local_dof_indices (dofs_per_cell);
  std::vector<Tensor<1,spacedim> > local_Hn_values(n_q_points, Tensor<1,spacedim>());
  
  typename DoFHandler<dim,spacedim>::active_cell_iterator 
    cell = bending_dof_handler.begin_active(),
    endc = bending_dof_handler.end();
  
  for ( ; cell!=endc; ++cell)
  { 
    local_gc_matrix = 0;
    local_gc_rhs    = 0;

    fe_values.reinit (cell);
    fe_values[W].get_function_values(Hn,local_Hn_values);

    for (unsigned int i=0; i<dofs_per_cell; ++i)
      for (unsigned int j=0; j<dofs_per_cell; ++j)
        for (unsigned int q_point=0; q_point<n_q_points; ++q_point)
        {

          local_gc_matrix(i,j) += scalar_product(fe_values[W].gradient(i,q_point),
                                                 fe_values[W].gradient(j,q_point)
                                                )*fe_values.JxW(q_point);
        }

    for (unsigned int i=0; i<dofs_per_cell; ++i)
      for (unsigned int q_point=0; q_point<n_q_points; ++q_point)
      {                          
        local_gc_rhs(i) += fe_values[W].value(i,q_point)*
                           local_Hn_values[q_point]*
                           fe_values.JxW(q_point);
      }

    cell->get_dof_indices (local_dof_indices);
    bending_constraints.distribute_local_to_global(local_gc_matrix,
                                                   local_gc_rhs,     
                                                   local_dof_indices,
                                                   gc_matrix,
                                                   gc_rhs);
  }

  SparseDirectUMFPACK gc_matrix_direct;
  gc_matrix_direct.initialize(gc_matrix);
  gc_matrix_direct.vmult(gc_X, gc_rhs);
  bending_constraints.distribute(gc_X);

  /* now update global_euler_vector using the geometrically consistent vector gc_X 
   *
   * gc_euler_vector = gc_X - p
   *                 = (X-p) - X + gc_X
   *                 = global_euler_vector - interpolated_X + gc_X
   * */
  VectorTools::interpolate(mapping, bending_dof_handler,
                           Identity<spacedim>(),
                           interpolated_X);
  
  std::cout << "l2 norm of global_euler_vector: " << global_euler_vector.l2_norm() << std::endl;
  std::cout << "l2 norm of Hn:                  " << Hn.l2_norm()             << std::endl;
  std::cout << "l2 norm of gc_X:                " << gc_X.l2_norm()           << std::endl;
  std::cout << "l2 norm of gc_rhs:              " << gc_rhs.l2_norm()         << std::endl;
  std::cout << "l2 norm of interpolated_X:      " << interpolated_X.l2_norm() << std::endl;
  
  // gc_euler_vector  = global_euler_vector;
  //gc_euler_vector = gc_X;
  //gc_euler_vector -= interpolated_X;
  /* DO THIS DIRECTLY (looping through cells...) gc_euler_vector = gc_X - p */ 
  
  //// debugging
  //
  Vector<double> tmp(bending_dof_handler.n_dofs());
  Vector<double> res(bending_dof_handler.n_dofs());
  tmp = interpolated_X;
  tmp -= gc_X;
  gc_matrix.residual(res, gc_X, gc_rhs); 
  //
  std::cout << "l2 norm of gc_X-interpolated_X: " << tmp.l2_norm()            << std::endl;
  std::cout << "|| L*gc_X - rhs ||:             " << res.l2_norm()            << std::endl;
  //
  //// 

/*}}}*/
}

/*}}}*/

template <int spacedim>
void VectorHelfrichFlow<spacedim>::run ()
{
  /*{{{*/
  
  /* geometric parameters */
  double a,b,c;
  a = 3;
  b = 2;
  c = 1;
  
  const int initial_global_refinement = 2;
  const int pre_refinement_steps      = 2;

  /* simulation parameters */
  const bool apply_surface_area_constraints = true;
  const bool apply_volume_constraints       = true;

  double time_step             = 1e-2;
  const double end_time        = 5*time_step;
  const int refinement_step    = 50;
  const int write_step         = 1;
  
  double initial_surface_area  = 0;
  double surface_area          = 0;
  double delta_surface_area    = 0;
  double initial_volume        = 0;
  double volume                = 0;
  double delta_volume          = 0;
  
  make_grid_and_global_refine(initial_global_refinement);
  setup_dofs();
  initialize_global_euler_vector(a,b,c);
  apply_initial_values();
  compute_initial_Hn(time_step);
  compute_scalar_H ();
  output_results (0,0);
  
  std::cout << "time to refine mesh"  << std::endl;
  int step = 0;
  while (step < pre_refinement_steps)
  {
    std::cout << "initial conditions applied, vector curvature and scalar quantities computed"  << std::endl;
    //refine_mesh (2, initial_global_refinement + pre_refinement_steps);
    std::cout << "initial adaptive refinement step = " << step << std::endl;
    std::cout << ".. gc_euler_vector updated using gcmm .." << std::endl;
    gc_euler_vector_update();
    //initialize_global_euler_vector(a,b,c);
    //apply_initial_values();
    //compute_initial_Hn(time_step);
    //compute_scalar_H ();
    ++step;
    output_results (0,step);
  }

  initial_surface_area = compute_surface_area(global_euler_vector);
  initial_volume       = compute_volume(global_euler_vector);
  printf("initial surf. area:      %0.9f\n", initial_surface_area);
  printf("initial volume:          %0.9f\n", initial_volume);
  std::cout << "--------------------------------------" << std::endl;
  
  double time         = 1*time_step;
  int iteration = 1; int write_step_number = iteration;
  while (time <= end_time)
  {
    
    if (iteration % refinement_step == 0)
    {
      std::cout << ".. refining mesh .." << std::endl;
      refine_mesh (std::max(2,initial_global_refinement-1),   
                   initial_global_refinement + pre_refinement_steps);
      gc_euler_vector_update();
      std::cout << ".. global_euler_vector updated using gcmm .." << std::endl;
    }
    
    assemble_bending_system(time_step);
    std::cout << "bending system assembled" <<  std::endl;
    solve_using_umfpack(time_step, apply_surface_area_constraints, apply_volume_constraints);
    std::cout << "system solved" <<  std::endl;
    compute_scalar_H ();
    update_global_euler_vector(time_step, V_constrained);
    std::cout << "global euler vector updated" <<  std::endl;
    
    surface_area       = compute_surface_area(global_euler_vector);
    delta_surface_area = 100*(surface_area - initial_surface_area)/initial_surface_area;
    volume             = compute_volume(global_euler_vector);
    delta_volume       = 100*(volume - initial_volume)/initial_volume;
    
    std::cout << "--------------------------------------" << std::endl;
    printf("iteration:         %8d\n", iteration);
    printf("time:              %0.8f\n", time);
    printf("surf. area:        %0.9f\n", surface_area);
    printf("delta surf. area:  %+0.3f %%\n", delta_surface_area);
    printf("volume:            %0.9f\n", volume);
    printf("delta volume:      %+0.3f %%\n", delta_volume);
    std::cout << "--------------------------------------" << std::endl;
    
    if (iteration % write_step == 0)
    {
      output_results (0,pre_refinement_steps+write_step_number);
      ++write_step_number;
    }
    std::cout << "*********************************\n\n";
    
    time += time_step;
    ++iteration;
  }
  /*}}}*/
}

} // end of namespace VectorHelfrich 


int main ()
{
  /*{{{*/
  try
  {
    using namespace dealii;
    using namespace VectorHelfrich;
    
    const unsigned int spacedim = 3;
    VectorHelfrichFlow<spacedim> helfrich_flow;
    helfrich_flow.run();
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
  /*}}}*/
}


