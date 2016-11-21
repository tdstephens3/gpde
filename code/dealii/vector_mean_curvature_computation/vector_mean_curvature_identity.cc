/* ---------------------------------------------------------------------
 *
 * vector_mean_curvature_identity.cc      Nov 16, 2016
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
 *
 */

#include <deal.II/base/quadrature_lib.h>
#include <deal.II/base/function.h>
#include <deal.II/base/tensor_function.h>
#include <deal.II/grid/tria.h>
#include <deal.II/grid/manifold_lib.h>
#include <deal.II/grid/grid_generator.h>
#include <deal.II/lac/full_matrix.h>
#include <deal.II/lac/vector.h>
#include <deal.II/lac/sparse_direct.h>
#include <deal.II/lac/sparse_matrix.h>
#include <deal.II/lac/dynamic_sparsity_pattern.h>
#include <deal.II/dofs/dof_handler.h>
#include <deal.II/dofs/dof_accessor.h>
#include <deal.II/dofs/dof_tools.h>
#include <deal.II/fe/fe_system.h>
#include <deal.II/fe/fe_values.h>
#include <deal.II/fe/mapping_q_eulerian.h>
#include <deal.II/numerics/data_out.h>
#include <deal.II/numerics/vector_tools.h>
#include <deal.II/numerics/error_estimator.h>

#include <fstream>

namespace VectorMeanCurvature
{
  using namespace dealii;


/* ********************************** 
 *
 * MeanCurvatureFromPosition<spacedim> class definition
 *
 * */
template <int spacedim>
class MeanCurvatureFromPosition
{
  /*{{{*/
  public:
    MeanCurvatureFromPosition () ;
    void run ();
  
  private:
    static const unsigned int dim = spacedim-1;
    bool verbose = false;                       // print short output statements
  
    void make_grid_and_global_refine (const unsigned int global_refinements);
    void setup_dofs ();
    void initialize_geometry (const double &a,const double &b,const double &c);
    
    double compute_surface_area();
    double compute_volume();
    
    void assemble_system ();
    void solve_using_umfpack(); 
    void compute_vector_H ();
    void compute_exact_vector_H (const double &a, const double &b, const double &c);
    void compute_vector_error (const double &a, const double &b, const double &c); 
    void compute_scalar_H (); 
    void compute_exact_scalar_H (const double &a, const double &b, const double &c); 
    void compute_scalar_error (const double &a, const double &b, const double &c); 
    
    
    void output_results ();
    //void compute_error (double, double, double, Point<3>) const;

    Triangulation<dim,spacedim> triangulation;
    Vector<double>              euler_vector;          // defines geometry through MappingQEulerian
    
    /* - data structures for vector mean curvature - */
    const unsigned int          vector_fe_degree = 2;
    FESystem<dim,spacedim>      vector_fe; 
    DoFHandler<dim,spacedim>    vector_dof_handler;
    
    SparsityPattern             vector_system_sparsity_pattern;
    SparseMatrix<double>        vector_system_matrix;
    
    Vector<double>              vector_system_rhs;  
    Vector<double>              vector_H;              // vector mean curvature
    Vector<double>              exact_vector_H;        // exact solution for vector mean curvature
    
    /* - data structures for scalar parameters - */
    const unsigned int          scalar_fe_degree = 2;
    FE_Q<dim,spacedim>          scalar_fe; 
    DoFHandler<dim,spacedim>    scalar_dof_handler;
    
    SparsityPattern             scalar_system_sparsity_pattern;
    SparseMatrix<double>        scalar_system_matrix;
    
    Vector<double>              scalar_system_rhs;       
    Vector<double>              scalar_H;               // scalar mean curvature
    Vector<double>              exact_scalar_H;         // exact solution for scalar mean curvature
    
    Vector<double>              diff_scalar_H;         // difference between 
                                                        // computed scalar_H and exact_scalar_H
    Vector<double>              diff_vector_H;         // difference between 
                                                        // computed vector_H and exact_vector_H
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
class MapToEllipsoid: public Function<spacedim>
{
/*{{{*/
  public:
    MapToEllipsoid(double a, double b, double c) : Function<spacedim>(3), abc_coeffs{a,b,c}  {}
    
    virtual void vector_value_list (const std::vector<Point<spacedim> > &points,
                                    std::vector<Vector<double> >   &value_list) const;
    virtual void vector_value (const Point<spacedim> &p, Vector<double> &value) const;
    virtual double value (const Point<spacedim> &p, const unsigned int component = 0) const;
  private:
    double abc_coeffs[3];
/*}}}*/
};

template<int spacedim>
double MapToEllipsoid<spacedim>::value(const Point<spacedim> &p, const unsigned int component)  const
{
  /*{{{*/
  
  double norm_p = p.distance(Point<spacedim>(0,0,0));
  return abc_coeffs[component]*p(component)/norm_p - p(component);   

  /*}}}*/
}

template<int spacedim>
void MapToEllipsoid<spacedim>::vector_value(const Point<spacedim> &p, Vector<double> &value) const
{
  /*{{{*/
  for (unsigned int c=0; c<this->n_components; ++c) 
  {
    value(c) = MapToEllipsoid<spacedim>::value(p,c);
  }
  /*}}}*/
}

template <int spacedim>
void MapToEllipsoid<spacedim>::vector_value_list (const std::vector<Point<spacedim> > &points,
                                            std::vector<Vector<double> >   &value_list) const
{
  /*{{{*/
  Assert (value_list.size() == points.size(),
          ExcDimensionMismatch (value_list.size(), points.size()));
  const unsigned int n_points = points.size();
  for (unsigned int p=0; p<n_points; ++p)
    MapToEllipsoid<spacedim>::vector_value (points[p], value_list[p]);
  /*}}}*/
}

template <int spacedim>
class ExactScalarMeanCurvatureOnEllipsoid: public Function<spacedim>
{
/*{{{*/
  public:
    ExactScalarMeanCurvatureOnEllipsoid (double a, double b, double c) : Function<spacedim>(1), a(a), b(b), c(c), abc_coeffs{a,b,c}, spherical_manifold(Point<spacedim>(0,0,0))  {}
    
    virtual double value (const Point<spacedim> &p, const unsigned int component = 0) const;
  private:
    double a,b,c;
    double abc_coeffs[3];
    SphericalManifold<2,spacedim> spherical_manifold;
/*}}}*/
};

template<int spacedim>
double ExactScalarMeanCurvatureOnEllipsoid<spacedim>::value(const Point<spacedim> &p, const unsigned int )  const
{
  /*{{{*/
  
  Point<spacedim> unmapped_p(p(0)/a, p(1)/b,  p(2)/c);

  Point<spacedim> chart_point = spherical_manifold.pull_back(unmapped_p);
  double theta = chart_point(1);
  double phi   = chart_point(2);
  
  double exact_mean_curv = 2*a*b*c*( 3*(pow(a,2) + pow(b,2)) + 2*pow(c,2) 
                               + (pow(a,2) + pow(b,2) - 2*pow(c,2))*cos(2*theta) 
                              - 2*(pow(a,2) - pow(b,2))*cos(2*phi)*pow(sin(theta),2) ) 
                           / ( 8*pow((pow(a,2)*pow(b,2)*pow(cos(theta),2)
                                + pow(c,2)*(pow(b,2)*pow(cos(phi),2) 
                                + pow(a,2)*pow(sin(phi),2))*pow(sin(theta),2)),1.5) );
  
  return -exact_mean_curv;
  /*}}}*/
}

template <int spacedim>
class ExactVectorMeanCurvatureOnEllipsoid: public TensorFunction<1,spacedim>
{
/*{{{*/
  public:
    ExactVectorMeanCurvatureOnEllipsoid (double a, double b, double c) : TensorFunction<1,spacedim>(), a(a), b(b), c(c), exact_scalar_H(a,b,c) {}
    
    virtual Tensor<1,spacedim> value (const Point<spacedim> &p) const;
  private:
    double a,b,c;
    ExactScalarMeanCurvatureOnEllipsoid<spacedim> exact_scalar_H;
/*}}}*/
};

template<int spacedim>
Tensor<1,spacedim> ExactVectorMeanCurvatureOnEllipsoid<spacedim>::value(const Point<spacedim> &p)  const
{
  /*{{{*/
  Tensor<1,spacedim> normal,vector_H;
  normal[0] = p(0)/a;
  normal[1] = p(1)/b;
  normal[2] = p(2)/c;
  normal /= normal.norm();

  vector_H  = normal;
  vector_H *= exact_scalar_H.value(p);

  return vector_H;
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
 * MeanCurvatureFromPosition<spacedim> functions
 *
 * */
/*{{{*/

template <int spacedim>
MeanCurvatureFromPosition<spacedim>::MeanCurvatureFromPosition ()
  :
  vector_fe(FE_Q<dim,spacedim>(vector_fe_degree),spacedim),
  vector_dof_handler(triangulation),
  scalar_fe(scalar_fe_degree),
  scalar_dof_handler(triangulation)
  {}


template <int spacedim>
double MeanCurvatureFromPosition<spacedim>::compute_surface_area ()
{
  /*{{{*/
  const MappingQEulerian<dim,Vector<double>,spacedim> mapping(2, vector_dof_handler, euler_vector);
  
  const QGauss<dim> quadrature_formula (2*vector_fe.degree);
  FEValues<dim,spacedim> fe_values (mapping, vector_fe, quadrature_formula,
                                    update_values              |
                                    update_normal_vectors      |
                                    update_gradients           |
                                    update_quadrature_points   |
                                    update_JxW_values);

  const unsigned int  n_q_points    = quadrature_formula.size();

  double surface_area = 0;
  for (typename DoFHandler<dim,spacedim>::active_cell_iterator
       cell = vector_dof_handler.begin_active(),
       endc = vector_dof_handler.end();
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
double MeanCurvatureFromPosition<spacedim>::compute_volume ()
{
  /*{{{*/
  const MappingQEulerian<dim,Vector<double>,spacedim> mapping(2, vector_dof_handler, euler_vector);
  
  const QGauss<dim> quadrature_formula (2*vector_fe.degree);
  FEValues<dim,spacedim> fe_values (mapping, vector_fe, quadrature_formula,
                                    update_values              |
                                    update_normal_vectors      |
                                    update_gradients           |
                                    update_quadrature_points   |
                                    update_JxW_values);

  const unsigned int  n_q_points    = quadrature_formula.size();

  double volume = 0;
  for (typename DoFHandler<dim,spacedim>::active_cell_iterator
       cell = vector_dof_handler.begin_active(),
       endc = vector_dof_handler.end();
       cell!=endc; ++cell)
  { 
    fe_values.reinit (cell);
    
    for (unsigned int q_point=0; q_point<n_q_points; ++q_point) /* -= to account for direction of normal vector */
      volume -= fe_values.quadrature_point(q_point)*fe_values.normal_vector(q_point)*fe_values.JxW(q_point);
  }
  return (1.0/3.0)*volume;  
  /*}}}*/
}

template <int spacedim>
void MeanCurvatureFromPosition<spacedim>::make_grid_and_global_refine (const unsigned int initial_global_refinement)
{
  /*{{{*/
  /*  --- build geometry --- */
  GridGenerator::hyper_sphere(triangulation,Point<spacedim>(0,0,0),1.0);

  triangulation.set_all_manifold_ids(0);
  
  triangulation.refine_global(initial_global_refinement);
  /*}}}*/
}

template <int spacedim>
void MeanCurvatureFromPosition<spacedim>::setup_dofs ()
{
  /*{{{*/                     
  
  vector_dof_handler.distribute_dofs (vector_fe);
  scalar_dof_handler.distribute_dofs (scalar_fe);

  const unsigned int vector_dofs = vector_dof_handler.n_dofs();
  const unsigned int scalar_dofs = scalar_dof_handler.n_dofs();
  
  /*---- build system matrix for vector equations ------*/
  DynamicSparsityPattern vector_dsp (vector_dofs,vector_dofs);
  DoFTools::make_sparsity_pattern (vector_dof_handler,vector_dsp);
  vector_system_sparsity_pattern.copy_from (vector_dsp);
  
  vector_system_matrix.reinit (vector_system_sparsity_pattern);
  vector_system_rhs.reinit (vector_dofs);
  
  /*---- build system matrix for scalar equations ------*/
  DynamicSparsityPattern scalar_dsp (scalar_dofs,scalar_dofs);
  DoFTools::make_sparsity_pattern (scalar_dof_handler,scalar_dsp);
  scalar_system_sparsity_pattern.copy_from (scalar_dsp);
  
  scalar_system_matrix.reinit (scalar_system_sparsity_pattern);
  scalar_system_rhs.reinit (scalar_dofs);
  
  /*---- initialize solution and auxillary vectors with correct number of dofs ------*/
  euler_vector.reinit   (vector_dofs);
  vector_H.reinit       (vector_dofs);
  exact_vector_H.reinit (vector_dofs);
  diff_vector_H.reinit  (vector_dofs);
  scalar_H.reinit       (scalar_dofs);
  exact_scalar_H.reinit (scalar_dofs);
  diff_scalar_H.reinit  (scalar_dofs);
  /*----------------------------------*/
  
  std::cout << "\nin setup_dofs(), Total number of degrees of freedom: "
            << vector_dofs + scalar_dofs
            << "\n( = dofs for vector equation vector elements + dofs for dynamic parameters scalar elements )"
            << "\n( = " << vector_dofs << " + " << scalar_dofs <<" )"
            << std::endl;
  /*}}}*/
}

  
template <int spacedim>
void MeanCurvatureFromPosition<spacedim>::initialize_geometry (const double &a, 
                                                               const double &b, 
                                                               const double &c)
{
/*{{{*/
/* interpolate the map from triangulation to desired geometry for the first time */
  VectorTools::interpolate(vector_dof_handler, 
                           MapToEllipsoid<spacedim>(a,b,c),
                           euler_vector);
/*}}}*/
}

template <int spacedim>
void MeanCurvatureFromPosition<spacedim>::assemble_system ()
{
/*{{{*/
  
  /*
   *    vector_H = -laplace_beltrami identity, 
   *    
   *    weakly:
   *    (phi_i,phi_j)_{i,j} * vector_H = (surface_gradient phi_i, surface_gradient identity)_{i}
   *
   */                               

  const MappingQEulerian<dim,Vector<double>,spacedim> mapping (2, vector_dof_handler, euler_vector);
   
  const QGauss<dim> quadrature_formula (2*vector_fe.degree);
  FEValues<dim,spacedim> fe_values (mapping, vector_fe, quadrature_formula,
                                    update_quadrature_points |
                                    update_values            |
                                    update_normal_vectors    |
                                    update_gradients         |
                                    update_JxW_values);
  
  const unsigned int  dofs_per_cell = vector_fe.dofs_per_cell;
  const unsigned int  n_q_points    = quadrature_formula.size();

  FullMatrix<double>  local_M (dofs_per_cell, dofs_per_cell);
  Vector<double>      local_rhs (dofs_per_cell);

  Identity<spacedim> identity_on_manifold;
  
  const FEValuesExtractors::Vector W (0);
  std::vector<types::global_dof_index> local_dof_indices (dofs_per_cell);
 
  typename DoFHandler<dim,spacedim>::active_cell_iterator 
    cell = vector_dof_handler.begin_active(),
    endc = vector_dof_handler.end();
  
  for ( ; cell!=endc; ++cell)
  { 

    local_M   = 0;
    local_rhs = 0;

    fe_values.reinit (cell);
    
    for (unsigned int i=0; i<dofs_per_cell; ++i)
      for (unsigned int j=0; j<dofs_per_cell; ++j)
        for (unsigned int q_point=0; q_point<n_q_points; ++q_point)
        {

          local_M (i,j) += fe_values[W].value(i,q_point)*
                           fe_values[W].value(j,q_point)*
                           fe_values.JxW(q_point);
        }

    
    for (unsigned int i=0; i<dofs_per_cell; ++i)
      for (unsigned int q_point=0; q_point<n_q_points; ++q_point)
      {                                                          
        local_rhs (i) += scalar_product(fe_values[W].gradient(i,q_point),
                         0.5*identity_on_manifold.symmetric_grad(fe_values.normal_vector(q_point)))
                         *fe_values.JxW(q_point);
      }
    
    cell->get_dof_indices (local_dof_indices);
    for (unsigned int i=0; i<dofs_per_cell; ++i)
    {                                                          
      vector_system_rhs (local_dof_indices[i]) += local_rhs(i);
      for (unsigned int j=0; j<dofs_per_cell; ++j)
        vector_system_matrix.add (local_dof_indices[i],
                                  local_dof_indices[j],
                                  local_M(i,j));
    }
  }
  /*}}}*/
}


template <int spacedim>
void MeanCurvatureFromPosition<spacedim>::compute_vector_H()
{
  /*{{{*/
  /* compute vector mean curvature  */  
  assemble_system();
  SparseDirectUMFPACK M;
  M.initialize(vector_system_matrix);    
  M.vmult(vector_H, vector_system_rhs);   
  
  if (verbose)
    std::cout << "vector mean curvature computed" << std::endl;
  /*}}}*/
}

template <int spacedim>
void MeanCurvatureFromPosition<spacedim>::compute_exact_vector_H(const double &a, const double &b, const double &c) 
{
  /*{{{*/
  ExactVectorMeanCurvatureOnEllipsoid<spacedim> tensor_H(a,b,c);
  const MappingQEulerian<dim,Vector<double>,spacedim> mapping(2, vector_dof_handler, euler_vector);
  VectorTools::interpolate(mapping,vector_dof_handler, 
                           VectorFunctionFromTensorFunction<spacedim>(tensor_H,0,spacedim),
                           exact_vector_H);
  
  if (verbose)
    std::cout << "exact scalar mean curvature computed" << std::endl;
  /*}}}*/
}

template <int spacedim>
void MeanCurvatureFromPosition<spacedim>::compute_vector_error(const double &a, const double &b, const double &c) 
{
  /*{{{*/
  ExactVectorMeanCurvatureOnEllipsoid<spacedim> tensor_H(a,b,c);
  const QGauss<dim> quadrature_formula (2*vector_fe.degree);
  const MappingQEulerian<dim,Vector<double>,spacedim> mapping(2, vector_dof_handler, euler_vector);
  VectorTools::integrate_difference (mapping, vector_dof_handler, 
                                     vector_H, 
                                     VectorFunctionFromTensorFunction<spacedim>(tensor_H,0,spacedim), 
                                     diff_vector_H,
                                     quadrature_formula, VectorTools::L2_norm);
  double err_vector_H = diff_vector_H.l2_norm();
  if (verbose)
    printf("error in vector_H: %0.2e\n", err_vector_H);
  /*}}}*/
}

template <int spacedim>
void MeanCurvatureFromPosition<spacedim>::compute_scalar_H() 
{
/*{{{*/
  /*
   *    scalar H from vector Hn
   *
   *    use scalar fevalues from scalar_fe to build a mass matrix
   *    M, and pull Hn values from vector_fe, then solve
   *
   *    (phi_i,phi_j)_{i,j}*scalar_H = (phi_i, vector_H*n)_{i}
   *    
   *    M*scalar_H = (phi_i, vector_H*n)_{i}  
   *
   *    solve for scalar_H, and phi is a scalar valued finite element
   *
   */
  
  const MappingQEulerian<dim,Vector<double>,spacedim> mapping (2, vector_dof_handler, euler_vector);
   
  scalar_system_matrix = 0;
  scalar_system_rhs    = 0;

  const QGauss<dim> quadrature_formula (2*vector_fe.degree);
  FEValues<dim,spacedim> vector_fe_values (mapping, vector_fe, quadrature_formula,
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

  const unsigned int  scalar_dofs_per_cell  = scalar_fe.dofs_per_cell;
  const unsigned int  n_q_points = quadrature_formula.size();

  FullMatrix<double>  local_M   (scalar_dofs_per_cell, scalar_dofs_per_cell);
  Vector<double>      local_rhs (scalar_dofs_per_cell);
  
  const FEValuesExtractors::Vector W (0);
  std::vector<types::global_dof_index> local_dof_indices (scalar_dofs_per_cell);
  std::vector<Tensor<1,spacedim> > local_vector_H_values(n_q_points, Tensor<1,spacedim>());
  
  typename DoFHandler<dim,spacedim>::active_cell_iterator 
    vector_cell = vector_dof_handler.begin_active();
  typename DoFHandler<dim,spacedim>::active_cell_iterator
    scalar_cell = scalar_dof_handler.begin_active(),
    endc = scalar_dof_handler.end();
  
  for ( ; scalar_cell!=endc; ++scalar_cell, ++vector_cell)
  { 
  
    local_M   = 0;
    local_rhs = 0;

    vector_fe_values.reinit (vector_cell);
    scalar_fe_values.reinit (scalar_cell);

    vector_fe_values[W].get_function_values(vector_H,local_vector_H_values);

    for (unsigned int i=0; i<scalar_dofs_per_cell; ++i)
      for (unsigned int j=0; j<scalar_dofs_per_cell; ++j)
        for (unsigned int q_point=0; q_point<n_q_points; ++q_point)
        {

          local_M(i,j) += scalar_fe_values.shape_value(i,q_point)*
                          scalar_fe_values.shape_value(j,q_point)*
                          scalar_fe_values.JxW(q_point);
        }

    for (unsigned int i=0; i<scalar_dofs_per_cell; ++i)
      for (unsigned int q_point=0; q_point<n_q_points; ++q_point)
      {                          
        local_rhs(i) += scalar_fe_values.shape_value(i,q_point)*
                        (scalar_fe_values.normal_vector(q_point)*local_vector_H_values[q_point])*
                        scalar_fe_values.JxW(q_point);
      }

    scalar_cell->get_dof_indices (local_dof_indices);
    for (unsigned int i=0; i<scalar_dofs_per_cell; ++i)
    {                                                          
      scalar_system_rhs (local_dof_indices[i]) += local_rhs(i);
      for (unsigned int j=0; j<scalar_dofs_per_cell; ++j)
        scalar_system_matrix.add (local_dof_indices[i],
                                  local_dof_indices[j],
                                  local_M(i,j));
    }
  }

  SparseDirectUMFPACK scalar_system_matrix_direct;
  scalar_system_matrix_direct.initialize(scalar_system_matrix);
  scalar_system_matrix_direct.vmult(scalar_H,scalar_system_rhs);
  
  if (verbose)
    std::cout << "scalar mean curvature computed" << std::endl;
/*}}}*/
}

template <int spacedim>
void MeanCurvatureFromPosition<spacedim>::compute_exact_scalar_H(const double &a, const double &b, const double &c) 
{
  /*{{{*/
  const MappingQEulerian<dim,Vector<double>,spacedim> mapping(2, vector_dof_handler, euler_vector);
  VectorTools::interpolate(mapping,scalar_dof_handler, 
                           ExactScalarMeanCurvatureOnEllipsoid<spacedim>(a,b,c),
                           exact_scalar_H);
  
  if (verbose)
    std::cout << "exact scalar mean curvature computed" << std::endl;
  /*}}}*/
}

template <int spacedim>
void MeanCurvatureFromPosition<spacedim>::compute_scalar_error(const double &a, const double &b, const double &c) 
{
  /*{{{*/
  const QGauss<dim> quadrature_formula (2*scalar_fe.degree);
  const MappingQEulerian<dim,Vector<double>,spacedim> mapping(2, vector_dof_handler, euler_vector);
  VectorTools::integrate_difference (mapping, scalar_dof_handler, 
                                     scalar_H, ExactScalarMeanCurvatureOnEllipsoid<spacedim>(a,b,c), diff_scalar_H,
                                     quadrature_formula, VectorTools::L2_norm);
  double err_scalar_H = diff_scalar_H.l2_norm();
  printf("error in scalar_H: %0.2e\n",err_scalar_H);
  /*}}}*/
}

template <int spacedim>
void MeanCurvatureFromPosition<spacedim>::output_results () 
{
  /*{{{*/
  

  VectorValuedSolutionSquared<spacedim> computed_mean_curvature_squared("H2");
  VectorValuedSolutionSquared<spacedim> computed_exact_vector_H_squared("exact_H2");
  
  DataOut<dim,DoFHandler<dim,spacedim> > data_out;
  
  data_out.add_data_vector (scalar_dof_handler, scalar_H, "scalar_H");
  data_out.add_data_vector (scalar_dof_handler, exact_scalar_H, "exact_scalar_H");
  
  diff_scalar_H = exact_scalar_H;
  diff_scalar_H -= scalar_H;
  data_out.add_data_vector (scalar_dof_handler, diff_scalar_H, "diff_scalar_H");
  
  std::vector<DataComponentInterpretation::DataComponentInterpretation> 
      euler_vector_ci(spacedim, DataComponentInterpretation::component_is_part_of_vector);
  
  std::vector<std::string> euler_vector_names (spacedim,"euler_vector");
  
  data_out.add_data_vector (vector_dof_handler, euler_vector, 
                            euler_vector_names,
                            euler_vector_ci);
  
  data_out.add_data_vector (vector_dof_handler, vector_H, computed_mean_curvature_squared);
  data_out.add_data_vector (vector_dof_handler, exact_vector_H, computed_exact_vector_H_squared);
  
  /* use the mapping if you don't want to deform your solution by euler_vector
   * in order to visualize the result */
  
  //const MappingQEulerian<dim,Vector<double>,spacedim> mapping(2, vector_dof_handler, euler_vector);
  //data_out.build_patches (mapping,2);
  data_out.build_patches ();

  char filename[80];
  sprintf(filename,"./data/vector_mean_curvature.vtk");
  std::ofstream output (filename);
  data_out.write_vtk (output);
  std::cout << "data written to " << filename << std::endl;
  /*}}}*/
}

/*}}}*/

template <int spacedim>
void MeanCurvatureFromPosition<spacedim>::run ()
{
  /*{{{*/
  verbose = true;
  
  /* geometric parameters */
  double a,b,c;
  a = 1;
  b = 2;
  c = 3;
  
  const int global_refinements = 3;
  
  make_grid_and_global_refine (global_refinements);
  setup_dofs();
  initialize_geometry (a,b,c);
  assemble_system();
  
  double computed_surface_area = compute_surface_area();
  double computed_volume       = compute_volume();
  
  std::cout << "--------------------------------------" << std::endl;
  printf("computed surface area: %0.9f\n", computed_surface_area);
  printf("computed volume:       %0.9f\n", computed_volume);
  std::cout << "--------------------------------------" << std::endl;
  
  compute_vector_H ();
  compute_exact_vector_H(a,b,c);
  compute_vector_error (a,b,c);
  
  compute_scalar_H ();
  compute_exact_scalar_H (a,b,c);
  compute_scalar_error (a,b,c);
  
  output_results ();
  /*}}}*/
}

} // end of namespace VectorMeanCurvature 


int main ()
{
  /*{{{*/
  try
  {
    using namespace dealii;
    using namespace VectorMeanCurvature;
    
    const unsigned int spacedim = 3;
    MeanCurvatureFromPosition<spacedim> vector_mean_curvature_on_surface;
    vector_mean_curvature_on_surface.run();
    
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


