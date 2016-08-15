// ---------------------------------------------------------------------
//
// Copyright (C) 1999 - 2016 by the deal.II authors
//
//
// This file is derived from deal.II/grid/manifold_lib.h by Tom Stephens,
// August 2016
//
// The deal.II library is free software; you can use it, redistribute
// it, and/or modify it under the terms of the GNU Lesser General
// Public License as published by the Free Software Foundation; either
// version 2.1 of the License, or (at your option) any later version.
// The full text of the license can be found in the file LICENSE at
// the top level of the deal.II distribution.
//
// ---------------------------------------------------------------------


/**
 * Manifold description for a spherical space coordinate system.
 *
 * You can use this Manifold object to describe any sphere, circle,
 * hypersphere or hyperdisc in two or three dimensions, both as a co-dimension
 * one manifold descriptor or as co-dimension zero manifold descriptor.
 *
 * The two template arguments match the meaning of the two template arguments
 * in Triangulation<dim, spacedim>, however this Manifold can be used to
 * describe both thin and thick objects, and the behavior is identical when
 * dim <= spacedim, i.e., the functionality of SphericalManifold<2,3> is
 * identical to SphericalManifold<3,3>.
 *
 * The two dimensional implementation of this class works by transforming
 * points to spherical coordinates, taking the average in that coordinate
 * system, and then transforming back the point to Cartesian coordinates. For
 * the three dimensional case, we use a simpler approach: we take the average
 * of the norm of the points, and use this value to shift the average point
 * along the radial direction. In order for this manifold to work correctly,
 * it cannot be attached to cells containing the center of the coordinate
 * system. This point is a singular point of the coordinate transformation,
 * and there taking averages does not make any sense.
 *
 * This class is used in step-1 and step-2 to describe the boundaries of
 * circles. Its use is also discussed in the results section of step-6.
 *
 * @ingroup manifold
 *
 * @author Luca Heltai, 2014
 */
using namespace dealii;

template <int dim,int spacedim>
class Ellipsoid: public SphericalManifold<dim,spacedim>
{
  /*{{{*/
public:

  Ellipsoid(double,double,double,Point<spacedim>);   

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
Ellipsoid<dim,spacedim>::Ellipsoid(double a, double b, double c, Point<spacedim> center) : SphericalManifold<dim,spacedim>(center), a(a), b(b),c(c)        
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
 
