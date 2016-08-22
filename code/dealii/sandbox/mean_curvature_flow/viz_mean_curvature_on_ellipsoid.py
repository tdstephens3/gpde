# viz_mean_curvature_on_ellipsoid.py

import numpy as np
import matplotlib
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from matplotlib import cm
import matplotlib.colors as colors
import scipy.integrate as integrate

import manifold

pi = np.pi

    
# plotting
fig = plt.figure(figsize=(20,15))
ax_surface = fig.add_subplot(111,projection='3d')

Np = 150; eps = 0.001
phi = np.linspace(0+eps, pi-eps, Np)
theta = np.linspace(0, 2*pi, Np)
phi,theta = np.meshgrid(phi,theta)

a,b,c = 1.0,2.0,3.0

abc = (a,b,c)
ellipsoid = manifold.Ellipsoid()
X,Y,Z = ellipsoid.chart(phi,theta,vesicle_params=abc)
#H = ellipsoid.mean_curvature(phi,theta,vesicle_params=abc)

H = 2*a*b*c*( 3*(a**2 + b**2) + 2*c**2 + (a**2 + b**2 - 2*c**2)*np.cos(2*phi) -
        2*(a**2 - b**2)*np.cos(2*theta)*np.sin(phi)**2 ) \
    / ( 8*(a**2*b**2*np.cos(phi)**2 + c**2*(b**2*np.cos(theta)**2 
        + a**2*np.sin(theta)**2)*np.sin(phi)**2)**(1.5) ) 

#Nx,Ny,Nz = ellipsoid.unit_normal(phi,theta,vesicle_params=abc)

H2 = H**2
#
#H2min = np.min(H2)
#H2max = np.max(H2)
H2max = np.nanmax(H2)
H2min = np.nanmin(H2)
H2 = H2/H2max

ellipsoid_surf = ax_surface.plot_surface(X,Y,Z,
                                        rstride=1,cstride=1,color='g',
                                        vmin=0,vmax=1,
                                        facecolors=cm.jet(H2))

#ellipsoid_surf = ax_surface.plot_surface(X,Y,Z,
#                                        rstride=1,cstride=1,color='g')

#ax_surface.quiver(X, Y, Z, Nx, Ny, Nz,alpha=1.0,color='k') 
#m_p = cm.ScalarMappable(cmap=ellipsoid_surf.cmap, norm=ellipsoid_surf.norm)
#m_p.set_array(H2)
#plt.colorbar(m_p)



ax_surface.set_xlim((-a,a))
ax_surface.set_ylim((-b,b))
ax_surface.set_zlim((-c,c))
ax_surface.set_aspect('equal')

print('min H2: ', H2min)
print('max H2: ', H2max)
plt.show()




