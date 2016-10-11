# patch_to_sphere_to_torus.py   July 26, 2016

import numpy as np
import numpy.linalg 
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D


pi = np.pi
""" 
sphere parameterized by:   phi,theta in [0,2pi]x[0,pi], r: radius
    
    r*cos(phi)*sin(theta)
    r*sin(phi)*sin(theta)
    r*cos(theta) 

    

torus parameterized by:   phi,theta in [0,2pi]x[0,pi], a,b,c: axes

    a*cos(phi)*sin(theta)
    b*sin(phi)*sin(theta)
    c*cos(theta) 

"""

def sphere(r,phi,theta):
    x = r*np.outer(np.sin(theta), np.cos(phi))
    y = r*np.outer(np.sin(theta), np.sin(phi))
    z = r*np.outer(np.cos(theta), np.ones(np.size(phi)))
    return x,y,z

def torus(a,b,c,phi,theta):
    u = a*np.outer(np.sin(theta), np.cos(phi))
    v = b*np.outer(np.sin(theta), np.sin(phi))
    w = c*np.outer(np.cos(theta), np.ones(np.size(phi)))
    return u,v,w

def push_forward(torus_params,sphere_params,X):
    a,b,c = torus_params
    r,    = sphere_params
    x,y,z = X
    
    u = a*x#*np.sqrt(x**2 - r**2 + 1)
    v = b*y#*np.sqrt(y**2 - r**2 + 1)
    w = c*z#*np.sqrt(z**2 - r**2 + 1)

    return u,v,w


def pull_back(torus_params,sphere_params,U):
    a,b,c = torus_params
    r, = sphere_params
    u,v,w = U
    
    x = u/a
    y = v/b
    z = w/c
    
    return x,y,z


# plotting:
fig = plt.figure(figsize=(30,10))
ax_patch_f  = fig.add_subplot(231)
ax_sphere_f = fig.add_subplot(232,projection='3d')
ax_torus_f  = fig.add_subplot(233,projection='3d')

ax_patch_b  = fig.add_subplot(234)
ax_sphere_b = fig.add_subplot(235,projection='3d')
ax_torus_b  = fig.add_subplot(236,projection='3d')


# sphere:
r = 1

# torus:
a = 4
b = 3
c = 2


Np=50
Nt=50

phii   = np.linspace(0,2*pi,Np)
thetaa = np.linspace(0,pi,Nt)

phi_mesh,theta_mesh = np.meshgrid(phii,thetaa)

ax_sphere_f.plot_surface(*sphere(r,phii,thetaa), 
                       rstride=1,cstride=1, color='b', 
                       linewidth=0.15, alpha=0.1)

ax_torus_f.plot_surface(*torus(a,b,c,phii,thetaa), 
                       rstride=1,cstride=1, color='g', 
                       linewidth=0.15, alpha=0.1)
ax_sphere_f.set_aspect('equal')
ax_torus_f.set_aspect('equal')


ax_sphere_b.plot_surface(*sphere(r,phii,thetaa), 
                       rstride=1,cstride=1, color='b', 
                       linewidth=0.15, alpha=0.1)

ax_torus_b.plot_surface(*torus(a,b,c,phii,thetaa), 
                       rstride=1,cstride=1, color='g', 
                       linewidth=0.15, alpha=0.1)
ax_sphere_b.set_aspect('equal')
ax_torus_b.set_aspect('equal')




test_phi   = [2*pi/k for k in np.arange(0,2*pi,2*pi/5)]
test_theta = [pi/k for k in np.arange(0,pi,pi/5)]

torus_params  = [a,b,c]
sphere_params = [r] 

# map point forward, from patch to torus
for phi,theta in zip(test_phi,test_theta):
    print(phi)
    print(theta)

    ax_patch_f.plot(phi,theta,color='b',marker='x',mfc='b',ms=10)
    ax_sphere_f.scatter(*sphere(r,phi,theta),color='b',marker='x')

    # project to torus:
    X = sphere(r,phi,theta)
    x,y,z = X[0][0][0],X[1][0][0],X[2][0][0]
    X = [x,y,z]
    
    u,v,w = push_forward(torus_params,sphere_params,X)
    
    ax_torus_f.scatter(u,v,w,color='k',marker='x')


# map point backward, from torus to patch
for phi,theta in zip(test_phi,test_theta):
    print(phi)
    print(theta)

    U = torus(a,b,c,phi,theta)
    u,v,w = U[0][0][0],U[1][0][0],U[2][0][0]
    U = np.asarray([u,v,w])
    
    ax_torus_b.scatter(u,v,w,color='b',marker='x')

    # project to sphere from torus
    x,y,z = pull_back(torus_params,sphere_params,U)

    ax_sphere_b.scatter(x,y,z,color='g',marker='x')
    
    ax_patch_b.plot(phi,theta,color='g',marker='x',mfc='b',ms=10)



ax_patch_f.set_xlim([phii[0],phii[-1]])
ax_patch_f.set_ylim([thetaa[0],thetaa[-1]])
          
ax_patch_f.set_xticks([phii[0],phii[-1]])
ax_patch_f.set_xticklabels([phii[0],phii[-1]])
          
ax_patch_f.set_yticks([thetaa[0],thetaa[-1]])
ax_patch_f.set_yticklabels([thetaa[0],thetaa[-1]])

ax_patch_b.set_xlim([phii[0],phii[-1]])
ax_patch_b.set_ylim([thetaa[0],thetaa[-1]])
          
ax_patch_b.set_xticks([phii[0],phii[-1]])
ax_patch_b.set_xticklabels([phii[0],phii[-1]])
          
ax_patch_b.set_yticks([thetaa[0],thetaa[-1]])
ax_patch_b.set_yticklabels([thetaa[0],thetaa[-1]])

plt.show()
