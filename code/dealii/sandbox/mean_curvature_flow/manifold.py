# manifold.py

import numpy as np

pi = np.pi
sqrt2 = np.sqrt(2)
sqrt3 = np.sqrt(3)

class Manifold(object):
    
    def __init__(self,part=None):
        self.part = part
        
    def unit_normal(self,phi,theta,vesicle_params=None,pore_params=None):
        #{{{   
        r_phi   = self.chart_phi(phi,theta,vesicle_params=vesicle_params,pore_params=pore_params)
        r_theta = self.chart_theta(phi,theta,vesicle_params=vesicle_params,pore_params=pore_params)
        
        normal_vector = np.cross(r_phi, r_theta, axis=0)
        unit_n = normal_vector/np.linalg.norm(normal_vector)
        
        return unit_n
        #}}}

    def mean_curvature(self,phi,theta,vesicle_params=None,pore_params=None):
        #{{{

        r_phi        =        self.chart_phi(phi,theta,vesicle_params=vesicle_params,pore_params=pore_params)
        r_phiphi     =     self.chart_phiphi(phi,theta,vesicle_params=vesicle_params,pore_params=pore_params)
        r_theta      =      self.chart_theta(phi,theta,vesicle_params=vesicle_params,pore_params=pore_params)
        r_thetatheta = self.chart_thetatheta(phi,theta,vesicle_params=vesicle_params,pore_params=pore_params)
        r_phitheta   =   self.chart_phiTheta(phi,theta,vesicle_params=vesicle_params,pore_params=pore_params)
        unit_n       =      self.unit_normal(phi,theta,vesicle_params=vesicle_params,pore_params=pore_params)
        
        E = (r_phi*r_phi).sum(axis=0)
        F = (r_phi*r_theta).sum(axis=0)
        G = (r_theta*r_theta).sum(axis=0)
        L = (r_phiphi*unit_n).sum(axis=0)
        M = (r_phitheta*unit_n).sum(axis=0)
        N = (r_thetatheta*unit_n).sum(axis=0)

        try:
            H = (E*N - 2*F*M + G*L)/(2*(E*G - F**2))  
        except:
            print('EG - F^2: ', E*G - F**2)
            print('zero denom. detected!')
        
        return H
        #}}}

class Ellipsoid(Manifold):
    #{{{

    def __init__(self,part=None):
        Manifold.__init__(self,part=None)
        self.shape = "Ellipsoid"

    
    def chart(self,phi,theta,vesicle_params=None,pore_params=None):
        #{{{
        a,b,c, = vesicle_params
        
        #x = a*np.sin(phi)*np.cos(theta)
        #y = b*np.sin(phi)*np.sin(theta)
        #z = c*np.cos(phi) 
        
        x = a*np.cos(theta)*np.sin(phi)
        y = b*np.sin(theta)*np.sin(phi)
        z = c*np.cos(phi) 
    
        return x,y,z
        #}}}
    
    def chart_phi(self,phi,theta,vesicle_params=None,pore_params=None):
        #{{{
        a,b,c, = vesicle_params
         
        x_phi =  a*np.cos(phi)*np.cos(theta)
        y_phi =  b*np.cos(phi)*np.sin(theta)
        z_phi = -c*np.sin(phi)
    
        return np.array([x_phi, y_phi, z_phi])
        #}}}
    
    def chart_phiphi(self,phi,theta,vesicle_params=None,pore_params=None):
        #{{{
        a,b,c, = vesicle_params
        
        x_phiphi = -a*np.sin(phi)*np.cos(theta)
        y_phiphi = -b*np.sin(phi)*np.sin(theta)
        z_phiphi = -c*np.cos(phi)
    
        return np.array([x_phiphi, y_phiphi, z_phiphi])
        #}}}
    
    def chart_theta(self,phi,theta,vesicle_params=None,pore_params=None):
        #{{{
        a,b,c, = vesicle_params
        
        x_theta = -a*np.sin(phi)*np.sin(theta)
        y_theta =  b*np.sin(phi)*np.cos(theta)
        z_theta =  0.0*phi
    
        return np.array([x_theta, y_theta, z_theta])
        #}}}
    
    def chart_thetatheta(self,phi,theta,vesicle_params=None,pore_params=None):
        #{{{
        a,b,c, = vesicle_params
        
        x_thetatheta = -a*np.sin(phi)*np.cos(theta)
        y_thetatheta = -b*np.sin(phi)*np.sin(theta)
        z_thetatheta =  0.0*phi   
    
        return np.array([x_thetatheta, y_thetatheta, z_thetatheta])
        #}}}
    
    def chart_phiTheta(self,phi,theta,vesicle_params=None,pore_params=None):
        #{{{
        a,b,c, = vesicle_params
        
        x_phi = -a*np.cos(phi)*np.sin(theta)
        y_phi =  b*np.cos(phi)*np.cos(theta)
        z_phi =  0.0*phi
    
        return np.array([x_phi, y_phi, z_phi])
        #}}}
    
    def surface_area_element(self,phi,theta,vesicle_params=None,pore_params=None):
        #{{{
        pass
        #element = b**2 * np.sin(phi)
        #return element
        #}}}
    
    def volume(self,params):
        #{{{
        pass #return 0
        #}}}
    #}}}

class Toroidal_Pore(Manifold):
    #{{{

    def __init__(self,shape):
        Manifold.__init__(self,part='Toroidal_Pore')
        self.shape = shape
    
    def chart(self,phi,theta,vesicle_params=None,pore_params=None):
    #{{{                 
        # map from R^2 -> R^3, (phi,theta) |--> (x,y,z)
        #b, = vesicle_params
        r,R = pore_params
        x = (r+R + R*np.cos(phi)) * np.cos(theta)
        y = (r+R + R*np.cos(phi)) * np.sin(theta)
        z = R*np.sin(phi) + R                    
        
        return x, y, z 
    #}}}
    
    def chart_phi(self,phi,theta,vesicle_params=None,pore_params=None):
        #{{{
        b, = vesicle_params
        r,R, = pore_params
        x_phi = -R*np.sin(phi)*np.cos(theta)
        y_phi = -R*np.sin(phi)*np.sin(theta)
        z_phi =  R*np.cos(phi)
    
        return np.array([x_phi, y_phi, z_phi])
        #}}}
    
    def chart_phiphi(self,phi,theta,vesicle_params=None,pore_params=None):
        #{{{
        b, = vesicle_params
        r,R, = pore_params
        
        x_phiphi = -R*np.cos(phi)*np.cos(theta)
        y_phiphi = -R*np.cos(phi)*np.sin(theta)
        z_phiphi = -R*np.sin(phi)
    
        return np.array([ x_phiphi, y_phiphi, z_phiphi ])
        #}}}
    
    def chart_theta(self,phi,theta,vesicle_params=None,pore_params=None):
        #{{{
        b, = vesicle_params
        r,R, = pore_params
        
        x_theta = -(r+R)*np.sin(theta) - R*np.cos(phi)*np.sin(theta)
        y_theta =  (r+R)*np.cos(theta) + R*np.cos(phi)*np.cos(theta)
        z_theta =  0*phi
    
        return np.array([ x_theta, y_theta, z_theta ])
        #}}}
    
    def chart_thetatheta(self,phi,theta,vesicle_params=None,pore_params=None):
        #{{{
        b, = vesicle_params
        r,R, = pore_params
        
        x_thetatheta = -(r+R)*np.cos(theta) - R*np.cos(theta)*np.cos(phi)
        y_thetatheta = -(r+R)*np.sin(theta) - R*np.cos(phi)*np.sin(theta)
        z_thetatheta =  0*phi                             
    
        return np.array([ x_thetatheta, y_thetatheta, z_thetatheta ])
        #}}}
    
    def chart_phiTheta(self,phi,theta,vesicle_params=None,pore_params=None):
        #{{{
        b, = vesicle_params
        r,R, = pore_params
        
        x_phitheta =  R*np.sin(phi)*np.sin(theta)
        y_phitheta = -R*np.sin(phi)*np.cos(theta)
        z_phitheta =  0*phi
    
        return np.array([ x_phitheta, y_phitheta, z_phitheta ])
        #}}}
    
    def surface_area_element(self,phi,theta,vesicle_params=None,pore_params=None):
    #    #{{{
    #    b,   = vesicle_params
        r,R, = pore_params
        element = R*(r+R + R*np.cos(phi))
        return element
    #    #}}}
    
    def volume(self,params):
        #{{{
        pass #return 0
        #}}}
    
    #}}}

class Vesicle(Manifold):
    #{{{

    def __init__(self,shape):
        Manifold.__init__(self,part='Vesicle')
        self.shape = shape
    
    def chart(self,phi,theta,vesicle_params=None,pore_params=None):
        #{{{
        b, = vesicle_params
        if pore_params:
            r,R, = pore_params
        else:
            r = b; R = b

        alpha = np.sqrt((b+R)**2 - (r+R)**2 ) + R
        
        x = b*np.sin(phi)*np.cos(theta)
        y = b*np.sin(phi)*np.sin(theta)
        z = b*np.cos(phi) + alpha
    
        return x,y,z
        #return np.array([ [x],
        #                  [y],
        #                  [z] ])
        #}}}
    
    def chart_phi(self,phi,theta,vesicle_params=None,pore_params=None):
        #{{{
        b, = vesicle_params
        
        x_phi =  b*np.cos(phi)*np.cos(theta)
        y_phi =  b*np.cos(phi)*np.sin(theta)
        z_phi = -b*np.sin(phi)
    
        return np.array([x_phi, y_phi, z_phi])
        #}}}
    
    def chart_phiphi(self,phi,theta,vesicle_params=None,pore_params=None):
        #{{{
        b, = vesicle_params
        
        x_phiphi = -b*np.sin(phi)*np.cos(theta)
        y_phiphi = -b*np.sin(phi)*np.sin(theta)
        z_phiphi = -b*np.cos(phi)
    
        return np.array([x_phiphi, y_phiphi, z_phiphi])
        #}}}
    
    def chart_theta(self,phi,theta,vesicle_params=None,pore_params=None):
        #{{{
        b, = vesicle_params
        
        x_theta = -b*np.sin(phi)*np.sin(theta)
        y_theta =  b*np.sin(phi)*np.cos(theta)
        z_theta =  0.0*phi
    
        return np.array([x_theta, y_theta, z_theta])
        #}}}
    
    def chart_thetatheta(self,phi,theta,vesicle_params=None,pore_params=None):
        #{{{
        b, = vesicle_params
        
        x_thetatheta = -b*np.sin(phi)*np.cos(theta)
        y_thetatheta = -b*np.sin(phi)*np.sin(theta)
        z_thetatheta =  0.0*phi   
    
        return np.array([x_thetatheta, y_thetatheta, z_thetatheta])
        #}}}
    
    def chart_phiTheta(self,phi,theta,vesicle_params=None,pore_params=None):
        #{{{
        b, = vesicle_params
        
        x_phi = -b*np.cos(phi)*np.sin(theta)
        y_phi =  b*np.cos(phi)*np.cos(theta)
        z_phi =  0.0*phi
    
        return np.array([x_phi, y_phi, z_phi])
        #}}}
    
    def surface_area_element(self,phi,theta,vesicle_params=None,pore_params=None):
        #{{{
        b, = vesicle_params
        element = b**2 * np.sin(phi)
        return element
        #}}}
    
    def volume(self,params):
        #{{{
        pass #return 0
        #}}}
    #}}}

