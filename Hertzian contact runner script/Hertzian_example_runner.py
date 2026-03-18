"""
Hertzian Contact Analysis - Self-Sufficient Example Runner
==========================================================

This script applies the volume and surface method in microparticle traction force microscopy on an exemplary Hertzian contact scenario.
All dependencies are explicitly imported from standard libraries and ShElastic.
Run in the conda environment from the provided .yml file to ensure all required packages are available.

For the volume method, the provided u_profile.mat and mesh_profile.mat files can be read in directly.
If these files are not detected, the script creates synthetic image data (deformed and undeformed nanoparticle distributions).
These can be analyzed in Matlab to recover the u_profile.mat and mesh_profile.mat files for analysis.
In that case, the volume method analysis has to be run again with the files available.
"""

# ============================================================================
# STANDARD LIBRARY IMPORTS
# ============================================================================
import os
import sys
import time
from datetime import date
# ============================================================================
# THIRD-PARTY SCIENTIFIC LIBRARIES
# ============================================================================
import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl
from matplotlib import cm, ticker as tck, patches
from matplotlib.ticker import AutoMinorLocator
from matplotlib.patches import FancyArrowPatch
import pandas as pd
import tifffile as tif
import pyshtools as _psh
import scipy as sp
import scipy.sparse as spm
from scipy.optimize import minimize
from scipy.interpolate import RegularGridInterpolator
from scipy.signal import fftconvolve
import scipy.io as sio
from tqdm import tqdm
import traceback

# ============================================================================
# CONFIGURATION: PATHS AND DIRECTORIES
# ============================================================================
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))

# ShElastic path - try parent directory first, then same directory
shelastic_path = os.path.join(os.path.dirname(SCRIPT_DIR), 'ShElastic')
if not os.path.exists(shelastic_path):
    shelastic_path = os.path.join(SCRIPT_DIR, 'ShElastic')

if not os.path.exists(shelastic_path):
    raise FileNotFoundError(f"ShElastic not found at:\n  {os.path.join(os.path.dirname(SCRIPT_DIR), 'ShElastic')} or\n  {os.path.join(SCRIPT_DIR, 'ShElastic')}")

if shelastic_path not in sys.path:
    sys.path.insert(0, shelastic_path)

# ============================================================================
# SHELASTIC LIBRARY COMPONENTS
# ============================================================================
from shelastic.shutil import (
    SphCoord_to_CartCoord, CartCoord_to_SphCoord, 
    SHmesh2Vec, GLQCartCoord, LM_list, SHVec2mesh
)
from shelastic.sphere_utils import (
    loadCoeffs, Ur_interp, usurf2vec, Tvec2Tres, 
    usurf2dr2, grad_usurf2dr2, genLmat, genSmat
)

# Set output and coefficient directories
OUTPUT_DIR = os.path.join(shelastic_path, 'examples', 'Data_Hertzian_contact')
COEFF_DIR = os.path.join(shelastic_path, 'shelastic', 'default_modes')
os.makedirs(OUTPUT_DIR, exist_ok=True)

# Centralized default save paths
DEFAULT_SAVEPATH = OUTPUT_DIR
DEFAULT_RELATIVE_SAVEPATH = 'Data'

# ============================================================================
# UTILITY FUNCTIONS
# ============================================================================

def lame_mu(E, nu):
    """Shear modulus from Young's modulus and Poisson's ratio."""
    return E / (2.0 * (1.0 + nu))

def full5tap3D(direction='x'):
    """5-point stencil for 3D spatial derivatives (separable: prefilter + derivative)."""
    prefilter = -np.array([0.037659, 0.249153, 0.426375, 0.249153, 0.037659])
    prefilter1 = prefilter.reshape((5, 1))
    prefilter2 = prefilter.reshape((5, 1, 1))
    derivative0 = np.array([-0.104550, -0.292315, 0, 0.292315, 0.104550])
    if direction == 'x':
        prefilterfull = np.kron(prefilter1, prefilter)
        derivative = derivative0.reshape((5, 1, 1))
    elif direction == 'y':
        prefilterfull = np.kron(prefilter, prefilter2)
        derivative = derivative0.reshape((5, 1))
    elif direction == 'z':
        prefilterfull = np.kron(prefilter1, prefilter2)
        derivative = derivative0
    else:
        raise ValueError(f"Unknown direction: {direction}. Must be 'x', 'y', or 'z'.")
    return np.kron(prefilterfull, derivative)

def extendflip(a):
    """Extend array: append flipped copy for symmetric periodic arrays."""
    return np.append(a, np.flip(a[:-1]))

def set_axes_equal(ax):
    '''Make axes of 3D plot have equal scale so that spheres appear as spheres,
    cubes as cubes, etc.  This is one possible solution to Matplotlib's
    ax.set_aspect('equal') and ax.axis('equal') not working for 3D.

    Input
      ax: a matplotlib axis, e.g., as output from plt.gca().
    '''
    x_limits = ax.get_xlim3d()
    y_limits = ax.get_ylim3d()
    z_limits = ax.get_zlim3d()

    x_range = abs(x_limits[1] - x_limits[0])
    x_middle = np.mean(x_limits)
    y_range = abs(y_limits[1] - y_limits[0])
    y_middle = np.mean(y_limits)
    z_range = abs(z_limits[1] - z_limits[0])
    z_middle = np.mean(z_limits)

    plot_radius = 0.5*max([x_range, y_range, z_range])
    ax.set_xlim3d([x_middle - plot_radius, x_middle + plot_radius])
    ax.set_ylim3d([y_middle - plot_radius, y_middle + plot_radius])
    ax.set_zlim3d([z_middle - plot_radius, z_middle + plot_radius])

def AAD_surface(array):
    """Compute Average Absolute Deviation of a surface/array."""
    return np.mean(np.abs(array))

def all_errors_2D(true_arr, corr_arr, type_=None, hist=False, boxsize=None, radius=None, AAD2=None):
    """Compute error metrics between true and reconstructed arrays.

    Parameters are intentionally broad for backward compatibility with existing
    call sites across the script.
    """
    del type_, boxsize, radius  # Accepted for compatibility with existing calls.

    true_flat = np.asarray(true_arr).flatten()
    corr_flat = np.asarray(corr_arr).flatten()

    diff = corr_flat - true_flat
    rmse = np.sqrt(np.mean(diff**2))
    mae = np.mean(np.abs(diff))
    mbe = np.mean(diff)

    true_magnitude = np.sqrt(np.mean(true_flat**2))
    if true_magnitude > 0:
        nrmse = rmse / true_magnitude
        nmae = mae / true_magnitude
        nmbe = mbe / true_magnitude
    else:
        nrmse = nmae = nmbe = 0.0

    aad0 = AAD_surface(true_arr)
    aad_est = AAD_surface(corr_arr)

    if AAD2 is None:
        AAD2 = np.sqrt(np.sum([aad0, aad_est, 0])**2)

    if AAD2 > 0:
        nrmse2 = rmse / AAD2
        nmae2 = mae / AAD2
        nmbe2 = mbe / AAD2
    else:
        nrmse2 = nmae2 = nmbe2 = 0.0

    result = (rmse, mae, mbe, nrmse, nmae, nmbe, aad0, aad_est, nrmse2, nmae2, nmbe2, AAD2)
    if hist:
        return result + (diff,)
    return result

def visSH3D_data3(cmesh=None, r0=1, lmax_plot=None, cmap='RdBu',
             vmin=None, vmax=None):
    """Plot reconstructed spherical shape and traction colored 3D plot.
    Returns x, y, z coordinates and colors for 3D visualization of spherical shapes.
    Parameters
    ----------
    cmesh : ndarray, dimension (lmax+1, 2*lmax+1, nd)
        Mesh point representation of displacement SH vector
    r0 : float
        Radius of the original spherical shape
    lmax_plot : int, optional
        If used, the mesh is truncated to the given lmax; If None, determined by mesh size
    cmap : str
        Colormap name
    vmin, vmax : float
        Color value limits
    """
    if lmax_plot is None:
        lmax_plot = cmesh.shape[0] - 1
    lats, lons = _psh.expand.GLQGridCoord(lmax_plot)
    nlat = lats.size; nlon = lons.size;
    lats_circular = np.hstack(([90.], lats, [-90.]))
    lons_circular = np.append(lons, [lons[0]])
    u = np.radians(lons_circular)
    v = np.radians(90. - lats_circular)
    x = r0 * np.sin(v)[:, None] * np.cos(u)[None, :]
    y = r0 * np.sin(v)[:, None] * np.sin(u)[None, :]
    z = r0 * np.cos(v)[:, None] * np.ones_like(lons_circular)[None, :]
    tpoints = np.zeros((nlat + 2, nlon + 1, 3))
    cmesh = np.swapaxes(cmesh, 0, -1)
    cmesh = np.swapaxes(cmesh, 0, 1)
    tpoints[1:-1, :-1, 0] = cmesh
    tpoints[0, :, 0] = np.mean(cmesh[0,:], axis=0)
    tpoints[-1, :, 0] = np.mean(cmesh[-1,:], axis=0)
    tpoints[1:-1, -1, 0] = cmesh[:, 0]
    magn_point = np.sum(tpoints, axis=-1)
    magn_face = 1./4. * (magn_point[1:, 1:] + magn_point[:-1, 1:] +
                            magn_point[1:, :-1] + magn_point[:-1, :-1])
    magnmax_face = np.max(np.abs(magn_face))
    if vmin is None:
        vmin = -magnmax_face
    if vmax is None:
        vmax = magnmax_face
    norm = plt.Normalize(vmin, vmax, clip=True)
    cmap_obj = plt.get_cmap(cmap)
    colors = cmap_obj(norm(magn_face.flatten()))
    colors = colors.reshape(nlat + 1, nlon, 4)
    return x, y, z, colors, cmap_obj, norm

# ============================================================================
# AXIALSYMMETRIC CLASS
# ============================================================================

class Axialsymmetric():
    def __init__(self,E1,v1,E2,v2,savefoldername,radius=1,nmax=20,
                savepath=DEFAULT_RELATIVE_SAVEPATH,
                integralboundaries=False, sigma_zz_bool=False, symmetric=True,
                 norm_radius=1):
        #1 describes the sphere and 2 describes the wall/indenter/...
        #if you set symmetric to False, you need to define a sigma_zz_bottom
        #In this case, the program will automatically calculate the necessary
        #magnitude of sigma_zz_bottom in order for the Scenario to be possible
        #(i.e., the forces are balanced and sphere is not moving)

        # E1 is a material Young's modulus (Pa). Do not scale by radius here;
        # keep E1 in physical units so derived shear modulus has correct magnitude.
        self.E1=E1
        self.norm_radius=norm_radius
        self.v1=v1
        self.E2=E2
        self.v2=v2
        self.calc_E_star()
        self.E=self.E1
        self.pr=self.v1   #Poisson's ratio
        self.G=self.E/(2*(1+self.pr))
        self.mu=self.G
        self.savepath=savepath
        now=date.today()
        self.datestring=now.strftime("%d%b%y")
        self.savefoldername=savefoldername
        self.radius=radius
        self.update_fullpath()
        self.norm_radius=norm_radius

        self.plotextent=[0,1,0,self.radius]
        self.nmax=nmax
        self.coefficient_ns=range(0,self.nmax+1)
        self.coefficients_an=np.zeros((self.nmax+1))
        self.coefficients_bn=np.zeros((self.nmax+1))
        self.symmetric=symmetric
        self.prefactor_bottom=1
        # number of physical contacts for Hertzian contact scenario (1 = single contact, 2 = symmetrical two-plate compression)
        self.contacts = 1

        
        if not self.symmetric:
            print('Setting sigma_zz to asymmetric')
            self.sigma_zz=self.sigma_zz_asymmetric
            self.calc_prefactor()
        
        if integralboundaries==False: #If integralboundaries should be calculated separately, it is defined as True and calculated in a function. Else it's just two boundaries: [0, pi]
            self.integralboundaries=[0,np.pi]
        else:
            self.integralboundaries=integralboundaries
        self.sigma_zz_bool=sigma_zz_bool
        if sigma_zz_bool:
            self.sigma_rr_true=self.sigma_rr_true_zz
            self.sigma_rt_true=self.sigma_rt_true_zz
            
    #Convert from zz to rr and rt using theta
    def sigma_rr_true_zz(self,th):
        return self.sigma_zz(th)*np.cos(th)
    def sigma_rt_true_zz(self,th):
        val=-self.sigma_zz(th)*np.sin(th)
        return val

    def sigma_zz_asymmetric(self,th):   #Extra factor for top/bottom (theta > pi/2) for asymmetric profile (asymmetry in z)
        if th<=np.pi/2:
            return self.sigma_zz_top(th)
        else:
            return self.prefactor_bottom*self.sigma_zz_bottom(th)


    def update_infostring(self,multi_E=False,multi_nu=False):
        if multi_E==False:
            ym=self.E
        else:
            ym=str(multi_E)
        if multi_nu==False:
            pr=self.pr
        else:
            pr=str(multi_nu)
        extrastring=''
        if hasattr(self, 'F'):
            extrastring+='F='+str(self.F)+'_'
        if hasattr(self, 'a'):
            extrastring+='a='+str(self.a)+'_'
        if self.radius!=1:
            extrastring+='Radius='+str(self.radius)+'_'
        self.infostring_all=(extrastring+'E='+str(ym)[:6]+'_nu='+str(pr)+'_')
    def update_fullpath(self,customfoldername=False,custompath=False):
        self.update_infostring()
        if customfoldername!=False:
            self.savefoldername=customfoldername
        if custompath!=False:
            self.savepath=custompath
        self.fullsavepath=self.savepath+self.savefoldername+self.infostring_all+'0/'

    def new_run(self,number):
        self.fullsavepath=self.fullsavepath[:-2]+str(number)+'/'
        os.makedirs(self.fullsavepath)


    def rts(self,x,p=2):
        #rounds to significant digits, #digits specified by p
        return float(np.format_float_scientific(x, precision=p, unique=False, trim='k'))

    def calc_E_star(self):   #from equation (70)
        self.E_star= 1/((1-self.v1**2)/self.E1 + (1-self.v2**2)/self.E2)

    def ortho_relation_L0(self,n):   #from equation (23)
        return 2/(2*n+1)

    def ortho_relation_L1(self,n):   #from equation (24)
        if n==0:
            n0=1  #I'm not sure about this, it seems like a placeholder so there is no division by 0 later.
        else:
            n0=n
        return 2/(2*n+1)*(n+1)*n0   #This is put in a weird way but it agrees.
                                    #P_m_n is undefined for m>0 because the completeness/orthogonality could not be fulfilled since the derivative is always 0.
    

    def get_ortho_relations_pref(self):   #calculates orthogonality relations for n up to cutoff
        self.L0=np.zeros(self.nmax+1)
        self.L1=np.zeros(self.nmax+1)
        for n in range(self.nmax+1):
            self.L0[n]=self.ortho_relation_L0(n)
            self.L1[n]=self.ortho_relation_L1(n)
         
            
    #This is calculated if sigma is asymmetric
    #So to get a prefactor_bottom (that is not the standard value 1), which can appear in the integrals, the integrals have to be calculated without the prefactors beforehand.
    def calc_prefactor(self,return_=False):
        a1_rr_top=self.calc_integral_rr(n=1,toponly=True)
        a1_rr_bottom=self.calc_integral_rr(n=1,bottomonly=True)
        a1_rt_top=self.calc_integral_rt(n=1,toponly=True)
        a1_rt_bottom=self.calc_integral_rt(n=1,bottomonly=True)
        A=-(a1_rr_top+a1_rt_top)/(a1_rr_bottom+a1_rt_bottom)
        if self.prefactor_bottom==1:   #if prefactor_bottom is set to 1 (standard), it gets replaced
            self.prefactor_bottom=A    #this factor is then used to calculate asymmetric sigma,
        else:                          #where sigma bottom is rescaled, and the factor is also used in the integrals later.
            print("Warning! Prefactor was already calculated" )
        if return_:
            return A
        
    def plotdata_u(self,rsteps=100,thsteps=100,thetarange=False,rrange=False,
                   savedata=True,saveplot=False,plot_=False,scale=5):   #changed saveplot and plot_ to False
        #Creates plotdata for u_r, u_th, and the magnitude of both combined.
        self.u_r_plot=np.zeros((rsteps,thsteps))
        self.u_th_plot=np.zeros((rsteps,thsteps))
        self.u_mag_plot=np.zeros((rsteps,thsteps))
        self.X_th=np.linspace(0,np.pi,thsteps)
        self.X_r=np.flip(np.linspace(0,self.radius,rsteps))
        if thetarange!= False:
            self.X_th=np.linspace(thetarange[0],thetarange[1],thsteps)
            self.plotextent[0]=self.rts(thetarange[0]/np.pi)
            self.plotextent[1]=self.rts(thetarange[1]/np.pi)
        if rrange!=False:
            self.X_r=np.flip(np.linspace(rrange[0],rrange[1],rsteps))
            self.plotextent[2]=rrange[0]
            self.plotextent[3]=rrange[1]
        for n,xr in enumerate(self.X_r):
            for nn,xth in enumerate(self.X_th):   #uses u_r and u_th to create data to plot later
                a=self.u_r(xth,xr)
                self.u_r_plot[n][nn]=a
                b=self.u_th(xth,xr)
                self.u_th_plot[n][nn]=b
                self.u_mag_plot[n][nn]=np.sqrt(a**2+b**2)
        self.update_infostring()
        self.all_u_data=[self.u_r_plot,self.u_th_plot,self.u_mag_plot]
        if savedata:
            # Ensure the directory exists before saving
            save_dir = os.path.dirname(self.fullsavepath + 'u_plotdata' + self.infostring_all + 'npy')
            if not os.path.exists(save_dir):
                os.makedirs(save_dir)

            np.save(self.fullsavepath+'u_plotdata'+self.infostring_all+'npy',self.all_u_data)
            print('Saved plotdata to '+self.fullsavepath+'u_plotdata'+self.infostring_all+'npy')



    def integrand_rr(self,th,n,toponly=False,bottomonly=False):
        if not (toponly or bottomonly):   #toponly=False AND bottomonly=False
            return self.sigma_rr_true(th)*np.sin(th)*sp.special.lpmv(0,n,np.cos(th))   
        
        elif toponly and th<=np.pi/2:
            return self.sigma_rr_true(th)*np.sin(th)*sp.special.lpmv(0,n,np.cos(th))
        
        elif bottomonly and th>np.pi/2:
            return self.sigma_rr_true(th)*np.sin(th)*sp.special.lpmv(0,n,np.cos(th))*self.prefactor_bottom
        
        else:
            return 0


    def integrand_rt(self,th,n,toponly=False,bottomonly=False):
        if not (toponly or bottomonly):
            return self.sigma_rt_true(th)*np.sin(th)*sp.special.lpmv(1,n,np.cos(th))*self.prefactor_bottom   #Here the factor is there in this case as well

        elif toponly and th<=np.pi/2:
            return self.sigma_rt_true(th)*np.sin(th)*sp.special.lpmv(1,n,np.cos(th))
        elif bottomonly and th>np.pi/2:
            return self.sigma_rt_true(th)*np.sin(th)*sp.special.lpmv(1,n,np.cos(th))*self.prefactor_bottom #Same here
        else:
            return 0

    ''' 

    These functions then calculate the integrals and introduce the other prefactors:
    So the result already gives the matrix entries from the LHS of equation (27)
    
    For Hertzian contact and a = 0.5, integralboundaries is calculated as [0, np.float64(0.5235987755982989), np.float64(2.617993877991494), 3.141592653589793]
    This will be [0, pi/6, 5pi/6, pi].
    Integral boundaries are split as to allow for a prefactor_bottom in two of the integration boundaries. C.f. integral_rr and integral_rt and then calc_prefactor.

    '''


    def calc_integral_rr(self,n,toponly=False,bottomonly=False,print_integral_value=False):
        bnumber=len(self.integralboundaries)   #This is 2 if integral boundaries are [0, pi]. For Hertzian contact, there are four boundaries that split up the integral domain from 0 to pi.
        I=0
        for nn in range(bnumber-1):   #Then this loop is only one iteration, integral from 0 to pi. Otherwise it always takes two neighboring integration boundaries and iterates from the beginning.
            lower=self.integralboundaries[nn]
            upper=self.integralboundaries[nn+1]
            I_=sp.integrate.quad(self.integrand_rr,lower,upper,args=(n,toponly,bottomonly))[0]
            if print_integral_value:
                print(I_)
            I+=I_    #In the case of only two integralboundaries, the result only arises from 1 term
        prefactor=1/(2*self.G)/(self.ortho_relation_L0(n))   #prefactor with mu and o_nn

        if toponly or bottomonly:   #If either of these is True, then the prefactor of 1/(2mu*o_nn) is not applied. Then the integral can be used to calculate prefactor_bottom in calc_prefactor.
            return I
        else:
            return I*prefactor

    def calc_integral_rt(self,n,toponly=False,bottomonly=False,print_integral_value=False):
        bnumber=len(self.integralboundaries)   #same as above
        I=0
        for nn in range(bnumber-1):
            lower=self.integralboundaries[nn]
            upper=self.integralboundaries[nn+1]
            I_=sp.integrate.quad(self.integrand_rt,lower,upper,args=(n,toponly,bottomonly))[0]
            if print_integral_value:
                print(I_)
            I+=I_
        prefactor=1/(2*self.G)/(self.ortho_relation_L1(n))
        if toponly or bottomonly:
            return I
        else:
            return I*prefactor

    #This calculates some error quantity AAD but I don't see why it is calculated like that
    #Maybe this is an error approximation for the integral

    def integrand_AAD(self,th):
        val=np.sin(th)*np.sqrt(self.sigma_rt_true(th)**2+self.sigma_rr_true(th)**2)
        return val

    def calc_integral_AAD2_prefactor(self):
        I=sp.integrate.quad(self.integrand_AAD,0,np.pi,)[0]
        self.AAD2=I/np.pi

    def calc_coeff_matrix(self,n):
        c_a_rr=(n+1)*(n**2-n-2-2*self.pr)*self.radius**n   #epsilon_n_(0)
        c_b_rr=n*(n-1)*self.radius**(n-2)   #alpha_n_(0), but we removed the - sign
        c_a_rt=(n**2+2*n-1+2*self.pr)*self.radius**n   #epsilon_n_(1)
        c_b_rt=(n-1)*self.radius**(n-2)   #alpha_n_(1), but we removed the - sign
        #The matrix is defined with the epsilons in the first column and alphas in the second, this explains why later a_n and e_n have to be swapped.
        return np.array([[c_a_rr,c_b_rr],[c_a_rt,c_b_rt]])
    


    def calc_and_store_lgs(self,printsteps=False):
        for n in range(0,self.nmax+1):   #Equation (27) is for each n from 0 to n_max
            if self.symmetric:
                if n % 2==0:
                    iterator=True
                else:
                    iterator=False   #no iteration if n is odd in the symmetric case (?)
            else:
                iterator=True

            if iterator:   #The LHS in equation (27) is obtained by integration (equations (25, 26)), including the prefactor with mu and o_nn
                I0=self.calc_integral_rr(n,print_integral_value=printsteps)
                I1=self.calc_integral_rt(n,print_integral_value=printsteps)
                I=np.array([I0,I1])   #LHS of equation (27) (including prefactors)

                A=self.calc_coeff_matrix(n)   #RHS matrix of equation (27), but with alpha_n_(0) and alpha_n_(1) missing a - sign and epsilons and alphas being swapped.

                if printsteps:
                    print(n)
                    print(I)
                    print(A)

                if np.linalg.det(A)==0:   #if detA=0, no inverse of A exists. Infinite or no solutions
                    a=I0/A[0][0]   #so a_n is I_n_(0)/(2mu*o_nn*alpha_n_(0))
                    b=0   #so e_n is 0; in that case, a_n checks out.

                else:
                    a,b=np.linalg.solve(A,I)   #this gives the (actual) values of a_n, e_n (RHS vector). b is supposed to be e_n
                if printsteps:
                    print(a)
                    print(b)
                    print(n)
                    #entering the a_n and e_n in arrays at position n:
                self.coefficients_an[n]=a
                if n>1:
                    self.coefficients_bn[n]=b

                else:   #e_0=0 since o_00_(0)=0 and alpha_0_(0)=0, so 0 = 0 + epsilon_0_(0) * e_0
                    self.coefficients_bn[n]=0


             #coefficients without r and Legendre polynomials
             #a_n and e_n were swapped from before to accommodate the switched matrix (epsilons and alphas)
             #we removed the - signs in the first term for each equation, which
             #fixes the sign errors from before



    def calc_other_coefficients(self):  
        self.coefficients_an_u_r=np.zeros((self.nmax+1))
        self.coefficients_bn_u_r=np.zeros((self.nmax+1))

        self.coefficients_an_u_th=np.zeros((self.nmax+1))
        self.coefficients_bn_u_th=np.zeros((self.nmax+1))

        self.coefficients_an_sigma_rr=np.zeros((self.nmax+1))
        self.coefficients_bn_sigma_rr=np.zeros((self.nmax+1))

        self.coefficients_an_sigma_rth=np.zeros((self.nmax+1))
        self.coefficients_bn_sigma_rth=np.zeros((self.nmax+1))

        for n in self.coefficient_ns:   #this goes from 0 to n_max
            
            an=self.coefficients_an[n]   #an and bn are from above, from the LGS, where they had the missing - sign from
            bn=self.coefficients_bn[n]

            #Now calculate the prefactors for displacements and stresses from a_n and e_n.
            #a_n and b_n (e_n) are swapped.
            #The sign difference from before is also fixed with the - signs missing here.

            self.coefficients_an_u_r[n]=an*(n+1)*(n-2+4*self.v1)   #second term in (17)
            self.coefficients_bn_u_r[n]=bn*n   #first term in (17), - sign fixed from alpha_n_(0)

            self.coefficients_an_u_th[n]=an*(n+5-4*self.v1)   #second term in (18)
            self.coefficients_bn_u_th[n]=bn   #first term in (18), - sign fixed from alpha_n_(1)

            self.coefficients_an_sigma_rr[n]=an*(n+1)*(n**2-n-2-2*self.v1)   #second term in (19)
            self.coefficients_bn_sigma_rr[n]=bn*n*(n-1)   #first term in (19), - sign again

            self.coefficients_an_sigma_rth[n]=an*(n**2+2*n-1+2*self.v1)   #second term in (20)
            self.coefficients_bn_sigma_rth[n]=bn*(n-1)   #first term in (20), - sign again



    def u_r(self,theta,r=None):
        value=0
        if r is None:
            r=self.radius
        if r==0 or r>self.radius*1.1:
            return 0
        else:
            for n in self.coefficient_ns:
                value+=(self.coefficients_an_u_r[n]*r**(n+1)+self.coefficients_bn_u_r[n]*r**(n-1))*sp.special.lpmv(0,n,np.cos(theta))
            return value

        
    def u_th(self,theta,r=None):
        value=0
        if r is None:
            r=self.radius
        if r==0 or r>self.radius*1.1:
            return 0
        else:
            for n in self.coefficient_ns:
                value+=(self.coefficients_an_u_th[n]*r**(n+1)+self.coefficients_bn_u_th[n]*r**(n-1))*sp.special.lpmv(1,n,np.cos(theta))
            return value
        
    def generate_analytic_surface_displacement(self, coord_path, out_path,
                                              coord_order='xyz', overwrite=True):
        """Generate analytic Hertzian surface displacement on coordinates saved in coord_path.
        Saves cartesian displacement array of shape (N,3) to out_path.
        """
        coords_raw = np.load(coord_path, allow_pickle=True)
        orig_shape = coords_raw.shape
        coords = np.reshape(coords_raw, (-1,3))
        x = coords[:,0]; y = coords[:,1]; z = coords[:,2]
        r_actual = np.linalg.norm(coords, axis=1)
        # avoid divide by zero
        valid = r_actual > 0
        theta = np.zeros_like(r_actual)
        phi = np.zeros_like(r_actual)
        theta[valid] = np.arccos(np.clip(z[valid]/r_actual[valid], -1, 1))
        phi[valid] = np.arctan2(y[valid], x[valid])
        # Map physical radius to the dimensionless radius expected by u_r/u_th
        # Detect likely units: if coords are O(1e1) assume microns, else assume same units as norm_radius
        if np.median(r_actual) > 1:
            norm_factor = float(self.norm_radius) * 1e6
        else:
            norm_factor = float(self.norm_radius)
        r_arg = r_actual / norm_factor
        # Evaluate spherical components
        u_r_vals = np.zeros_like(r_arg)
        u_th_vals = np.zeros_like(r_arg)
        for i in range(len(r_arg)):
            u_r_vals[i] = float(self.u_r(theta[i], r_arg[i]))
            u_th_vals[i] = float(self.u_th(theta[i], r_arg[i]))
        # Spherical unit vectors (physics convention: theta = polar colatitude)
        sin_t = np.sin(theta); cos_t = np.cos(theta)
        cos_p = np.cos(phi); sin_p = np.sin(phi)
        e_r = np.column_stack((sin_t*cos_p, sin_t*sin_p, cos_t))
        e_th = np.column_stack((cos_t*cos_p, cos_t*sin_p, -sin_t))
        u_cart = (u_r_vals[:,None]*e_r) + (u_th_vals[:,None]*e_th)
        if not overwrite:
            if os.path.exists(out_path):
                raise FileExistsError(out_path + ' exists')
        # reshape back to original coordinate grid shape if appropriate
        if len(orig_shape) == 3 and orig_shape[2] == 3:
            u_cart = np.reshape(u_cart, orig_shape)
        np.save(out_path, u_cart)
        print('Saved analytic displacement to', out_path)
        


    def sigma_rr(self,theta,r=None):
        value=0
        if r is None:
            r=self.radius
        if r==0:
            return 0
        else:
            for n in self.coefficient_ns:
                value+=(self.coefficients_an_sigma_rr[n]*r**n+self.coefficients_bn_sigma_rr[n]*r**(n-2))*sp.special.lpmv(0,n,np.cos(theta))
            return 2*self.G*value   #2mu is in front of the sum
        
    def sigma_rt(self,theta,r=None):
        value=0
        if r is None:
            r=self.radius
        for n in self.coefficient_ns:
            if n!=0:
                value+=(self.coefficients_an_sigma_rth[n]*r**n+self.coefficients_bn_sigma_rth[n]*r**(n-2))*sp.special.lpmv(1,n,np.cos(theta))
        return 2*self.G*value



    def GLQCartCoord(self,lmax,extend=False):
        '''Compute the Cartesian coordinates used in Gauss-Legendre quadrature grids on
            a unit sphere.
            Usage
            -----
            X0 = GLQCartCoord (lmax)
            Returns
            -------
            X0 : float, dimension (lmax+1, 2*lmax+1, 3)
                The Cartesian coordinates of a GLQ grid on a unit sphere
            Parameters
            ----------
            lmax : integer
                The maximum spherical harmonic degree that will be integrated exactly by
                Gauss-Legendre quadrature.
        '''
        latsdeg, lonsdeg = _psh.expand.GLQGridCoord(lmax,extend=extend)
        #Lattitude and longitude coordinates from GLQ grid
        lon = np.deg2rad(lonsdeg)
        colat = np.deg2rad(90-latsdeg)
        PHI, THETA = np.meshgrid(lon, colat)   #makes a coordinate grid (on sphere)
        R = np.ones_like(PHI)
        X,Y,Z = self.SphCoord_to_CartCoord(R, THETA, PHI)   #converts cartesian coordinates
        return np.stack([X,Y,Z], axis=-1)

    def SphCoord_to_CartCoord(self, R, THETA, PHI):
        """translate spherical coordinates into Cartesian coordinates
            Parameters
            ----------
            R,THETA,PHI : ndarray
                make sure R, THETA, PHI have the same dimensions

            Returns
            -------
            X,Y,Z : ndarray
                same size as R, THETA, PHI.

        """
        Z = R * np.cos(THETA)
        X = R * np.sin(THETA) * np.cos(PHI)
        Y = R * np.sin(THETA) * np.sin(PHI)
        return (X, Y, Z)

    def distance(self,a,b):
            l=len(a)
            return np.sqrt(sum([(a[n]-b[n])**2 for n in range (l)]))
            #sqrt((a1-b1)² + (a2-b2)² + ... (al-bl)²)
            
    def cart_to_sph_coords(self,vector):
        #input [x,y,z]
        #output [r,theta,phi]
        x,y,z=vector
        r=np.linalg.norm(vector)
        '''
        if y==0:
            if x>=0:
                phi=0
            else:
                phi=np.pi
        elif x==0:
            if y>=0:
                phi=np.pi/2
            else:
                phi=3*np.pi/2
        else:
        '''
        phi=np.arctan2(y,x)
        if z==0:
            theta=np.pi/2
        else:
            theta=np.arccos(z/r)
        return np.array([r,theta,phi])

        #Use the matrix A including theta, phi to transform between coordinate systems
        #Transposed matrix transforms back: v_c = A v_s and A⁻¹ = A^T since A is orthogonal
        #A represents basis vector orientations to Cartesian coordinate system (3x3)
        
    def vector_sph_to_cart(self,vector,theta,phi):
        A=np.array([[np.sin(theta)*np.cos(phi),np.cos(theta)*np.cos(phi),-np.sin(phi)],
            [np.sin(theta)*np.sin(phi),np.cos(theta)*np.sin(phi),np.cos(phi)],
            [np.cos(theta),-np.sin(theta),0]])
        return A.dot(vector)
    def vector_cart_to_sph(self,vector,theta,phi):
        A=np.array([[np.sin(theta)*np.cos(phi),np.cos(theta)*np.cos(phi),-np.sin(phi)],
            [np.sin(theta)*np.sin(phi),np.cos(theta)*np.sin(phi),np.cos(phi)],
            [np.cos(theta),-np.sin(theta),0]]).transpose()
        return A.dot(vector)

    def tensor_cart_to_sph(self,vector,theta,phi):
        A=np.array([[np.sin(theta)*np.cos(phi),np.cos(theta)*np.cos(phi),-np.sin(phi)],
            [np.sin(theta)*np.sin(phi),np.cos(theta)*np.sin(phi),np.cos(phi)],
            [np.cos(theta),-np.sin(theta),0]])
        return A.dot(A.transpose().dot(vector))
    def tensor_sph_to_cart(self,vector,theta,phi):
        A=np.array([[np.sin(theta)*np.cos(phi),np.cos(theta)*np.cos(phi),-np.sin(phi)],
            [np.sin(theta)*np.sin(phi),np.cos(theta)*np.sin(phi),np.cos(phi)],
            [np.cos(theta),-np.sin(theta),0]])
        return A.transpose().dot(A.dot(vector))


    def u_quarterplot(self,dpi=300):
        plt.rcParams['figure.figsize']=[5,5]
        numpoints=400
        xrange=np.linspace(-1,0,numpoints)
        zrange=np.sqrt(1-xrange**2)
        plt.plot(xrange,zrange,label='undeformed')   #plot quarter arc without deformations
        _,th,p=CartCoord_to_SphCoord(xrange,np.zeros_like(xrange),zrange)   #don't need radius: _
        xrange2=xrange.copy()
        zrange2=zrange.copy()
        dispx=np.zeros_like(xrange)
        dispz=np.zeros_like(zrange)
        for n in range(numpoints):
            thn=th[n]
            ur=self.u_r(thn)   #u_r is the function evaluating equation (17) etc
            #print(ur)
            ut=self.u_th(thn)
            u=[ur,ut,0]   #u_theta = 0. We are in the r-theta plane and u_theta would be out of plane
            ux,uy,uz=self.vector_sph_to_cart(u,thn,-np.pi)   #transform to cartesian, save
            xrange2[n]+=ux
            zrange2[n]+=uz
            dispx[n]=ux
            dispz[n]=uz
        plt.plot(xrange2,zrange2,label='deformed')
        thetas=np.linspace(np.pi/2,0,11)
        x_arrows=np.int64((1-np.sin(thetas))*numpoints)-1   #position of arrows
        #for n in range(0,numpoints,40):
        for n in x_arrows[1:]:   #Plot arrows showing the displacement. plt.arrow docs for info
            plt.arrow(xrange[n],zrange[n],dispx[n],dispz[n],length_includes_head=True,head_width=0.02, head_length=0.02)

        plt.axis('equal')   #Have spheres stay spheres in the 3D plots
        plt.axis('off')
        plt.legend()
        filename=self.fullsavepath+'u_quarterplot'+self.infostring_all+".svg"
        plt.savefig(filename,format='svg',dpi=dpi)
        #plt.show()

    def u_halfplot(self, dpi=300, add2a=False, forringrange=0, rangefactor=0,
                ztop=1.2, add_axis_arrow=True):
            factor = 1 + rangefactor

            mpl.use("Agg")
            plt.rcParams['figure.figsize'] = [10, 5 * factor]

            numpoints = 400
            xrange = np.linspace(-1, 0, numpoints)
            zrange = np.sqrt(1 - xrange ** 2)
            xrange0 = np.linspace(-1, 1, numpoints * 2 - 1)
            zrange0 = extendflip(zrange)

            # Plot undeformed profile
            plt.plot(xrange0, zrange0, label='Undeformed', color='#1f77b4', linewidth=2)

            _, th, p = CartCoord_to_SphCoord(xrange, np.zeros_like(xrange), zrange)
            xrange2 = xrange.copy()
            zrange2 = zrange.copy()
            dispx = np.zeros_like(xrange)
            dispz = np.zeros_like(zrange)

            # Compute deformed positions
            for n in range(numpoints):
                thn = th[n]
                ur = self.u_r(thn)
                ut = self.u_th(thn)
                u = [ur, ut, 0]
                ux, uy, uz = self.vector_sph_to_cart(u, thn, -np.pi)
                xrange2[n] += ux
                zrange2[n] += uz
                dispx[n] = ux
                dispz[n] = uz

            # Extend to half-sphere
            xrange3 = np.append(xrange2, np.abs(np.flip(xrange2[:-1])))
            zrange3 = extendflip(zrange2)
            dispx2 = extendflip(dispx)
            dispz2 = extendflip(dispz)

            # Plot deformed profile
            plt.plot(xrange3, zrange3, label='Deformed', color='#ff7f0e', linewidth=2)

            # Draw displacement arrows
            thetas = np.linspace(np.pi / 2, 0, 11)
            x_arrows = np.int64((1 - np.sin(thetas)) * numpoints) - 1
            x_arrows2 = np.append(x_arrows, np.flip(2 * numpoints - (x_arrows[1:])) - 1)

            # Draw displacement arrows (black)
            for n in x_arrows2[1:]:
                dx = dispx2[n] if xrange0[n] <= 0 else -dispx2[n]
                dz = dispz2[n]
                plt.arrow(xrange0[n], zrange0[n], dx, dz,
                        length_includes_head=True, head_width=0.02,
                        head_length=0.02, color='black')

            # Optional: add 2a bar for contact area
            if add2a:
                X1 = np.linspace(-self.a, self.a, 50)
                plt.plot(X1, (ztop + 0.03) * self.radius * np.ones_like(X1), color='black')
                plt.text(-0.03, ztop + 0.06, '2a', fontsize=17)

            # Add z-axis in the middle to indicate axisymmetry
            if add_axis_arrow:
                z_axis = np.linspace(0, ztop, 50)
                x_axis = np.zeros_like(z_axis)
                plt.plot(x_axis, z_axis, linestyle=(0, (3, 6)), color='black', linewidth=1)
                plt.arrow(0, ztop * 0.9, 0, ztop * 0.1, head_width=0.02, head_length=0.04,
                        color='black', length_includes_head=True)
                plt.text(0.04, ztop - 0.05, 'z', fontsize=20, color='black')

            # --- Curved rotation arrow at the bottom center ---
            try:
                ax = plt.gca()
                # move the arc slightly up so it sits around the axis
                arc_center = (0.0, 0.04 * self.radius)
                arc_radius = 0.22 * self.radius
                # increase angular span to wrap more than 180°
                theta_start_deg = 100
                theta_end_deg = 430
                N = 240
                angles = np.linspace(np.deg2rad(theta_start_deg), np.deg2rad(theta_end_deg), N)

                # To give a 'tilted into 3D' look, compress the vertical component (elliptical projection)
                tilt = 0.55  # 1.0 = circle, <1 squashes vertically to simulate tilt
                arc_x = arc_center[0] + arc_radius * np.cos(angles) + 0.02 * self.radius * np.sin(angles)
                arc_z = arc_center[1] + (arc_radius * np.sin(angles)) * tilt

                # Reserve a small segment at the very end for the arrowhead so the circle doesn't overshoot
                reserve = 8
                if len(arc_x) > reserve:
                    ax.plot(arc_x[:-reserve], arc_z[:-reserve], color='black', lw=2, zorder=10)
                else:
                    ax.plot(arc_x, arc_z, color='black', lw=2, zorder=10)

                # compute arrow start/end using the actual plotted coordinates so tip aligns exactly
                end = (arc_x[-1], arc_z[-1])
                start = (arc_x[-reserve - 1], arc_z[-reserve - 1])

                arrow = FancyArrowPatch(start, end, mutation_scale=18, lw=2, arrowstyle='-|>', color='black', zorder=11)
                ax.add_patch(arrow)
            except Exception as e:
                # If anything fails drawing the arc, continue silently (keeps backward compatibility)
                print("Warning: could not draw curved rotation arrow:", e)

            plt.axis('equal')
            plt.axis('off')

            # Legend: move slightly left and down
            plt.legend(loc='upper right', bbox_to_anchor=(0.96, 0.93), prop={'size': 15}, frameon=False)

            # Save figure
            filename = self.fullsavepath + "halfplotaxisym.png"
            print(f'Saving halfplot to {filename}')
            plt.savefig(filename, format='png', dpi=dpi, bbox_inches='tight', pad_inches=0)
            plt.gcf().set_dpi(dpi)
            plt.close()


    def get_u_t_tractionfree(self,lmax,surface_vec=False,extend=True,
                         force_on_y=False, traction_free_weights=False):
        """Compute displacement, traction, and traction-free regions on GLQ grid.
        
        Parameters
        ----------
        lmax : int
            Maximum SH degree integrated exactly by GLQ
        surface_vec : bool
            If True, add cartesian coordinates of each grid point including displacements
        extend : bool
            If True, include 360°E longitude
        force_on_y : bool
            If True, compute u and T based on transformations of spherical coordinates
        traction_free_weights : bool
            If True, add weighted traction-free data
        """
        latsdeg, lonsdeg = _psh.expand.GLQGridCoord(lmax,extend=extend)
        latsdeg=90-latsdeg
        n_lat=len(latsdeg)
        n_lon=len(lonsdeg)
        u_mesh=np.zeros((n_lat,n_lon,3))
        traction_mesh=np.zeros((n_lat,n_lon,3))
        traction_free=np.zeros((n_lat,n_lon),dtype=bool)
        traction_free_weighted=np.zeros((n_lat,n_lon))
        
        if force_on_y:
            for n,theta1 in enumerate(latsdeg):
                theta1=theta1*np.pi / 180
                for nn,phi1 in enumerate(lonsdeg):
                    phi1=phi1*np.pi / 180
                    theta=np.arccos(np.sin(phi1)*np.cos(theta1))
                    phi=np.arctan2(np.cos(theta1),np.sin(theta1)*np.sin(phi1))
                    ur=self.u_r(theta)
                    ut=self.u_th(theta)
                    ux,uy,uz=self.vector_sph_to_cart(np.array([ur,ut,0]),theta,phi)
                    u_cart1=np.array([ux,uz,uy])
                    if self.sigma_zz_bool:
                        T_z=self.sigma_zz(theta)
                        if T_z==0:
                            traction_free[n][nn]=True
                        T_cart1=np.array([0,T_z,0])
                    else:
                        Tr=self.sigma_rr(theta)
                        Tt=self.sigma_rt(theta)
                        T_sph=np.array([Tr,Tt,0])
                        Tx,Ty,Tz=self.vector_sph_to_cart(T_sph,theta,phi)
                        T_cart1=np.array([Tx,Tz,Ty])
                        if Tr==0 and Tt==0:
                            traction_free[n][nn]=True
                            traction_free_weighted[n][nn]=np.sin(theta1)
                    u_mesh[n][nn]=u_cart1   
                    traction_mesh[n][nn]=T_cart1
        else:
            for n,theta0 in enumerate(latsdeg):
                theta=theta0*np.pi / 180
                u_r=self.u_r(theta)
                u_th=self.u_th(theta)
                if self.sigma_zz_bool:
                    T_z=self.sigma_zz(theta)
                    if self.sigma_zz(theta)==0:
                        traction_free[n]=np.ones((n_lon,))
                        traction_free_weighted[n]=np.ones((n_lon,))*np.sin(theta)
                else:
                    Tr=self.sigma_rr_true(theta)
                    Tt=self.sigma_rt_true(theta)
                    T_sph=np.array([Tr,Tt,0])
                    if (Tr==0 and Tt==0):
                        traction_free[n]=np.ones((n_lon,))
                        traction_free_weighted[n]=np.ones((n_lon,))*np.sin(theta)
                u_sph=[u_r,u_th,0]

                for nn,phi in enumerate(lonsdeg):
                    phi=phi*np.pi / 180
                    A=np.array([[np.sin(theta)*np.cos(phi),np.cos(theta)*np.cos(phi),-np.sin(phi)],
                        [np.sin(theta)*np.sin(phi),np.cos(theta)*np.sin(phi),np.cos(phi)],
                        [np.cos(theta),-np.sin(theta),0]])
                    u_cart=A.dot(u_sph)
                    u_mesh[n][nn]=u_cart
                    if self.sigma_zz_bool:
                        traction_mesh[n][nn]=np.array([0,0,T_z])
                    else:
                        T_cart=A.dot(T_sph)
                        traction_mesh[n][nn]=T_cart
        Data=[traction_mesh, u_mesh, traction_free]

        if surface_vec:
            VX0 = self.GLQCartCoord(lmax,extend=True)*self.radius
            Vp = (VX0 + u_mesh).reshape(-1, 3)
            Data.append(Vp)

        if traction_free_weights:
            Data.append(traction_free_weighted)

        return Data

    def export_SHE_GLQ(self,lmax=100,lmax_low=40, save_=True, save_low_T_u=True,
                       export_=False, flipyz=False,exporthighres_t=False,GLQ=True):
        #exports u_calc, sigma_true and the Traction-free-region on the GLQ Gridcoords
        #needed for the SH-Elastic package. We need to change the representation
        #to cartesian vectors before though for the u_calc for compability with
        #the other package
        #currently only works for Traction BC || z-direction
        
        if GLQ==True:   #Use the function get_u_t_tractionfree to compute data. surface_vec is True, so the standard option False is not used.
            traction_mesh,u_mesh,traction_free,Vp=self.get_u_t_tractionfree(lmax,surface_vec=True,
                                                                                    force_on_y=flipyz)   #good res
            traction_mesh_low, u_mesh_low, traction_free_low,tf_weights_low=self.get_u_t_tractionfree(
                lmax_low,force_on_y=flipyz, traction_free_weights=True)   #low res
            
        else:   #Need data from elsewhere
            self.all_exp_data=[traction_mesh,u_mesh ,Vp, traction_free]
            labels=['T_mesh','u_mesh_','r_surface_','traction_free_surface_']
            
        if save_:   #Save according data
            for n in range(len(labels)):   #But labels is only defined if GLQ = False and needs all_exp_data
                savepath_full=self.fullsavepath+'export_data_SHE_lmax='+str(lmax)+labels[n]+self.infostring_all
                np.save(savepath_full,self.all_exp_data[n])
                if n==len(labels)-1:
                    print('\''+savepath_full+'\'')
                else:
                    print('\''+savepath_full+'\',')
            if save_low_T_u:
                self.all_exp_data=[traction_mesh_low,u_mesh_low ,Vp, traction_free_low]
                labels=['T_mesh','u_mesh_','r_surface_','traction_free_surface_']
                for n in range(len(labels)):
                    savepath_full=self.fullsavepath+'export_data_SHE_lmax='+str(lmax_low)+labels[n]+self.infostring_all
                    np.save(savepath_full,self.all_exp_data[n])
                    if n==len(labels)-1:
                        print('\''+savepath_full+'\'')
                    else:
                        print('\''+savepath_full+'\',')
        if export_:
            return traction_mesh_low, u_mesh_low,  traction_free, Vp,tf_weights_low   #low res
        if exporthighres_t:
            return traction_mesh   #high res
    
    def fibonacci_sphere(self, samples=None, halfz=False):
        """Generate uniformly distributed points on a sphere using Fibonacci spiral.
        
        Parameters
        ----------
        samples : int, optional
            Number of points to generate. If None, calculated from self.res
        halfz : bool, optional
            If True, only generate points on upper hemisphere (z > 0)
            
        Returns
        -------
        points : ndarray, shape (samples, 3)
            Cartesian coordinates (x, y, z) on unit sphere
        pointslatslons : ndarray, shape (samples, 2)
            Spherical coordinates (colatitude theta, azimuthal phi) in radians
        """
        if samples is None:
            surfacesphere = 4*np.pi if not halfz else 2*np.pi
            surfaceperpixel = np.min(self.res)**2
            samples = int(surfacesphere/surfaceperpixel)
            print('Number of Fibonacci samples:', samples)
        
        points = []
        pointslatslons = []
        phi_golden = np.pi * (np.sqrt(5.) - 1.)  # golden angle in radians
        
        total_samples = samples if not halfz else samples * 2
        
        for i in range(total_samples):
            y = 1 - (i / float(total_samples - 1)) * 2  # y goes from 1 to -1
            
            if halfz and y < 0:  # Skip lower hemisphere if halfz=True
                continue
                
            radius_xy = np.sqrt(1 - y * y)  # radius at y
            theta = phi_golden * i  # golden angle increment
            x = np.cos(theta) * radius_xy
            z = np.sin(theta) * radius_xy
            
            points.append([x, y, z])
            r, colat, azim = self.cart_to_sph_coords([x, y, z])
            pointslatslons.append([colat, azim])
        
        return np.array(points), np.array(pointslatslons)

    def get_u_t_tractionfree_fibonacci(self, samples=None, surface_vec=False,
                                      force_on_y=False, halfz=False):
        """Generate displacement and traction data on Fibonacci grid points.
        
        This is an alternative to get_u_t_tractionfree that uses a Fibonacci
        spiral grid instead of Gauss-Legendre Quadrature for more uniform
        point distribution on the sphere surface.
        
        Parameters
        ----------
        samples : int, optional
            Number of Fibonacci grid points. If None, estimated from self.res
        surface_vec : bool
            If True, return surface point positions with displacement
        force_on_y : bool
            If True, apply coordinate transformation (y-axis oriented)
        halfz : bool
            If True, only sample upper hemisphere
            
        Returns
        -------
        Data : list
            [traction_array, displacement_array, traction_free_mask, surface_points]
            Arrays have shape (n_points, 3) instead of (n_lat, n_lon, 3)
        """
        # Generate Fibonacci sphere points
        points_cart, points_sph = self.fibonacci_sphere(samples=samples, halfz=halfz)
        n_points = len(points_cart)
        
        print(f'Generated {n_points} Fibonacci grid points')
        
        # Initialize arrays - note: 1D structure instead of 2D grid
        u_array = np.zeros((n_points, 3))
        traction_array = np.zeros((n_points, 3))
        traction_free = np.zeros(n_points, dtype=bool)
        
        # Evaluate at each Fibonacci point
        for i, (xyz, thetaphi) in enumerate(zip(points_cart, points_sph)):
            theta, phi = thetaphi  # colatitude, azimuthal angle in radians
            
            if force_on_y:
                # Apply coordinate transformation
                lat = np.pi/2 - theta  # convert colatitude to latitude
                theta_new = np.arccos(np.sin(phi)*np.cos(lat))
                phi_new = np.arctan2(np.cos(lat), np.sin(lat)*np.sin(phi))
                theta, phi = theta_new, phi_new
            
            # Compute displacement
            ur = self.u_r(theta)
            ut = self.u_th(theta)
            u_sph = np.array([ur, ut, 0])
            
            # Transform to Cartesian using transformation matrix
            A = np.array([[np.sin(theta)*np.cos(phi), np.cos(theta)*np.cos(phi), -np.sin(phi)],
                         [np.sin(theta)*np.sin(phi), np.cos(theta)*np.sin(phi),  np.cos(phi)],
                         [np.cos(theta),             -np.sin(theta),              0]])
            u_cart = A.dot(u_sph)
            u_array[i] = u_cart
            
            # Compute traction
            if self.sigma_zz_bool:
                T_z = self.sigma_zz(theta)
                if T_z == 0:
                    traction_free[i] = True
                traction_array[i] = np.array([0, 0, T_z])
            else:
                Tr = self.sigma_rr(theta)
                Tt = self.sigma_rt(theta)
                T_sph = np.array([Tr, Tt, 0])
                T_cart = A.dot(T_sph)
                traction_array[i] = T_cart
                if Tr == 0 and Tt == 0:
                    traction_free[i] = True
        
        Data = [traction_array, u_array, traction_free]
        
        if surface_vec:
            # Compute surface point positions (radius * unit_vector + displacement)
            Vp = self.radius * points_cart + u_array
            Data.append(Vp)
        
        return Data

    #Generate positions for a specified number of beads in 3D under constraints of volume, density and self-exclusion
    def generate_bead_positions(self,ignorevolume=False):
        lowertresh = - self.radius*self.boxsize
        uppertresh = self.radius*self.boxsize
        numbeads=int(self.beaddensity*self.V)   # rho = N / V, therefore N = rho * V
        reference_bead_positions=np.zeros((numbeads,3))
        if ignorevolume:   #overlaps are allowed
            reference_bead_positions0=np.random.uniform(lowertresh,uppertresh,(numbeads,3))
            #For each bead, generate 3 random coordinates in volume
            distances=np.linalg.norm(reference_bead_positions0,axis=-1)
            reference_bead_positions=reference_bead_positions0[distances<=self.radius]   #Remove outside beads
        else:   #no overlaps allowed
            reference_bead_positions=np.zeros((numbeads,3))
            for n in tqdm(range(numbeads), desc='Placing beads', unit='bead'):
                bead_coord=np.random.uniform(lowertresh,uppertresh,(1,3))   #Random coordinates
                smalldistance=True
                while smalldistance:
                    dm=sp.spatial.distance_matrix(bead_coord,reference_bead_positions)
                    #Matrix of all pair-wise distances, computationally expensive!
                    if (np.min(dm)>self.beaddiameter and np.linalg.norm(bead_coord)<self.radius):
                        #If the smallest pair-wise distance is larger than the diameter and the coordinate is within
                        #the volume, keep it.
                        reference_bead_positions[n]=bead_coord
                        smalldistance=False
                    else:
                        bead_coord=np.random.uniform(lowertresh,uppertresh,(1,3))
                        #Else, reroll the position
                        
        return reference_bead_positions


    #Displace bead positions by the computed displacement values
    #Add potential rotation here to test how it affects volume correlation?

    def generate_displaced_bead_positions(self, array, pixelshift=False):
        '''
        Displaces bead positions.
        - If pixelshift is True: applies fixed displacement to all beads.
        - Otherwise: uses spherical displacement functions u_r and u_th.
        '''

        displaced_beads=np.zeros_like(array)   #Takes bead positions as input "array"
        if pixelshift:  
            #pixelshift to add a fixed displacement
            print('pixelshift is True')
            for n,p in enumerate(array):
                u=np.array([0,0,2])*self.res
                displaced_beads[n]=p+u

        else:
            for n,p in enumerate(array):
                r,th,phi=self.cart_to_sph_coords(p)   #convert displacements to spherical coordinates
                u_r=self.u_r(th,r)
                u_t=self.u_th(th,r)
                u=self.vector_sph_to_cart([u_r,u_t,0],th,phi)   #convert displacements to spherical coordinates
                displaced_beads[n]=p+u   #Add corresponding displacement u to each bead position p for each bead n
        return displaced_beads

    def psf(self,distance,I0_amp=1,decayconstant_rel=1):   #Point spread function
        decayconstant=decayconstant_rel*self.beaddiameter   #decay of PSF depends on bead diameter and a constant (1)
        if distance/self.beaddiameter<3:   
            return abs(I0_amp*np.exp(-0.5*(distance/decayconstant)**4))   #formula for PSF with strong exp decay

        else:
            return 0
        #PSF considered negligible for normalized distances less than 3
    
    #Generate 3D image data representation of beads based on coordinates, applying the PSF, and introducing noise. 
    def generate_image_data(self,bead_coords,res_=None,noise_amp=0,disp_noise_amp=0):
        print('Generating image data')
        if res_==None:   #If no res is given, use original res from class
            res=self.res
        else:
            res=self.res_
        self.N=self.l/res   #Total number of beads N = length / resolution, can be non-integer
        # Dimensions of self.N:
        #   self.l = 2*self.boxsize*self.radius       in generate_image_pair, so scalar
        #   res=np.array([0.015,0.015,0.03])          as an example argument in generate_image_pair, so 3D array
        # N is a 3D array


        I0=np.zeros(np.rint(self.N).astype(np.int32))   #Round and convert N to integer, initialize intensity 3D array
        # The shape of I0 is determined by rounding the entries in N, yielding the number of pixels in each direction

        offsetx=offsety=np.rint(self.N[0]/2).astype(np.int32)   #Offsets to x,y to center beads within image grid
        offsetz=np.rint(self.N[-1]/2).astype(np.int32)   #Offsets to z
        print('Array size: ',self.N)
        print('Offsetx: ',offsetx)
        print('Offsetz: ',offsetz)
        offset=np.array([offsetx,offsety,offsetz])
        width=3   #Beads will influence neighbors in a cube with side length "width"
        
        #pc means pixelcoordinate
        #so for each bead_coordinate, I calculate the index values for the
        #neighbouring pixels (and if necessary, cap at 0 or N), and for each of
        #those I add the intensity value of the respective bead
        for p in tqdm(bead_coords, desc='Generating image data', unit='bead'):
            ratio=p/self.l   #Determines where to place the bead in the image, index value
            pc=self.N*ratio+offset   #Adjusted by offset
            #pc+=pcnoise
            #print(pc)
            #Boundaries of where beads will be distributed, not going to zero or outside of the image
            pc_low=np.maximum(pc-width,np.zeros_like(pc)).astype(int)
            pc_high=np.minimum(pc+width,self.N).astype(int)
            #print(pc_low, pc_high)
            if disp_noise_amp>0:   #Gaussian noise for each dimension if active
                pxnoise=np.random.normal(0,disp_noise_amp)
                pynoise=np.random.normal(0,disp_noise_amp)
                pznoise=np.random.normal(0,disp_noise_amp)
            else:
                pxnoise=0
                pynoise=0
                pznoise=0
            for px0 in range(pc_low[0],pc_high[0]):   #Iterate through neighboring pixels (0:x, 1:y, 2:z)
                px=px0+pxnoise   #Apply noise
                px_coord=(px-offset[0]+0.5)*res[0]   #Center, align and rescale the pixel
                for py0 in range(pc_low[1],pc_high[1]):
                    py=py0+pynoise
                    py_coord=(py-offset[1]+0.5)*res[1]
                    for pz0 in range(pc_low[2],pc_high[2]):
                        pz=pz0+pznoise
                        pz_coord=(pz-offset[2]+0.5)*res[2]
                        vector=[px_coord,py_coord,pz_coord]   #3D vector of pixel
                        dist=self.distance(p,vector)   #Distance from bead position p to pixel's physical position
                        noise=abs(np.random.normal(0,noise_amp))   #Add noise
                        I=self.psf(dist)+noise   #PSF of distance and noise added
                        I0[px0][py0][pz0]+=I
        print('Image data generated')
        return I0

    #Generate a pair of images with beads in reference and in displaced positions
    def generate_image_pair(self,boxsize=1.2,beaddensity=0.001,beaddiameter=0.03,
                            res=np.array([0.015,0.015,0.03]),noise_amp=0,
                            savemat=True, seed=100, variablesonly=False, pixelshift=False,
                            ignorevolume=False,disp_noise_amp=0):
        print('Generating image pair')
        np.random.seed(seed)   #Set RNG to seed=100 for reproducibility
        self.boxsize=boxsize
        self.beaddensity=beaddensity
        print(f'Defining self.res as res={res}')
        self.res=res   #Surface roughness will be applied as noise model
        self.l=2*self.boxsize*self.radius   #Box length in each direction
        self.N=self.l/res   #Dimension of 3D image in terms of pixels ...
        self.N2=2*self.radius/res   # ... and res
        if not np.allclose(np.mod(self.N, 1), 0):
            print('Warning: pixels don\'t exactly match')
            print(self.N)
        self.V=self.N2[0]*self.N2[1]*self.N2[2]*1/6*np.pi   #Effective volume within which beads are distributed
        self.beaddiameter=beaddiameter
        if not variablesonly:
            #Calculate reference and displaced bead positions, and data
            reference_bead_positions=self.generate_bead_positions(ignorevolume=ignorevolume)
            displaced_bead_positions=self.generate_displaced_bead_positions(
                                                    reference_bead_positions,
                                                    pixelshift=pixelshift)
            self.img_ref=self.generate_image_data(reference_bead_positions,noise_amp=noise_amp,disp_noise_amp=disp_noise_amp)
            self.img_disp=self.generate_image_data(displaced_bead_positions,noise_amp=noise_amp,disp_noise_amp=disp_noise_amp)
            
            #Use maximum (3D) intensity between both images to normalize to 16-bit range for saving and analysis
            I0max=max([np.max(self.img_ref),np.max(self.img_disp)])
            self.img_ref_norm=((self.img_ref*(2**16-1)/I0max).astype(np.uint16))
            self.img_disp_norm=((self.img_disp*(2**16-1)/I0max).astype(np.uint16))
            print(self.img_ref_norm.shape)
            
            if savemat:   #Save as matlab file in a dictionary so it appears as "x" in Matlab (both appear as "x"?)
                path=self.fullsavepath
                sp.io.savemat(path+'00Referencepicture_size:'+str(self.N)+self.infostring_all+'.mat',
                                {'x':self.img_ref_norm})
                sp.io.savemat(path+'01Deformedpicture_size:'+str(self.N)+self.infostring_all+'.mat',
                                {'x':self.img_disp_norm})
                print(path)
                print('Deformed and undeformed volumes saved as .mat files')

        print('Image pair generated')

    # --- PSF function ---
    def custom_psf(self, distance, I0_amp=1, decayconstant_rel=1):
        decayconstant = decayconstant_rel*self.beaddiameter
        if distance/self.beaddiameter < 3:
            return abs(I0_amp*np.exp(-0.5*(distance/decayconstant)**4))
        else:
            return 0


    #Then, analyse the data in Matlab and get results labeled accordingly.
    #FIDVC loading pipeline
    def load_prep_FIDVC(self, full_u_true=False, full_u_corr=True,
                        full_u_true_only=False,maskradius=1.7, coordinate_order = 'XYZ'):   #True values (u_r, u_th) and correlated values (from FIDVC) in spherical and cartesian coordinates

        print('Loading and preparing FIDVC data')

        if full_u_true_only:   #Dimensions of mesh grid are calculated
            dimx,dimy,dimz=(1+2*self.boxsize*self.radius/self.res/4).astype('int0')   
            print(dimx,dimy,dimz)
            self.xfactor=1
            self.yfactor=1
            self.zfactor=1

        else:   #Needs the matlab files to get mesh grid and displacement values
            u_mat_path = os.path.join(self.fullsavepath, 'u_profile.mat')
            mesh_mat_path = os.path.join(self.fullsavepath, 'mesh_profile.mat')

            if not (os.path.isfile(u_mat_path) and os.path.isfile(mesh_mat_path)):
                print('WARNING: Missing FIDVC .mat files:', u_mat_path, mesh_mat_path)
                print('Skipping load_prep_FIDVC. Run the MATLAB pipeline to generate these files.')
                return

            X=sio.loadmat(u_mat_path)
            mesh=sio.loadmat(mesh_mat_path)
            print('Loaded mesh and displacement data from path '+self.fullsavepath)
            mesharray = mesh['m'][0]
            u_array0 = X['u'][0][0][0]
            print('u_array0.shape:', u_array0.shape)
            print('Mesharray shape:', mesharray.shape)
            
            print('Type of mesharray[0]:', type(mesharray[0]))
            u_array=np.array([u_array0[1], u_array0[0], u_array0[2]])       # This corrects Matlab's (y, x, z) coordinate ordering to (x, y, z)

            numdims=np.shape(mesharray)
            print(numdims)
            dimx,dimy,dimz=np.shape(mesharray[0])
            print(dimx,dimy,dimz)
            assert numdims[0]==3
            x0=mesharray[0][0,0,0]
            x1=mesharray[0][-1,0,0]
            x_extent=x1-x0
            y0=mesharray[1][0,0,0]
            y1=mesharray[1][0,-1,0]
            y_extent=y1-y0
            z0=mesharray[2][0,0,0]
            z1=mesharray[2][0,0,-1]
            z_extent=z1-z0
            # Calculate spacing factors from mesh extents and dimensions
            self.xfactor=x_extent/dimx if dimx > 0 else 1.0
            self.yfactor=y_extent/dimy if dimy > 0 else 1.0
            self.zfactor=z_extent/dimz if dimz > 0 else 1.0

            print('[DEBUG] FIDVC mesh extents:')
            print(f'  x_extent={x_extent}, y_extent={y_extent}, z_extent={z_extent}')
            print('[DEBUG] FIDVC mesh spacing factors (extent/N):')
            print(f'  xfactor={self.xfactor}, yfactor={self.yfactor}, zfactor={self.zfactor}')
        #assert dimx==dimy

        #Initiate arrays for true and corrected values in spherical and cartesian coordinates
        self.u_true_full_sph=np.zeros((3,dimx,dimy,dimz))
        self.u_corr_full_sph=np.zeros((3,dimx,dimy,dimz))

        self.u_true_full_cart=np.zeros((3,dimx,dimy,dimz))
        self.u_corr_full_cart=np.zeros((3,dimx,dimy,dimz))

        #Initiate coordinate grid to define over which region the displacements are calculated
        l0=self.boxsize*self.radius
        l0x=l0
        l0y=l0
        l0z=l0
        self.l0x=l0
        self.l0y=l0
        self.l0z=l0
        self.coords_x=np.linspace(-l0x,l0x,dimx)
        coords_x=self.coords_x
        self.coords_y=np.linspace(-l0y,l0y,dimy)
        coords_y=self.coords_y
        self.coords_z=np.linspace(-l0z,l0z,dimz)
        coords_z=self.coords_z

        print('[DEBUG] Derived coordinate bounds (code units):')
        print(f'  x: [{coords_x[0]}, {coords_x[-1]}], n={len(coords_x)}')
        print(f'  y: [{coords_y[0]}, {coords_y[-1]}], n={len(coords_y)}')
        print(f'  z: [{coords_z[0]}, {coords_z[-1]}], n={len(coords_z)}')

        if dimx%2==1:
            enumeration_pos=int(dimx/2)

        else:
            enumeration_pos=int(dimx/2)
            print('Warning: Evaluation will not be exact as array has even dimensionality')
        print("Enumeration position: "+str(enumeration_pos))

        #Iterate over all coordinates to compute true (u_r, u_th) and corrected (from u_array) displacement values
        
        if full_u_true:
            for nx,x in enumerate(coords_x):
                for ny,y in enumerate(coords_y):
                    for nz,z in enumerate(coords_z):
                        if (x**2+y**2+z**2)<maskradius*self.radius**2:
                            r,th,p=self.cart_to_sph_coords([x,y,z])
                            u_sph_true=np.array([self.u_r(th,r),self.u_th(th,r),0])
                            u_cart_true= self.vector_sph_to_cart(u_sph_true,th,p)
                            self.u_true_full_sph[:,nx,ny,nz]=u_sph_true
                            self.u_true_full_cart[:,nx,ny,nz]=u_cart_true

                            u_corr_cart=np.array([u_array[0][nx,ny,nz],
                            u_array[1][nx,ny,nz],
                            u_array[2][nx,ny,nz]])*self.res
                            u_corr_sph=self.vector_cart_to_sph(u_corr_cart,th,p)      

                            self.u_corr_full_sph[:,nx,ny,nz]=u_corr_sph
                            self.u_corr_full_cart[:,nx,ny,nz]=u_corr_cart

            if full_u_true_only:    #Assume correlated data = true data in this case
                self.u_corr_full_sph=self.u_true_full_sph
                self.u_corr_full_cart=self.u_true_full_cart

        #But if full_u_true is false, only specific cross sections (in xz or yz planes) are calculated for better runtime
        if not full_u_true:
            ny=enumeration_pos
            y=0   #no y loop, xz plane
            for nx,x in enumerate(coords_x):
                for nz,z in enumerate(coords_z):
                    if (x**2+y**2+z**2)<maskradius*self.radius**2:
                        r,th,p=self.cart_to_sph_coords([x,y,z])
                        u_sph_true=np.array([self.u_r(th,r),self.u_th(th,r),0])
                        u_cart_true= self.vector_sph_to_cart(u_sph_true,th,p)
                        self.u_true_full_sph[:,nx,ny,nz]=u_sph_true
                        self.u_true_full_cart[:,nx,ny,nz]=u_cart_true


                        u_corr_cart=np.array([u_array[0][nx,ny,nz],
                                                u_array[1][nx,ny,nz],
                                                u_array[2][nx,ny,nz]])*self.res
                        u_corr_sph=self.vector_cart_to_sph(u_corr_cart,th,p)

                        self.u_corr_full_sph[:,nx,ny,nz]=u_corr_sph
                        self.u_corr_full_cart[:,nx,ny,nz]=u_corr_cart

            nx=enumeration_pos
            x=0   #no x loop, yz plane
            for ny,y in enumerate(coords_y):
                for nz,z in enumerate(coords_z):
                    if (x**2+y**2+z**2)<maskradius*self.radius**2:
                        r,th,p=self.cart_to_sph_coords([x,y,z])
                        u_sph_true=np.array([self.u_r(th,r),self.u_th(th,r),0])
                        u_cart_true= self.vector_sph_to_cart(u_sph_true,th,p)
                        self.u_true_full_sph[:,nx,ny,nz]=u_sph_true
                        self.u_true_full_cart[:,nx,ny,nz]=u_cart_true

                        u_corr_cart=np.array([u_array[0][nx,ny,nz],
                                                u_array[1][nx,ny,nz],
                                                u_array[2][nx,ny,nz]])*self.res
                        
                        u_corr_sph=self.vector_cart_to_sph(u_corr_cart,th,p)

                        self.u_corr_full_sph[:,nx,ny,nz]=u_corr_sph
                        self.u_corr_full_cart[:,nx,ny,nz]=u_corr_cart


        if full_u_corr:   #Loads all volume correlation displacement values
                            #This will overwrite u_corr_full_cart from the last block. u_corr_full_sph will stay but it's not needed for 3d traction plots?
            for n in range(3):
                X=sio.loadmat(self.fullsavepath+'u_profile.mat')
                mesh=sio.loadmat(self.fullsavepath+"mesh_profile.mat")
                mesharray=mesh['m'][0]
                u_array0=X['u'][0][0][0]
                u_array=np.array([u_array0[1], u_array0[0], u_array0[2]])

                self.u_corr_full_cart[n]=np.array(u_array[n])*self.res[n]
                
        #Calculate error fields by subtracting volume correlated from true displacement values
        self.u_err_sph=self.u_corr_full_sph - self.u_true_full_sph
        self.u_err_cart=self.u_corr_full_cart - self.u_true_full_cart
        self.enumeration_pos=enumeration_pos   #Enumeration position (midpoint index of dimx dimension) is saved
        print('Loading finished')

    #Plot traction forces, using VC or MF
    def plot_T_3D(self, Textsize=25, customminmax=False, numplots=[0, 1, 2],
                            samescale='all', custommax=[1],
                            coord='sph', remove=False, zmax=1.4,
                            method_='VC', view='side', xextra=None, plotdpi=100, pad=7,
                            show_analytic=False, show_deviation=False, colorbar_mode='row'):
        '''
        Plots only the experimental traction data (first column) in 3D.

        Parameters
        ----------
        method_: 'VC' or 'MF' for Volume Correlation or Minimizing Functional
        coord: 'sph' or 'cart'
        '''

        if method_ == 'VC':
            self.T_sph = self.T_corr_full_sph
            self.T_cart = self.T_corr_full_cart
        elif method_ == 'MF':
            self.T_sph = self.T_MF_full_sph
            self.T_cart = self.T_MF_full_cart


        # Determine traction arrays in physical Pascals for plotting.
        # If caller set `self._plot_in_Pa = True`, the stored T arrays are
        # already in Pascals; otherwise assume stored arrays are in code
        # units and convert using the object's shear modulus `self.mu`.
        if coord == 'sph':
            raw_T = self.T_sph
        elif coord == 'cart':
            raw_T = self.T_cart

        try:
            if getattr(self, '_plot_in_Pa', False):
                T_pa = raw_T
            else:
                # convert code units -> Pa by multiplying with shear modulus
                T_pa = raw_T * float(getattr(self, 'mu', 1.0))
        except Exception:
            T_pa = raw_T

        # Use Pa values directly for plotting (do not divide by norm_radius)
        u_data = - T_pa

        # Build columns to plot: experimental, optional analytic, optional deviation
        plot_columns = [u_data]
        if 'show_analytic' in locals() and show_analytic:  # backwards-compatible check
            # NOTE: analytic tractions should be negated to match plotting sign convention
            if coord == 'sph':
                u_ana = - self.T_true_full_sph
            else:
                u_ana = - self.T_true_full_cart
            plot_columns.append(u_ana)
        elif 'show_analytic' not in locals() and False:
            # placeholder to keep linter happy
            pass

        # If caller passed show_deviation in kwargs, it will be present in locals()
        if 'show_deviation' in locals() and show_deviation:
            if 'u_ana' not in locals():
                if coord == 'sph':
                    u_ana = self.T_true_full_sph
                else:
                    u_ana = self.T_true_full_cart
            u_dev = u_data - u_ana
            plot_columns.append(u_dev)

        all_u_data = np.array(plot_columns)  # shape (ncols, ncomp, lat, lon)
        ncols = all_u_data.shape[0]
        ncomp = all_u_data.shape[1]
        ny = len(numplots)

        if xextra is None:
            xextra = 2 if ny == 3 else 3

        scale = 2
        sizex = scale * (2 * ncols + xextra)
        sizey = scale * (2 * ny + 1)

        # Determine per-column/component color scaling
        min_vals = np.nanmin(all_u_data, axis=(-1, -2))
        max_vals = np.nanmax(all_u_data, axis=(-1, -2))
        vals = np.maximum(np.abs(min_vals), np.abs(max_vals))  # shape (ncols, ncomp)

        # Normalize vals selection modes to per-column/component matrix
        # vals already shape (ncols, ncomp)
        if samescale == 'all':
            vals = np.ones_like(vals) * np.nanmax(vals)
        elif samescale == 'allcustom':
            vals = np.ones_like(vals) * custommax[0]
        elif samescale == 'row':
            # same scale across columns for each component
            max_per_comp = np.nanmax(vals, axis=0)
            vals = np.tile(max_per_comp, (ncols, 1))
        elif samescale == 'rowcustom':
            vals = np.tile(np.array(custommax), (ncols, 1))
        elif samescale == 'no':
            # per-column/component independent scaling
            for c in range(ncols):
                for comp in range(ncomp):
                    if vals[c, comp] == 0:
                        vals[c, comp] = np.nanmax(vals[c, :])

        # DEBUG: print computed color scaling ranges so caller can verify
        try:
            print(f"DEBUG: plot_T_3D samescale={samescale}, method={method_ if 'method_' in locals() else 'n/a'}")
            #print('DEBUG: min_vals per column/component =', min_vals)
            #print('DEBUG: max_vals per column/component =', max_vals)
            #print('DEBUG: vals (abs maxima) used for color scaling =', vals)
        except Exception:
            pass

        fig, ax = plt.subplots(ny, ncols, figsize=(sizex, sizey), subplot_kw=dict(projection='3d'))
        # Ensure ax is a 2D array of shape (ny, ncols)
        ax = np.array(ax)
        try:
            ax = ax.reshape(ny, ncols)
        except Exception:
            # Fallback to safer reshaping for odd shapes
            ax = np.atleast_2d(ax)
            ax = ax.reshape((ny, ncols))

        # Fontsize for per-component titles (make consistent with surface-method labels)
        comp_label_fontsize = int(Textsize * 0.9)

        # Loop over columns (different fields) and rows (components chosen by numplots)
        # Column labels for reconstructed / analytic / difference
        if method_=='VC':
            column_labels = ['Volume method reconstruction', 'Analytic', 'Difference']
        elif method_=='MF':
            column_labels = ['Surface method reconstruction', 'Analytic', 'Difference']
            
        for n, u in enumerate(all_u_data):
            for m, mm in enumerate(numplots):
                ax_ = ax[m, n]
                ax_.tick_params(axis="x", labelsize=int(Textsize * 0.55))
                ax_.tick_params(axis="y", labelsize=int(Textsize * 0.55))
                ax_.tick_params(axis="z", labelsize=int(Textsize * 0.55))

                min_val = -vals[n, mm]
                max_val = vals[n, mm]
                #print(f"Component {mm}, vmin={min_val}, vmax={max_val}")

                Tvec_mesh = u[mm]
                x, y, z, colors, cmap, norm = visSH3D_data3(Tvec_mesh, vmin=min_val, vmax=max_val)

                # (No automatic rolling/centering) Rely on Matplotlib's bottom
                # view orientation and the explicit coordinate swap (z,y,x)
                # to present the bottom projection. Rolling the arrays here
                # can misalign colors/mesh when shapes or axis order differ
                # and was producing a missing slice in some cases, so it is
                # intentionally omitted.

                # Create the surface inside the view-specific branches so that
                # the bottom view can use the (z,y,x) ordering explicitly and
                # we don't overwrite it later. Keep facecolors so colormap
                # mapping remains consistent; let Matplotlib handle shading.
                if view == 'sidexz':
                    ax_.view_init(elev=0, azim=90)
                    surf = ax_.plot_surface(x, y, z, rstride=1, cstride=1, facecolors=colors)
                    ax_.set_xlabel('x/$r_0$', labelpad=pad, fontsize=int(Textsize*0.65))
                    ax_.set_zlabel('z/$r_0$', labelpad=pad, fontsize=int(Textsize*0.65))
                    ax_.yaxis.set_ticklabels([])
                elif view == 'sideyz':
                    ax_.view_init(elev=0, azim=0)
                    surf = ax_.plot_surface(x, y, z, rstride=1, cstride=1, facecolors=colors,)
                    ax_.set_ylabel('y/$r_0$', labelpad=pad, fontsize=int(Textsize*0.65))
                    ax_.set_zlabel('z/$r_0$', labelpad=pad, fontsize=int(Textsize*0.65))
                    ax_.xaxis.set_ticklabels([])
                elif view == 'bottom':
                    # Explicit coordinate swap for bottom view to match original behavior
                    ax_.view_init(elev=0, azim=180)
                    surf = ax_.plot_surface(z, y, x, rstride=1, cstride=1, facecolors=colors)
                    ax_.set_zlabel('x/$r_0$', labelpad=pad, fontsize=int(Textsize*0.65))
                    ax_.set_ylabel('y/$r_0$', labelpad=pad, fontsize=int(Textsize*0.65))
                    ax_.xaxis.set_ticklabels([])
                else:
                    surf = ax_.plot_surface(x, y, z, rstride=1, cstride=1, facecolors=colors, linewidth=0, antialiased=False)

                # Defer colorbar creation until after all surfaces are plotted
                if remove:
                    ax_.axis('off')
                ax_.xaxis.set_major_locator(tck.MultipleLocator(base=0.5))
                ax_.yaxis.set_major_locator(tck.MultipleLocator(base=0.5))
                ax_.zaxis.set_major_locator(tck.MultipleLocator(base=0.5))
                ax_.set_box_aspect([1, 1, 1])

                if coord == 'cart':
                    titles = ['$T_x$ [Pa]', '$T_y$ [Pa]', '$T_z$ [Pa]']
                else:
                    titles = [r'$T_r$ [Pa]', r'$T_{\theta}$ [Pa]', r'$T_{\varphi}$ [Pa]']
                ax_.set_title(titles[mm], fontsize=comp_label_fontsize)

                # add column header labels (only once, above the top-row axes)
                if m == 0 and n < len(column_labels):
                    pos = ax_.get_position()
                    # place label slightly higher so it sits directly above the column
                    fig.text((pos.x0 + pos.x1) / 2, pos.y1 + 0.07, column_labels[n], ha='center', va='bottom', fontsize=int(Textsize * 0.9))


            if view == 'sidexz':
                ax_.view_init(elev=0, azim=90)
                surf = ax_.plot_surface(x, y, z, rstride=1, cstride=1, facecolors=colors, shade=False, linewidth=0, antialiased=False)
                ax_.set_xlabel('x/$r_0$', labelpad=pad, fontsize=int(Textsize*0.65))
                ax_.set_zlabel('z/$r_0$', labelpad=pad, fontsize=int(Textsize*0.65))
                ax_.yaxis.set_ticklabels([])
            elif view == 'sideyz':
                ax_.view_init(elev=0, azim=0)
                surf = ax_.plot_surface(x, y, z, rstride=1, cstride=1, facecolors=colors, shade=False, linewidth=0, antialiased=False)
                ax_.set_ylabel('y/$r_0$', labelpad=pad, fontsize=int(Textsize*0.65))
                ax_.set_zlabel('z/$r_0$', labelpad=pad, fontsize=int(Textsize*0.65))
                ax_.xaxis.set_ticklabels([])
            elif view == 'bottom':
                ax_.view_init(elev=-90, azim=0)
                surf = ax_.plot_surface(z, y, x, rstride=1, cstride=1, facecolors=colors, shade=False, linewidth=0, antialiased=False)
                # For bottom view show x (horizontal) and y (vertical) labels only
                ax_.set_xlabel('x/$r_0$', labelpad=pad, fontsize=int(Textsize*0.65))
                ax_.set_ylabel('y/$r_0$', labelpad=pad, fontsize=int(Textsize*0.65))
                ax_.set_zlabel('')
                ax_.xaxis.set_ticklabels([])


            if remove:
                ax_.axis('off')

            ax_.xaxis.set_major_locator(tck.MultipleLocator(base=0.5))
            ax_.yaxis.set_major_locator(tck.MultipleLocator(base=0.5))
            ax_.zaxis.set_major_locator(tck.MultipleLocator(base=0.5))
            ax_.set_box_aspect([1, 1, 1])
            set_axes_equal(ax_)

        # --- Create colorbars according to samescale ---
        # Prepare a default colormap base
        try:
            cmap_base = plt.cm.get_cmap('coolwarm')
        except Exception:
            cmap_base = cm.coolwarm

        # give space for headers and avoid cramped axes
        try:
            #fig.subplots_adjust(top=0.88) # original
            fig.subplots_adjust(left=0.1, right=0.90, top=0.88, bottom=0.05, hspace=0.1) # to align experimental volume+surface plots
        except Exception:
            pass

        # If samescale=='no' we provide per-axis colorbars and allow each
        # subplot to use its own colormap. This gives each component/column
        # independent scaling and visual palette as requested.
        if samescale == 'no':
            # Use the same colormap (`coolwarm`) for all subplots, but with independent vmin/vmax per subplot.
            try:
                cmap_single = plt.cm.get_cmap('coolwarm')
            except Exception:
                cmap_single = cmap_base
                
            for m_ in range(ny):
                for n_ in range(ncols):
                    try:
                        ax_i = ax[m_, n_]
                        comp = numplots[m_]
                        # fall back if vals entry is NaN or zero
                        colmax = vals[n_, comp]
                        if np.isnan(colmax) or colmax == 0:
                            colmax = np.nanmax(vals)
                        norm_cb = mpl.colors.Normalize(vmin=-colmax, vmax=colmax)
                        mappable = mpl.cm.ScalarMappable(norm=norm_cb, cmap=cmap_single)
                        mappable.set_array([])
                        # attach one colorbar per-axis
                        fig.colorbar(mappable, ax=ax_i, shrink=0.6)
                    except Exception:
                        pass

        else:
            
            if colorbar_mode == 'column':
                for n_ in range(ncols):
                    colmax = np.nanmax(vals[n_, :])
                    norm_cb = mpl.colors.Normalize(vmin=-colmax, vmax=colmax)
                    mappable = mpl.cm.ScalarMappable(norm=norm_cb, cmap=cmap_base)
                    mappable.set_array([])
                    fig.colorbar(mappable, ax=[ax[row, n_] for row in range(ny)], shrink=0.6)
            elif colorbar_mode == 'row':
                for m_ in range(ny):
                    rowmax = np.nanmax(vals[:, m_])
                    norm_cb = mpl.colors.Normalize(vmin=-rowmax, vmax=rowmax)
                    mappable = mpl.cm.ScalarMappable(norm=norm_cb, cmap=cmap_base)
                    mappable.set_array([])
                    fig.colorbar(mappable, ax=[ax[m_, col] for col in range(ncols)], shrink=0.6)
            elif colorbar_mode == 'all':
                globalmax = np.nanmax(vals)
                norm_cb = mpl.colors.Normalize(vmin=-globalmax, vmax=globalmax)
                mappable = mpl.cm.ScalarMappable(norm=norm_cb, cmap=cmap_base)
                mappable.set_array([])
                fig.colorbar(mappable, ax=ax.flatten().tolist(), shrink=0.6)

        # Ensure axis labels are consistent for the bottom view across all axes
        # For bottom view we use the explicit coordinate ordering (z, y, x) so set labels accordingly:
        # x-axis -> z, y-axis -> y, z-axis -> x.
        if view == 'bottom':
            try:
                # Clear any ambiguous 3D axis labels (they can point to the
                # wrong physical axes after rotation). We'll add clear 2D
                # labels below positioned relative to each subplot instead.
                for a in np.array(ax).flatten():
                    try:
                        a.set_xlabel('')
                        a.set_ylabel('')
                        a.set_zlabel('')
                    except Exception:
                        pass
                # Remove old small-text labels near the bottom so our new
                # annotations don't overlap; we'll re-add controlled labels.
            except Exception:
                pass
            # Remove any stray figure-level text labels that landed near the
            # bottom-right of the figure (likely repeated headers); keep only
            # those near the top (y>0.3).
            try:
                new_texts = []
                for t in fig.texts:
                    try:
                        xpos, ypos = t.get_position()
                        if ypos > 0.3:
                            new_texts.append(t)
                        else:
                            # remove from figure
                            t.remove()
                    except Exception:
                        # if we can't query position, preserve it
                        new_texts.append(t)
                fig.texts = new_texts
            except Exception:
                pass

            # Ensure every bottom-view subplot shows the two in-plane labels
            try:
                ax_arr = np.array(ax).reshape(ny, ncols)
                for a in ax_arr.flatten():
                    try:
                        # Horizontal (side) label -> y, Vertical (up) label -> x
                        a.set_zlabel('x/$r_0$', labelpad=pad, fontsize=int(Textsize*0.65))
                        a.set_ylabel('y/$r_0$', labelpad=pad, fontsize=int(Textsize*0.65))
                        # hide the out-of-plane label
                        a.set_xlabel('')
                        try:
                            a.zaxis.label.set_visible(True)
                            a.yaxis.label.set_visible(True)
                        except Exception:
                            pass
                    except Exception:
                        pass
            except Exception:
                pass
            # Force a consistent camera orientation for all bottom-view axes
            try:
                for a in np.array(ax).flatten():
                    try:
                        a.view_init(elev=0, azim=180)
                    except Exception:
                        pass
            except Exception:
                pass

            # Clear tick labels for the bottom-right subplot to avoid stacked/overlaid
            # tick text artifacts that sometimes appear due to tight layout.
            try:
                ax_arr = np.array(ax).reshape(ny, ncols)
                br = ax_arr[-1, -1]
                # keep numeric tick labels visible on the bottom-right subplot
                try:
                    # restore default tick formatting by resetting locators
                    br.xaxis.set_major_locator(tck.MultipleLocator(base=0.5))
                    br.yaxis.set_major_locator(tck.MultipleLocator(base=0.5))
                    br.zaxis.set_major_locator(tck.MultipleLocator(base=0.5))
                except Exception:
                    pass
            except Exception:
                pass

        filename_base = f"T_plot_{view}_{coord}{method_}{self.infostring_all}_Mag={self.magnitude}_EXP"
        fig.savefig(self.fullsavepath + filename_base + ".png", format='png')
        print('Saved 3D experimental traction plot to ' + self.fullsavepath + filename_base + ".png")

 

    # Get plot data for T
    def plot_T_spheresurface(self,Textsize=20,phi_plot=False, customminmax=False,
                                samescale=True,samescale_symmetric=True, save=True, show=True,
                                method_='VC', coord='sph',numplots=[0,1,2]):
        '''
        Parameters
        ----------
        method: 'VC' or 'MF' for Volume Correlation or Minimizing Functional

        coord: 'sph' oder 'sph+phi' or 'cart'

        '''
        if method_=='VC':
            self.T_sph=self.T_corr_full_sph   #Functions defined in T_spheresurface_VC
            self.T_cart=self.T_corr_full_cart
        if method_=='MF':
            self.T_sph=self.T_MF_full_sph
            self.T_cart=self.T_MF_full_cart
        self.T_err_sph=self.T_sph-self.T_true_full_sph
        self.T_err_cart=self.T_cart-self.T_true_full_cart

        if coord=='sph':
            all_u_data=[self.T_sph,self.T_true_full_sph,self.T_err_sph]
            titlestr='spherical'

        if coord=='cart':
            all_u_data=[self.T_cart,self.T_true_full_cart,self.T_err_cart]
            titlestr='cartesian'
        Textsize=25
        plt.rcParams['axes.titlesize']=Textsize
        plt.rcParams['axes.labelsize']=Textsize*0.7
        nx=3
        ny=len(numplots)
        scale=2
        sizex=scale*(2*nx+2)
        sizey=scale*(1*ny+2)
        plotextent=[0,2,1,0]
        if customminmax!=False:
            min_val=-customminmax
            max_val=customminmax
        min_val, max_val = np.nanmin(all_u_data), np.nanmax(all_u_data)
        val=max([abs(min_val),abs(max_val)])
        min_val=-val
        max_val=+val
        fig, ax = plt.subplots(ny, nx,figsize=(sizex,sizey))
        fig.suptitle('Traction components (%s coordinates) on sphere surface' % titlestr, fontsize=int(Textsize))
        fig.add_subplot(111, frameon=False)
        # hide tick and tick label of the big axes
        plt.tick_params(labelcolor='none', top=False, bottom=False, left=False, right=False)
        plt.grid(False)
        plt.xlabel(r'Latitude $\theta$')
        plt.ylabel(r'Longitude $\varphi$')
        for n, u in enumerate(all_u_data):
            for m in numplots:
                if samescale:
                    ax[m,n].imshow(u[m],extent=plotextent,
                                vmin=min_val,vmax=max_val)
                else:
                    img=ax[m,n].imshow(u[m], extent=plotextent)
                    plt.colorbar(img, ax=ax[m,n])
                if n==0:
                    ax[m,n].yaxis.set_major_formatter(tck.FormatStrFormatter('%g $\\pi$'))
                    ax[m,n].yaxis.set_major_locator(tck.MultipleLocator(base=0.5))
                    ax[m,n].yaxis.set_minor_locator(AutoMinorLocator(2))
                else:
                    ax[m,n].yaxis.set_major_formatter(tck.NullFormatter())
                    ax[m,n].yaxis.set_major_locator(tck.MultipleLocator(base=0.5))
                    ax[m,n].yaxis.set_minor_locator(AutoMinorLocator(2))
                if m==ny-1:
                    ax[m,n].xaxis.set_major_formatter(tck.FormatStrFormatter('%g $\\pi$'))
                    ax[m,n].xaxis.set_major_locator(tck.MultipleLocator(base=1))
                    ax[m,n].xaxis.set_minor_locator(AutoMinorLocator(2))
                else:
                    ax[m,n].xaxis.set_major_formatter(tck.NullFormatter())
                    ax[m,n].xaxis.set_major_locator(tck.MultipleLocator(base=1))
                    ax[m,n].xaxis.set_minor_locator(AutoMinorLocator(2))
                ax[m,n].tick_params(axis='both', which='major', labelsize=Textsize*0.5)
                ax[m,n].tick_params(axis='both', which='major', labelsize=Textsize*0.5)



        if coord=='sph' or coord=='sph+phi':
            ax[0,0].set_title(r'$T_{r,%s}$' % method_)
            ax[0,1].set_title(r'$T_{r,True}$')
            ax[0,2].set_title(r'$T_{r,%s}-T_{r,True}$ (Error)' % method_)
            ax[1,0].set_title(r'$T_{\theta,%s}$' % method_)
            ax[1,1].set_title(r'$T_{\theta,True}$')
            ax[1,2].set_title(r'$T_{\theta,%s}-T_{\theta,True}$ (Error)' % method_)
            if coord=='sph+phi':
                ax[2,0].set_title(r'$T_{\varphi,%s}$' % method_)
                ax[2,1].set_title(r'$T_{\varphi,True}$')
                ax[2,2].set_title(r'$T_{\varphi,%s}-T_{\varphi,True}$ (Error)' % method_)

        if coord=='cart':
            ax[0,0].set_title(r'$T_{x,%s}$' % method_)
            ax[0,1].set_title(r'$T_{x,True}$')
            ax[0,2].set_title(r'$T_{x,%s}-T_{x,True}$ (Error)' % method_)

            ax[1,0].set_title(r'$T_{y,%s}$' % method_)
            ax[1,1].set_title(r'$T_{y,True}$')
            ax[1,2].set_title(r'$T_{y,%s}-T_{y,True}$ (Error)' % method_)


            ax[2,0].set_title(r'$T_{z,%s}$' % method_)
            ax[2,1].set_title(r'$T_{z,True}$')
            ax[2,2].set_title(r'$T_{z,%s}-T_{z,True}$ (Error)' % method_)

        if samescale:
            fig.tight_layout()
            fig.canvas.draw()
            fig.subplots_adjust(right=0.8)
            cbar_ax = fig.add_axes([0.85, 0.15, 0.05, 0.7])
            norm = mpl.colors.Normalize(vmin=min_val, vmax=max_val)
            scm=mpl.cm.ScalarMappable(norm=norm)
            fig.colorbar(scm,cax=cbar_ax)

        if show:
            fig.show
        if save:
            filename="T_plot_"+coord+method_+self.infostring_all+".svg"
            fig.savefig(filename,format='svg')


    #Calculate all the error metrics in 3D from the 2D functions for them, save as .csv
    def calc_save_errors_VC(self,hist=False,bottom=1e-4,top=1e5,numbins=50,scale=2,
                            printfull=False,printimportant=True,comment='',printlengths=False):
        self.Errors_VC={}
        enumeration_pos=self.enumeration_pos
        datasets=[[self.T_true_full_sph,
                    self.T_corr_full_sph],
                    [self.T_true_full_cart,
                    self.T_corr_full_cart],
                  [self.u_true_full_sph[:,enumeration_pos],
                   self.u_corr_full_sph[:,enumeration_pos]],
                  [self.u_true_full_cart[:,enumeration_pos],
                   self.u_corr_full_cart[:,enumeration_pos]],
                  [self.u_true_full_sph[:,:,enumeration_pos],
                   self.u_corr_full_sph[:,:,enumeration_pos]],
                  [self.u_true_full_cart[:,:,enumeration_pos],
                   self.u_corr_full_cart[:,:,enumeration_pos]]]

        dataset_strings=['T_sph','T_cart','u_sph_yz','u_cart_yz',
                         'u_sph_xz','u_cart_xz']    #According dataset labels
        num=len(datasets)
        self.Errors_VC['Name']=self.savefoldername+self.infostring_all
        self.Errors_VC['Date']=self.datestring
        self.Errors_VC['Errortype']=['RMSE','MAE','MBE','NRMSE','NMAE','NMBE'
                                            ,'AAD0','AAD_est',
                                             'NRMSE2','NMAE2','NMBE2','AAD2']   #Error type labels

        if printlengths:
            print('length Errortype', len(self.Errors_VC['Errortype']))
        #['MAE','MAE_rel','RMSD','NRMSD','mean_abs']
        if hist:
            nx=3
            ny=num
            sizex=scale*(2*nx+2)
            sizey=scale*(2*ny+2)
            fig,ax=plt.subplots(num,3,figsize=(sizex,sizey))

            fig.suptitle('Relative Errors (x,y,z/r,t,p in column 1,2,3) for quantities:')




        for n in range(num):
            d=datasets[n]
            dstring=dataset_strings[n]
            if printimportant and n<2:
                print()
                print('dstring')
            AAD2_=[]
            for nn in range(3):
                AAD2_.append(AAD_surface(d[0][nn]))
            AAD2=self.distance(AAD2_,[0,0,0])
            for nn,xi in enumerate(['_1','_2','_3']):
                if n<2:
                    err=all_errors_2D(d[0][nn],d[1][nn],AAD2=AAD2)
                else:
                    err=all_errors_2D(d[0][nn],d[1][nn],
                                      boxsize=self.boxsize, radius=self.radius,AAD2=AAD2)
                if hist:
                    histdata=err[-1].flatten()
                    err=err[:-1]
                    ax[n][nn].hist(np.abs(histdata),bins=np.logspace(np.log10(bottom),
                                                               np.log10(top),numbins))
                    if nn==1:
                        ax[n][nn].set_title(dstring)
                    ax[n][nn].set_xscale('log')
                if printlengths:
                    print('length',dataset_strings[n],xi,':', len(err))
                self.Errors_VC[dstring+xi]=err
                if printimportant and n<2:
                    print('Component',xi)
                    print('NRMSE: ',err[3])
                    print('NMAE: ',err[4])
                    print('NMBE: ',err[5])
                    print('AAD_true: ',err[6])
                    print('AAD_est: ', err[7])
                    print('AAD2: ', err[11])
                    print('NRMSE2: ',err[8])
                    print('NMAE2: ',err[9])
                    print('NMBE2: ',err[10])
                    print()

        self.Errors_VC['comment']=comment
        df1=pd.DataFrame(self.Errors_VC)
        df1.to_csv(self.savepath+'FIDVC_Results.csv', mode='a', index = False,sep=';' ,header=None)
        if printfull:
            print(df1)


    def calc_save_errors_MF(self,hist=True,bottom=1e-4,top=1e5,numbins=50,scale=2,
                            printfull=True,comment=''):
        self.Errors_MF={}
        datasets=[[self.T_true_full_sph,
                    self.T_MF_full_sph],
                    [self.T_true_full_cart,
                    self.T_MF_full_cart]]

        dataset_strings=['T_sph','T_cart']
        num=len(datasets)
        self.Errors_MF['Name']=self.savefoldername+self.infostring_all
        self.Errors_MF['Date']=self.datestring
        stringcollec=['RMSE','MAE','MBE','NRMSE','NMAE','NMBE','AAD','AAD_est','NRMSE2','NMAE2','NMBE2' ,'AAD2']
        self.Errors_MF['Errortype']=stringcollec
        if hist:
            nx=3
            ny=num
            sizex=scale*(2*nx+2)
            sizey=scale*(2*ny+2)
            fig,ax=plt.subplots(num,3,figsize=(sizex,sizey))
            fig.suptitle('Relative Errors  for quantities: (x,y,z in row 1, column 1,2,3) (r,theta,phi in row 2, column 1,2,3)')

        for n in range(num):
            d=datasets[n]
            dstring=dataset_strings[n]
            AAD2_=[]
            for nn in range(3):
                AAD2_.append(AAD_surface(d[0][nn]))
            AAD2=self.distance(AAD2_,[0,0,0])
            if n==1:
                components=['_x','_y','_z']
            if n==0:
                components=['_r','_theta','_phi']
            for nn,xi in enumerate(components):
                err=all_errors_2D(d[0][nn],d[1][nn], hist=hist,AAD2=AAD2)
                if hist:
                    histdata=err[-1].flatten()
                    err=err[:-1]
                    ax[n][nn].hist(np.abs(histdata),bins=np.logspace(np.log10(bottom),
                                                               np.log10(top),numbins))
                    if nn==1:
                        ax[n][nn].set_title(dstring)
                    ax[n][nn].set_xscale('log')
                assert len(stringcollec)==len(err)
                self.Errors_MF[dstring+xi]=err
        self.Errors_MF['comment']=comment
        df1=pd.DataFrame(self.Errors_MF)
        df1.to_csv(self.savepath+'MF_Results.csv', mode='a', index = False,sep=';' ,header=None)
        if printfull:
            print(df1)

    #Traction recovery part FIDVC
    def get_T(self,lmax=30, real_u=False):
        self.load_prep_FIDVC(full_u_true=real_u)

    def T_spheresurface_VC(self,lmax,sigma=None,returnsigmafuncs=False,
                           eval_r=1):
        
        if sigma==None:   #If not given (standard option), use sigma_corr
            sigma=self.sigma_corr
            
        dimx,dimy,dimz=np.shape(sigma[0])   #Grid dimensions determined by sigma
        l0=self.boxsize*self.radius
        coords_x=np.linspace(-l0,l0,dimx)   #Cartesian coordinates
        coords_y=np.linspace(-l0,l0,dimy)
        coords_z=np.linspace(-l0,l0,dimz)
        latsdeg, lonsdeg = _psh.expand.GLQGridCoord(lmax)
        latsdeg=90-latsdeg

        self.latsdeg, self.lonsdeg = latsdeg, lonsdeg

        sigma_sphere=np.zeros((len(latsdeg),len(lonsdeg),6))
        self.T_corr_full_sph=np.zeros((3,len(latsdeg),len(lonsdeg)))
        self.T_corr_full_cart=np.zeros((3,len(latsdeg),len(lonsdeg)))
        sigmafuncs=[]
        #print(sigma)
        #print(enumerate(sigma))
        for n,s in enumerate(sigma):   #Interpolate sigma from grid onto spherical coordinates (onto sphere for T)
            # Since sigma=self.sigma_corr=[sgxx, sgyy, sgzz, sgyz, sgxz, sgxy],
            # n goes from 0 to 5 while s has the stress components.

            sigma_interp = RegularGridInterpolator((coords_x, coords_y, coords_z), s)  
            # RegularGridInterpolator sets up the interpolator object with appropriate coordinates for each stress component
            # Coordinates define the regular grid in 3 dimensions, while s is the data on the grid
            # This object is used later for interpolation across 3D cartesian grid

            print(n, sigma_interp)

            for np_,phi in enumerate(lonsdeg):   #For each combination of theta and phi, create corresponding cartesian coordinates and use interpolator to gain sigma on sphere
                p=np.deg2rad(phi)
                for nt,th in enumerate(latsdeg):
                    th=np.deg2rad(th)
                    x,y,z=SphCoord_to_CartCoord(self.radius*eval_r,th,p)
                    sigma_sphere[nt,np_,n]=sigma_interp((x,y,z))
            sigmafuncs.append(sigma_interp)

        #Compute traction from (interpolated) sigmas:
        for nt,th in enumerate(latsdeg):   #Go through each combination of theta and phi
            th=np.deg2rad(th)
            for np_,phi in enumerate(lonsdeg):
                p=np.deg2rad(phi)
                x,y,z=self.SphCoord_to_CartCoord(self.radius*eval_r,th,p)   #Convert coordinates at eval_r
                sgxx, sgyy, sgzz, sgyz, sgxz, sgxy=sigma_sphere[nt,np_]   #Extract components of sigma
                sigmat=np.array(([sgxx,sgxy,sgxz],[sgxy,sgyy,sgyz],[sgxz,sgyz,sgzz]))   #3x3 matrix with all components
                normalvec=np.array([x,y,z])/(self.radius*eval_r)   #Normal vector at this point is (x,y,z)/r at eval_r
                T_corr_cart=sigmat.dot(normalvec)   #Equation (8)
                T_corr_sph=self.vector_cart_to_sph(T_corr_cart,th,p) #Convert coordinates, write into arrays
                self.T_corr_full_sph[:,nt,np_]=T_corr_sph
                self.T_corr_full_cart[:,nt,np_]=T_corr_cart

        if returnsigmafuncs:   #Can return interpolator functions
            return sigmafuncs

    #Gain traction from sigma_zz
    #This calculates T_cart_true and T_sph_true in different ways, depending on sigma_zz_bool.
    #If True, (cartesian) stress is assumed to be (0,0,sigma_zz), so traction in spherical coordinates can be calculated by converting coordinates.
    #If False, (spherical) stress is assumed to be (sigma_rr,sigma_rt,0) and vice versa.
    #These are then used to calculate T_true_full_sph and T_true_full_cart by iterating over theta and phi.
    #They are used for the 3D traction plots.

    def T_spheresurface_true(self, lmax):
        latsdeg, lonsdeg = _psh.expand.GLQGridCoord(lmax)
        latsdeg = 90 - latsdeg
        # store grid for downstream helpers
        self.latsdeg, self.lonsdeg = latsdeg, lonsdeg
        self.T_true_full_sph=np.zeros((3,len(latsdeg),len(lonsdeg)))
        self.T_true_full_cart=np.zeros((3,len(latsdeg),len(lonsdeg)))

        for nt,th in enumerate(latsdeg):   #Gain traction from sigma_zz (analytically) for each theta, phi
            th2=np.deg2rad(th)   #convert theta to radians
            for np_,phi in enumerate(lonsdeg):
                p=np.deg2rad(phi)
                if self.sigma_zz_bool:   #Calculate T_sph_true from T_cart_true by assuming the latter to be (0,0,sigma_zz)
                    T_cart_true= np.array([0,0,self.sigma_zz(th2)])
                    T_sph_true=self.vector_cart_to_sph(T_cart_true,th2,p)
                if not self.sigma_zz_bool:   #Calculate T_cart_true from T_sph_true by assuming the latter to be (sigma_rr,sigma_rt,0)
                    T_sph_true=np.array([self.sigma_rr_true(th2),self.sigma_rt_true(th2),0])
                    T_cart_true=self.vector_sph_to_cart(T_sph_true,th2,p)

                self.T_true_full_sph[:,nt,np_]=T_sph_true
                self.T_true_full_cart[:,nt,np_]=T_cart_true

 
    def T_spheresurface_MF(self,Tvec,lmax):   #Minimizing functional
        #Set up grid
        latsdeg, lonsdeg = _psh.expand.GLQGridCoord(lmax)
        latsdeg=90-latsdeg
        self.latsdeg, self.lonsdeg = latsdeg, lonsdeg
        #Convert traction vectors from spherical harmonics coefficients to spherical and cartesian coordinates
        self.T_MF_full_sph=SHVec2mesh(Tvec*self.mu,lmax=lmax, SphCoord=True, Complex=True)[:,:-1].swapaxes(0,2).swapaxes(1,2)
        self.T_MF_full_cart=SHVec2mesh(Tvec*self.mu,lmax=lmax, SphCoord=False, Complex=True)[:,:-1].swapaxes(0,2).swapaxes(1,2)
        #Shape consistency check (three components for each grid point)
        assert (3,len(latsdeg),len(lonsdeg))==self.T_MF_full_sph.shape

    def create_sigma_functions(self):   #Set up interpolation functions for stress tensor components by interpolating over the spatial grid derived from computated derivatives
        sgxx, sgyy, sgzz, sgyz, sgxz, sgxy=self.calc_all_derivatives(kernel='5tap',return_=True)   #Calculate partial derivatives of displacement data to get stress tensor components
        #Set up grid
        dimx,dimy,dimz=np.shape(sgxx)
        l0=self.boxsize*self.radius
        coords_x=np.linspace(-l0,l0,dimx)
        coords_y=np.linspace(-l0,l0,dimy)
        coords_z=np.linspace(-l0,l0,dimz)
        #Interpolate each stress tensor component over this 3d grid    
        self.sgxx=RegularGridInterpolator((coords_x, coords_y, coords_z), sgxx)
        self.sgyy=RegularGridInterpolator((coords_x, coords_y, coords_z), sgyy)
        self.sgzz=RegularGridInterpolator((coords_x, coords_y, coords_z), sgzz)
        self.sgyz=RegularGridInterpolator((coords_x, coords_y, coords_z), sgyz)
        self.sgxz=RegularGridInterpolator((coords_x, coords_y, coords_z), sgxz)
        self.sgxy=RegularGridInterpolator((coords_x, coords_y, coords_z), sgxy)


    #Compute derivatives, used for the interpolation functions. Use true or correlated displacement values.
    def calc_all_derivatives(self, uspacing, return_=False,
                             true_reference=False, return_u=False):
        if true_reference:
            if not hasattr(self, 'u_true_full_cart'):
                raise AttributeError('Missing u_true_full_cart. Run load_prep_FIDVC() first.')
            u0=self.u_true_full_cart
        else:
            if hasattr(self, 'u_corr_full_cart'):
                u0=self.u_corr_full_cart    # So u_corr_full_cart is used from load_prep_FIDVC
            elif hasattr(self, 'u_true_full_cart'):
                print('Warning: u_corr_full_cart missing. Falling back to u_true_full_cart for derivatives.')
                u0=self.u_true_full_cart
            else:
                raise AttributeError('Missing u_corr_full_cart and u_true_full_cart. Run load_prep_FIDVC() first and ensure FIDVC data is available.')
        # Here it is important for experimental data to have the right N (class and in script)
        # N is the image size, e.g., [256, 256, 128] for synthetic, or something like [113, 113, 49] for experimental

        self.u_spacing=uspacing
        print(self.boxsize,self.res,self.u_spacing)
        resnew=(2*self.boxsize)/(2*self.boxsize/self.res+1)*self.u_spacing
        print(resnew)
        ux=u0[0]/resnew[0]
        uy=u0[1]/resnew[1]
        uz=u0[2]/resnew[2]

        #Compute derivatives by using kernels on discretized displacement data, then use FFT to efficiently calculate the convolution of displaced/undisplaced data
        #The weight function w was included in Matlab already.
        #Use linear elasticity equations to go from strain to stress

        delx = -full5tap3D('x')
        dely = -full5tap3D('y')
        delz = -full5tap3D('z')

        uxx=fftconvolve(ux,delx,'same')
        uxy=fftconvolve(ux,dely,'same')
        uxz=fftconvolve(ux,delz,'same')
        uyx=fftconvolve(uy,delx,'same')
        uyy=fftconvolve(uy,dely,'same')
        uyz=fftconvolve(uy,delz,'same')
        uzx=fftconvolve(uz,delx,'same')
        uzy=fftconvolve(uz,dely,'same')
        uzz=fftconvolve(uz,delz,'same')

        #calc_stresses using equations (6, 7)
        mu=self.G
        nu=self.pr
        print(f'[DEBUG calc_all_derivatives] mu=self.G={mu:.2f}, nu=self.pr={nu:.4f}')
        if np.isclose(nu, 0.5):
            # In the incompressible case we can make use of div(u) = 0
            # to avoid the divergent term nu/(1-2*nu)
            sgxx = 2 * mu * uxx
            sgyy = 2 * mu * uyy
            sgzz = 2 * mu * uzz
        else:
            #compressible case
            divU = uxx + uyy + uzz
            sgxx = 2 * mu * (uxx + nu / (1 - 2 * nu) * divU)
            sgyy = 2 * mu * (uyy + nu / (1 - 2 * nu) * divU)
            sgzz = 2 * mu * (uzz + nu / (1 - 2 * nu) * divU)

        #Off-diagonal elements:
        sgyz = mu * (uyz + uzy)
        sgxz = mu * (uxz + uzx)
        sgxy = mu * (uxy + uyx)
        self.sigma_corr=[sgxx, sgyy, sgzz, sgyz, sgxz, sgxy]

        if return_:
            return sgxx, sgyy, sgzz, sgyz, sgxz, sgxy
        if return_u:
            return uxx,uxy,uxz,uyx,uyy,uyz,uzx,uzy,uzz

    #export MF
    def export_surface(self,lmax=200,dimensions=[200,200,100],save=True,boxsize=2,plotxz=True,
                       plotyz=True, point_coords_=False, plot3Dpoints=False):
        latsdeg, lonsdeg = _psh.expand.GLQGridCoord(lmax)
        point_coords=VX0+u_mesh   #Add displacements to base coordinates
        self.l=2*boxsize*self.radius
        pixr=np.array(dimensions)
        surface=np.zeros(dimensions)
        offset=pixr/2
        for p in point_coords.reshape(-1,3):   #Map transformed points to 3D grid surface
            ratio=p/self.l
            pc=pixr*ratio+offset

            pc0=round(pc[0]-0.5)
            pc1=round(pc[1]-0.5)
            pc2=round(pc[2]-0.5)
            surface[pc0,pc1,pc2]=2**32-1
        self.surface=surface.astype('int32')

        #Saving and plotting, cross-sections and scatter plots
        if save==True:
            tif.imsave('test.tif', self.surface, bigtiff=True)
        if plotxz==True:
            plt.rcParams['figure.figsize']=[10,10]
            fig,ax=plt.subplots()
            ax.imshow(self.surface[:,int(dimensions[1]/2)].transpose())
            factor=self.radius/boxsize
            rect = patches.Ellipse((dimensions[0]/2, dimensions[2]/2), dimensions[0]*factor , dimensions[2]*factor, linewidth=1, edgecolor='r', facecolor='none')
            ax.add_patch(rect)
            fig.show
        if plotyz==True:
            plt.rcParams['figure.figsize']=[10,10]
            fig,ax=plt.subplots()
            ax.imshow(self.surface[int(dimensions[1]/2)].transpose())
            factor=self.radius/boxsize
            rect = patches.Ellipse((dimensions[1]/2, dimensions[2]/2), dimensions[1]*factor , dimensions[2]*factor, linewidth=1, edgecolor='r', facecolor='none')
            ax.add_patch(rect)
            fig.show
        if point_coords_:
            return point_coords, VX0
        if plot3Dpoints:
            for p in [point_coords,VX0]:
                fig = plt.figure()
                ax = fig.add_subplot(111, projection='3d')
                X=p[:,:,0].flatten()
                Y=p[:,:,1].flatten()
                Z=p[:,:,2].flatten()
                surf = ax.plot_trisurf(X[::10], Y[::10], Z[::10], linewidth=1,alpha=0.3)
                ax.view_init(azim=0, elev=0)
                fig.show

    def calc_integralboundaries(self):   #This gives out four integral boundaries that depend on a and radius.
        a0=self.a/self.radius
        self.integralboundaries=[0,np.arcsin(a0),np.pi-np.arcsin(a0),np.pi] #Why is the integration split up like this?

    

# %%


# ============================================================================
# HERTZIAN_CONTACT CLASS
# Here you could implement a different profile by prescribing different stress functions.
# ============================================================================

class Hertzian_contact(Axialsymmetric):
    def __init__(self, E1, v1, E2, v2, savefoldername, radius=1, nmax=20,
                 savepath=DEFAULT_SAVEPATH,
                 integralboundaries=False, sigma_zz_bool=True, symmetric=True,
                 norm_radius=1):
        super().__init__(E1, v1, E2, v2, savefoldername, radius, nmax,
                         savepath, integralboundaries, sigma_zz_bool, symmetric, norm_radius)

    def sigma_zz(self, th):
        """Hertzian contact stress profile."""
        r = np.sin(th) * self.radius
        if (r < self.a and th < np.pi / 2):
            val = -2 * self.E_star / np.pi / self.radius * np.sqrt(self.a**2 - r**2)
        elif (r < self.a and th > np.pi / 2):
            val = 2 * self.E_star / np.pi / self.radius * np.sqrt(self.a**2 - r**2)
        else:
            val = 0
        return val

    def calc_from_F(self, F):
        """Calculate contact parameters from applied force."""
        self.F = F
        self.a = (3 * self.F * self.radius / 4 / self.E_star) ** (1 / 3)
        self.d = self.a**2 / self.radius
        self.p0 = 3 * self.F / (2 * np.pi * self.a**2)
        self.calc_integralboundaries()
        self.magnitude = self.F

    def calc_from_a(self, a):
        """Calculate contact parameters from contact radius."""
        self.a = a
        self.d = self.a**2 / self.radius
        self.F = 4 / 3 * self.E_star * self.radius**0.5 * self.d**(3 / 2)
        self.p0 = 3 * self.F / (2 * np.pi * self.a**2)
        self.calc_integralboundaries()
        self.magnitude = self.F

    def calc_integralboundaries(self):
        """Calculate integration boundaries for contact region."""
        a0 = self.a / self.radius
        self.integralboundaries = [0, np.arcsin(a0), np.pi - np.arcsin(a0), np.pi]
        print("Using a =", self.a, "and r =", self.radius, 
              "to calculate integralboundaries as", self.integralboundaries)



# ============================================================================
# CONFIGURATION SETTINGS
# ============================================================================

# Method selection (can choose both)
surface_method = True             # Use surface method reconstruction
volume_method = False              # Use volume method reconstruction

# Surface method settings:
# Fibonacci grid is used as standard for robust sampling (GLQ would introduce bias to poles)
FIBONACCI_SAMPLES = 2000           # Number of Fibonacci points for sampling synthetic surface data (standard 2000)

# Surface roughness noise (spectral spherical harmonic model)
ENABLE_SURFACE_ROUGHNESS = True
ROUGHNESS_RMS = 0.02               # relative RMS (e.g., 0.02 = 2% of radius) for visibility
ROUGHNESS_BETA = 3.0               # power-law exponent for C_l ~ (l+1)^-beta (rougher)
ROUGHNESS_LMIN = 2                 # remove monopole and dipole to avoid global gradient
ROUGHNESS_LMAX = None              # default: use lJmax
ROUGHNESS_SEED = 1234              # seed for reproducible roughness

# Roughness visualization settings
VISUALIZE_ROUGHNESS_FIELD = True           # Visualize the roughness perturbation as a heatmap
VISUALIZE_ROUGHENED_SURFACE = True         # Visualize the roughness-perturbed surface in point cloud
ROUGHNESS_VIZ_DPI = 150                    # Resolution for roughness PNG output

# Note: OUTPUT_DIR is configured at script top in the paths section

print('\n*** RUN SETTINGS ***')
print(f'  OUTPUT_DIR      = {OUTPUT_DIR}')
print(f'  ROUGHNESS_RMS   = {ROUGHNESS_RMS}')
print(f'  ROUGHNESS_BETA  = {ROUGHNESS_BETA}')
print(f'  ROUGHNESS_LMIN  = {ROUGHNESS_LMIN}')
print(f'  ROUGHNESS_LMAX  = {ROUGHNESS_LMAX}')
print(f'  ROUGHNESS_SEED  = {ROUGHNESS_SEED}')
print(f'  ROUGHNESS_ON    = {ENABLE_SURFACE_ROUGHNESS}')
print('*******************\n')


def compute_component_metrics(T_true_sph, T_recon_sph, mask, eps=1e-10, aad2_ref=None):
    """
    Compute error metrics for traction field components.
    
    Parameters:
    -----------
    T_true_sph : ndarray
        True traction field in spherical coordinates (3, n_points)
    T_recon_sph : ndarray
        Reconstructed traction field in spherical coordinates (3, n_points)
    mask : ndarray
        Boolean mask for points to evaluate
    eps : float
        Machine epsilon for normalization
    aad2_ref : float
        Reference amplitude for normalized metrics
    
    Returns:
    --------
    dict : Metrics for each component (T_r, T_theta, T_phi)
    """
    comp_names = ['T_r', 'T_theta', 'T_phi']
    metrics = {}
    for i, name in enumerate(comp_names):
        true = T_true_sph[i].flatten()[mask]
        recon = T_recon_sph[i].flatten()[mask]
        if true.size == 0:
            metrics[name] = {
                'rmse': 0,
                'mae': 0,
                'mbe': 0,
                'nrmse': 0,
                'nmae': 0,
                'nmre': 0,
                'nrmse2': 0,
                'nmae2': 0,
                'nmbe2': 0,
                'mean_true_abs': 0,
                'mean_recon_abs': 0,
            }
            continue
        diff = recon - true
        rmse = np.sqrt(np.mean(diff**2))
        mae = np.mean(np.abs(diff))
        mbe = np.mean(diff)
        mean_true_abs = np.mean(np.abs(true))
        nrmse = rmse / (mean_true_abs + eps)
        nmae = mae / (mean_true_abs + eps)
        nmre = np.mean(np.abs(diff) / (np.abs(true) + eps))
        mean_recon_abs = np.mean(np.abs(recon))
        if aad2_ref is not None and aad2_ref > 0:
            nrmse2 = rmse / (aad2_ref + eps)
            nmae2 = mae / (aad2_ref + eps)
            nmbe2 = mbe / (aad2_ref + eps)
        else:
            nrmse2 = 0
            nmae2 = 0
            nmbe2 = 0
        metrics[name] = {
            'rmse': rmse,
            'mae': mae,
            'mbe': mbe,
            'nrmse': nrmse,
            'nmae': nmae,
            'nmre': nmre,
            'nrmse2': nrmse2,
            'nmae2': nmae2,
            'nmbe2': nmbe2,
            'mean_true_abs': mean_true_abs,
            'mean_recon_abs': mean_recon_abs,
        }
    return metrics


def _build_surface_roughness_coeffs(lmax, rms, beta, lmin=1, seed=None):
    if lmax is None or rms <= 0:
        return None
    ell = np.arange(lmax + 1)
    power = np.zeros(lmax + 1, dtype=float)
    valid = ell >= max(lmin, 0)
    power[valid] = (ell[valid] + 1.0) ** (-beta)
    denom = np.sum((2 * ell[valid] + 1) / (4 * np.pi) * power[valid])
    if denom <= 0:
        return None
    power *= (rms ** 2) / denom
    return _psh.SHCoeffs.from_random(
        power,
        lmax=lmax,
        seed=seed,
        kind='real',
        normalization='4pi'
    )


def _normalize_rms(field, target_rms):
    if field is None:
        return None
    current = np.sqrt(np.mean(field ** 2))
    if current > 0:
        return field * (target_rms / current)
    return field


def _visualize_roughness_field(delta_glq, lmax, save_path=None, dpi=150):
    """
    Visualize the roughness perturbation field δ(θ,φ) as a heatmap.
    
    Parameters:
    -----------
    delta_glq : ndarray
        (51, 102) GLQ grid of relative radius perturbations
    lmax : int
        Maximum spherical harmonic degree used
    save_path : str, optional
        Directory to save the PNG. If None, don't save.
    dpi : int
        Resolution for saved figure
    """
    if delta_glq is None or delta_glq.size == 0:
        return
    
    mpl.use('Agg')
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    
    # Left: Roughness field as heatmap
    im1 = axes[0].imshow(delta_glq, cmap='RdBu_r', aspect='auto', origin='upper')
    axes[0].set_xlabel('Longitude index')
    axes[0].set_ylabel('Latitude index')
    axes[0].set_title(f'Surface Roughness δ(θ,φ) [lmax={lmax}]')
    plt.colorbar(im1, ax=axes[0], label='Relative radius perturbation')
    
    # Right: Statistics
    axes[1].axis('off')
    stats_text = (
        f'Roughness Field Statistics\n'
        f'{"─"*30}\n'
        f'Min: {delta_glq.min():.4e}\n'
        f'Max: {delta_glq.max():.4e}\n'
        f'Mean: {delta_glq.mean():.4e}\n'
        f'RMS: {np.sqrt(np.mean(delta_glq**2)):.4e}\n'
        f'Std Dev: {delta_glq.std():.4e}\n'
        f'\n'
        f'Grid: {delta_glq.shape[0]}×{delta_glq.shape[1]} ({delta_glq.size} points)\n'
        f'Max SH degree: {lmax}\n'
    )
    axes[1].text(0.1, 0.5, stats_text, fontfamily='monospace', fontsize=10,
                verticalalignment='center', transform=axes[1].transAxes)
    
    plt.tight_layout()
    if save_path:
        full_path = os.path.join(save_path, 'roughness_field_heatmap.png')
        plt.savefig(full_path, dpi=dpi, bbox_inches='tight')
        print(f'  Saved roughness field visualization: {full_path}')
    plt.close()


def _visualize_roughened_surface(Vp0, delta_glq, r0, lmax_plot=None, lJmax=None, save_path=None, dpi=150):
    """
    Create a side-by-side comparison of smooth vs roughened surface.
    
    Parameters:
    -----------
    Vp0 : ndarray
        (n, 3) smooth reference surface in code units (optimization grid)
    delta_glq : ndarray
        (51, 102) relative radius perturbations at full SH resolution
    r0 : float
        Characteristic radius for scaling
    lmax_plot : int
        Full SH resolution degree (default: 50)
    lJmax : int
        Optimization grid resolution degree (default: 20)
    save_path : str, optional
        Directory to save the PNG. If None, don't save.
    dpi : int
        Resolution for saved figure
    """
    if Vp0 is None or delta_glq is None:
        return
    
    if lmax_plot is None:
        lmax_plot = 50  # Standard plotting GLQ order (50) — keep plotting resolution separate from lJmax
    if lJmax is None:
        lJmax = 20
    
    mpl.use('Agg')
    fig = plt.figure(figsize=(14, 6))
    
    # Properly downsample roughness from full resolution (51×102) to optimization grid (21×42)
    # Using proper indexing to get exactly the right grid points
    lat_indices = np.linspace(0, lmax_plot, lJmax+1, dtype=int)
    lon_indices = np.linspace(0, 2*lmax_plot+1, 2*lJmax+2, dtype=int)
    
    delta_opt = delta_glq[np.ix_(lat_indices, lon_indices)]  # (21, 42)
    delta_flat = delta_opt.reshape(-1)  # 882 points
    
    # DEBUG: Print shapes to diagnose the issue
    print(f'  [DEBUG] delta_glq shape: {delta_glq.shape}, expected: (51, 102)')
    print(f'  [DEBUG] lat_indices: min={lat_indices.min()}, max={lat_indices.max()}, len={len(lat_indices)}')
    print(f'  [DEBUG] lon_indices: min={lon_indices.min()}, max={lon_indices.max()}, len={len(lon_indices)}')
    print(f'  [DEBUG] delta_opt shape: {delta_opt.shape}, expected: ({lJmax+1}, {2*lJmax+2})')
    print(f'  [DEBUG] delta_flat size: {len(delta_flat)}, expected: {(lJmax+1) * (2*lJmax+2)}')
    print(f'  [DEBUG] Vp0 size: {len(Vp0)}, expected: {(lJmax+1) * (2*lJmax+2)}')
    
    # Smooth surface
    ax1 = fig.add_subplot(121, projection='3d')
    pts_smooth = Vp0 / r0
    ax1.scatter(pts_smooth[:, 0], pts_smooth[:, 1], pts_smooth[:, 2], 
               c='C0', s=8, alpha=0.9, depthshade=False, edgecolors='none')
    ax1.set_title('Smooth Surface (Analytical)')
    ax1.set_xlabel('X'); ax1.set_ylabel('Y'); ax1.set_zlabel('Z')
    max_range = np.ptp(pts_smooth, axis=0).max() / 2
    mid = pts_smooth.mean(axis=0)
    ax1.set_xlim([mid[0] - max_range, mid[0] + max_range])
    ax1.set_ylim([mid[1] - max_range, mid[1] + max_range])
    ax1.set_zlim([mid[2] - max_range, mid[2] + max_range])
    ax1.view_init(elev=20, azim=30)
    
    # Roughened surface - ensure we have enough roughness points
    if len(delta_flat) < len(Vp0):
        # Pad if too few points
        delta_flat = np.pad(delta_flat, (0, len(Vp0) - len(delta_flat)), mode='edge')
    elif len(delta_flat) > len(Vp0):
        # Truncate if too many points
        delta_flat = delta_flat[:len(Vp0)]
    
    ax2 = fig.add_subplot(122, projection='3d')
    Vp_rough = Vp0 * r0 * (1.0 + delta_flat[:, None])  # Apply roughness in microns
    pts_rough = Vp_rough / r0
    
    # Check for extreme perturbations
    print(f'  [DEBUG] delta_flat range: [{delta_flat.min():.6f}, {delta_flat.max():.6f}]')
    print(f'  [DEBUG] Vp0 radius range: [{np.linalg.norm(Vp0, axis=1).min():.3f}, {np.linalg.norm(Vp0, axis=1).max():.3f}]')
    print(f'  [DEBUG] Vp_rough radius range: [{np.linalg.norm(Vp_rough, axis=1).min():.3f}, {np.linalg.norm(Vp_rough, axis=1).max():.3f}]')
    
    # Better color normalization: use symmetric vmin/vmax around 0
    delta_abs_max = np.max(np.abs(delta_flat))
    scatter = ax2.scatter(pts_rough[:, 0], pts_rough[:, 1], pts_rough[:, 2],
                         c=delta_flat, cmap='RdBu_r', s=12, alpha=1.0,
                         vmin=-delta_abs_max, vmax=delta_abs_max,
                         depthshade=False)  # Symmetric color scale
    ax2.set_title(f'Roughened Surface (N={len(pts_rough)} points)')
    ax2.set_xlabel('X'); ax2.set_ylabel('Y'); ax2.set_zlabel('Z')
    max_range_rough = np.ptp(pts_rough, axis=0).max() / 2
    mid_rough = pts_rough.mean(axis=0)
    ax2.set_xlim([mid_rough[0] - max_range_rough, mid_rough[0] + max_range_rough])
    ax2.set_ylim([mid_rough[1] - max_range_rough, mid_rough[1] + max_range_rough])
    ax2.set_zlim([mid_rough[2] - max_range_rough, mid_rough[2] + max_range_rough])
    ax2.view_init(elev=20, azim=30)
    cbar = plt.colorbar(scatter, ax=ax2, label='Perturbation (relative radius)')
    cbar.ax.ticklabel_format(style='scientific', scilimits=(0,0))
    
    plt.tight_layout()
    if save_path:
        full_path = os.path.join(save_path, 'roughness_surface_comparison.png')
        plt.savefig(full_path, dpi=dpi, bbox_inches='tight')
        print(f'  Saved roughened surface comparison: {full_path}')
    plt.close()

start_time = time.time()
print('='*60)
print('Hertzian Contact Analysis')
print('='*60)

# ============================================================================
# MATERIAL AND GEOMETRY PARAMETERS
# ============================================================================

E0 = 1500                        # Young's modulus (Pa) - standard 1500
nu0 = 0.4                        # Poisson's ratio - standard 0.4
microparticle_radius_meter = 12  # Microparticle radius in physical microns - standard 12

# Setup resolution (required for noise generation)
res = np.array([0.01, 0.01, 0.02])
boxsize = 1.28

# Synthetic bead parameters for visualization
beaddensity = 0.002             # Bead density - standard 0.002
beaddiameter = 0.2e-6 / microparticle_radius_meter  # 200nm physical diameter in code units
disp_noise_amp = 0.17           # Displacement noise amplitude - standard 0.17

# Bead generation options
IGNORE_VOLUME = False           # Allow overlapping beads (faster but less realistic)

# Control flow flags
STOP_AFTER_U = False            # Exit after computing displacement field
STOP_AFTER_GENERATE = False     # Exit after generating synthetic image pair

# Contact radius parameter
contact_radius_a = 0.5          # Standard contact radius parameter - standard 0.5

# Initialize Hertzian contact
t_Hertzian_contact = Hertzian_contact(
    E0, nu0, 1e9, 0.49999, 
    norm_radius=microparticle_radius_meter, 
    nmax=50,
    savefoldername='Hertzian_contact_fibonacci_',
    savepath=OUTPUT_DIR
)

t_Hertzian_contact.calc_from_a(contact_radius_a)
th0 = np.arcsin(t_Hertzian_contact.a / t_Hertzian_contact.radius)
t_Hertzian_contact.update_fullpath()  # Make sure fullsavepath is properly constructed
t_Hertzian_contact.fullsavepath = t_Hertzian_contact.savepath + '/'
t_Hertzian_contact.calc_and_store_lgs()
t_Hertzian_contact.calc_other_coefficients()

# Visualize traction/displacement profile
t_Hertzian_contact.u_halfplot(rangefactor=0.3, forringrange=0, add2a=False)

print(f'\n*** OUTPUT DIRECTORY: {t_Hertzian_contact.fullsavepath} ***\n')

# Early exit point: stop after computing displacement field
if STOP_AFTER_U:
    print('STOP_AFTER_U is True — exiting after computing u (early stop).')
    sys.exit(0)


# Generate synthetic image pair with noise (only needed for volume method)
if not surface_method:
    # If FIDVC .mat files already exist in the output directory, use them directly
    # instead of regenerating synthetic image data. This allows batch runners to
    # copy precomputed .mat inputs into the output folder and run the volume
    # traction pipeline without the (now-unnecessary) image-generation step.
    fidvc_mesh = os.path.join(t_Hertzian_contact.fullsavepath, 'mesh_profile.mat')
    fidvc_u = os.path.join(t_Hertzian_contact.fullsavepath, 'u_profile.mat')
    if os.path.exists(fidvc_mesh) and os.path.exists(fidvc_u):
        print('Found existing FIDVC .mat files in output dir — skipping synthetic image generation.')
        # Ensure minimal attributes that would be set by generate_image_pair are present
        try:
            t = t_Hertzian_contact
            t.res = res
            t.boxsize = boxsize
            # derive l, N, N2, and V as generate_image_pair would
            try:
                t.l = 2 * boxsize * t.radius
                t.N = t.l / t.res
                t.N2 = 2 * t.radius / t.res
                t.V = t.N2[0] * t.N2[1] * t.N2[2] * 1/6 * np.pi
            except Exception:
                pass
        except Exception:
            pass
    else:
        # Full image generation for volume method
        t_Hertzian_contact.generate_image_pair(
            beaddensity=beaddensity,
            res=res,
            boxsize=boxsize,
            beaddiameter=beaddiameter,
            disp_noise_amp=disp_noise_amp,
            variablesonly=False,
            ignorevolume=IGNORE_VOLUME
        )
        # If requested, stop immediately after generating image pair(s)
        if STOP_AFTER_GENERATE:
            print('STOP_AFTER_GENERATE is True — exiting after image generation (volume method).')
            sys.exit(0)
else:
    # Surface method only needs resolution variable
    t_Hertzian_contact.res = res
    t_Hertzian_contact.boxsize = boxsize

print(f'\nres = {t_Hertzian_contact.res}')
print('Time for setup:', time.time()-start_time)

# ============================================================================
# SPHERICAL HARMONIC SETTINGS
# ============================================================================

lJmax = 20       # Maximum spherical harmonic degree - standard 20
lKmax = lJmax    # Maximum K-index (set equal to lJmax)
lmax_plot = 50   # Maximum degree for plotting visualization - standard 50

# Surface roughness setup (shared by GLQ and Fibonacci)
roughness_lmax = ROUGHNESS_LMAX if ROUGHNESS_LMAX is not None else lJmax
roughness_coeffs = None
if ENABLE_SURFACE_ROUGHNESS and ROUGHNESS_RMS > 0:
    roughness_coeffs = _build_surface_roughness_coeffs(
        roughness_lmax,
        ROUGHNESS_RMS,
        ROUGHNESS_BETA,
        lmin=ROUGHNESS_LMIN,
        seed=ROUGHNESS_SEED
    )
    if roughness_coeffs is None:
        print('  ⚠️  Surface roughness disabled (invalid parameters)')
    else:
        print(
            f'  Surface roughness enabled: rms={ROUGHNESS_RMS:.3e}, '
            f'beta={ROUGHNESS_BETA}, lmin={ROUGHNESS_LMIN}, lmax={roughness_lmax}'
        )

if surface_method:
    print('\n' + '='*60)
    print('SURFACE METHOD - ANALYSIS')
    print('='*60)
    
    # Surface method parameters
    shapename = 'fibonacci_test'
    mu0 = lame_mu(E0, nu0)
    r0 = microparticle_radius_meter * 1e6  # convert to microns
    
    print(f'Shear modulus mu0 = {mu0} Pa')
    print(f'Radius r0 = {r0} μm')
    
    
    # Assign physical modulus
    t_Hertzian_contact.mu = mu0
    t_Hertzian_contact.G = mu0
    
    # Regularization parameters in energy functional
    myalpha = 1
    mybeta = 1
    mygamma = 0.3
    
    # Optimization settings
    plot_figure = False
    savefig = True
    myord = 1
    maxiter_per_period = 10  # Iterations per optimization period - standard 10
    CG_gtol = 1e-7           # Conjugate gradient tolerance - standard 1e-7
    eps = 1e-6
    minimizer = 'CG'
    minimizer_config = {'maxiter': maxiter_per_period, 'disp': True, 'gtol': CG_gtol}
    
    savename = shapename + ('_b%.0f_lmax%d'%(mybeta, lJmax))
    
    # Generate synthetic data with Fibonacci grid
    print(f'\nGenerating synthetic data with Fibonacci grid...')
    print('\n  Generating Fibonacci grid...')

    n_fib = FIBONACCI_SAMPLES if FIBONACCI_SAMPLES else 2000
    golden_ratio = (1 + np.sqrt(5)) / 2
    i = np.arange(0, n_fib)
    theta_fib = np.arccos(1 - 2 * (i + 0.5) / n_fib)  # colatitude
    phi_fib = 2 * np.pi * i / golden_ratio  # longitude
    
    # Convert to Cartesian (unit sphere)
    x_fib = np.sin(theta_fib) * np.cos(phi_fib)
    y_fib = np.sin(theta_fib) * np.sin(phi_fib)
    z_fib = np.cos(theta_fib)
    Vp_fib = np.stack([x_fib, y_fib, z_fib], axis=1)  # (n_fib, 3)
    
    print(f'  Generated {n_fib} Fibonacci points')
    
    # Save Fibonacci points for visualization (to show actual sampling)
    sampling_points_vis = Vp_fib  # Unit sphere Fibonacci points
    sampling_method_vis = 'Fibonacci'
    n_sampling_vis = n_fib
    
    # Evaluate analytical displacement at Fibonacci points
    u_r_fib = np.zeros(n_fib)
    u_theta_fib = np.zeros(n_fib)
    for idx in range(n_fib):
        u_r_fib[idx] = t_Hertzian_contact.u_r(theta_fib[idx])
        u_theta_fib[idx] = t_Hertzian_contact.u_th(theta_fib[idx])
    
    sin_t = np.sin(theta_fib)
    cos_t = np.cos(theta_fib)
    cos_p = np.cos(phi_fib)
    sin_p = np.sin(phi_fib)
    
    e_r = np.stack([sin_t * cos_p, sin_t * sin_p, cos_t], axis=1)
    e_theta = np.stack([cos_t * cos_p, cos_t * sin_p, -sin_t], axis=1)
    
    U_fib = u_r_fib[:, None] * e_r + u_theta_fib[:, None] * e_theta
    
    print(f'  Displacement at Fibonacci points: max |U| = {np.linalg.norm(U_fib, axis=1).max():.3e}')
    
    # Create deformed surface directly from Fibonacci points
    Vp_fib_deformed = r0 * (Vp_fib + U_fib)
    
    print(f'  Deformed Fibonacci surface: {len(Vp_fib_deformed)} points')
    print(f'    Position range: [{Vp_fib_deformed.min():.3e}, {Vp_fib_deformed.max():.3e}] μm')
    print(f'    Radii: [{np.linalg.norm(Vp_fib_deformed, axis=1).min():.2f}, {np.linalg.norm(Vp_fib_deformed, axis=1).max():.2f}] μm')
    
    if roughness_coeffs is not None:
        print('  Applying roughness to Fibonacci points...')
        delta_glq_fine = roughness_coeffs.expand(grid='GLQ', lmax=lmax_plot, extend=True).data
        delta_glq_fine = _normalize_rms(delta_glq_fine, ROUGHNESS_RMS)
    
        # RegularGridInterpolator is imported at module level
        latsdeg_fine, lonsdeg_fine = _psh.expand.GLQGridCoord(lmax_plot, extend=True)
        colatsdeg_fine = 90.0 - latsdeg_fine  # colatitude [0, 180]
    
        roughness_interp = RegularGridInterpolator(
            (colatsdeg_fine, lonsdeg_fine),
            delta_glq_fine,
            method='cubic',
            bounds_error=False,
            fill_value=0.0
        )
    
        colat_fib_deg = np.degrees(theta_fib)
        lon_fib_deg = np.degrees(phi_fib) % 360.0
        fib_coords = np.column_stack([colat_fib_deg, lon_fib_deg])
        delta_fib = roughness_interp(fib_coords)
        Vp_fib_deformed *= (1.0 + delta_fib[:, None])
        print(
            f'    Roughness applied: rms={ROUGHNESS_RMS:.3e}, '
            f'δ range: [{delta_fib.min():.3e}, {delta_fib.max():.3e}]'
        )
    
    Vp = Vp_fib_deformed
    
    print('\n  Getting analytical traction on GLQ grid (for comparison only)...')
    Tvec_mesh, Uvec_mesh, Tfv_mesh, Vp0_glq, isTfv = t_Hertzian_contact.export_SHE_GLQ(
        lmax_plot,
        lmax_low=lJmax,
        save_=False,
        export_=True,
        flipyz=False
    )
    
    print('\n  *** GEOMETRY VALIDATION ***')
    Vp_radii = np.linalg.norm(Vp, axis=1)
    print(f'  Deformed surface radii: [{Vp_radii.min():.2f}, {Vp_radii.max():.2f}] μm')
    print('  (Expected for Hertzian: ~9.5-12.5 μm)')
    print(f'  Number of points: {len(Vp)} (unstructured Fibonacci cloud)')
    
    expected_min, expected_max = 9.0, 13.0
    if not (expected_min <= Vp_radii.min() < expected_min + 1.0 and
            expected_max - 1.0 < Vp_radii.max() <= expected_max + 0.5):
        print('  ⚠️  WARNING: Deformed surface radii outside expected range!')
        print('     Proceeding with caution...')
    else:
        print('  ✓ Geometry validation passed')
    
    print(f'  ✓ Analytical traction mesh shape: {Tvec_mesh.shape} (GLQ grid for comparison)')
    print(f'  ✓ Deformed surface: {Vp.shape} (unstructured Fibonacci points)')
    print('  ✓ NO SH FITTING USED - direct analytical evaluation only!')
    
    theta_fib_deg = np.degrees(theta_fib)
    th0_deg_fib = np.degrees(th0)
    fib_contact_mask = (theta_fib_deg <= th0_deg_fib) | (theta_fib_deg >= 180 - th0_deg_fib)
    
    # Ensure isTfv_grid matches the grid shape  
    # (renaming to isTfv_grid for consistency with paper script)
    if isTfv.ndim == 1:
        isTfv_grid = isTfv.reshape(lJmax+1, 2*lJmax+2)
    else:
        isTfv_grid = isTfv.copy()
    
    print(f'[DEBUG] isTfv_grid shape: {isTfv_grid.shape}, dtype: {isTfv_grid.dtype}')
    print(f'[DEBUG] Contact region: {np.sum(~isTfv_grid.astype(bool))} points, Traction-free: {np.sum(isTfv_grid.astype(bool))} points')
    
    # Convert mesh to vector format (from working Hertzian_contact.py)
    Tvec_true = SHmesh2Vec(Tvec_mesh.astype(complex)/mu0)
    Uvec_true = SHmesh2Vec(Uvec_mesh.astype(complex))
    
    # Visualize point distribution (ACTUAL SAMPLING POINTS)
    print('\nVisualizing point cloud...')
    mpl.use('Agg')
    fig = plt.figure(figsize=plt.figaspect(1))
    ax = fig.add_subplot(111, projection='3d')
        
    # Use the actual sampling points (Fibonacci), not the optimization grid
    grid_vis = sampling_points_vis
    method_str = sampling_method_vis
        
    # Use appropriate contact mask (Fibonacci)
    vis_contact_mask = fib_contact_mask
    filtered_Tfv = grid_vis[~vis_contact_mask]
    filtered_nTfv = grid_vis[vis_contact_mask]
        
    ax.scatter3D(filtered_Tfv[:, 0], filtered_Tfv[:, 1], filtered_Tfv[:, 2],
            label=f'Traction-free region ({len(filtered_Tfv)} pts)', alpha=0.2, s=1, c='blue')
    ax.scatter3D(filtered_nTfv[:, 0], filtered_nTfv[:, 1], filtered_nTfv[:, 2],
            label=f'Contact region ({len(filtered_nTfv)} pts)', alpha=0.5, s=1, c='red')
        
    # Equal aspect ratio
    pts = grid_vis
    max_range = np.ptp(pts, axis=0).max() / 2
    mid = pts.mean(axis=0)
    ax.set_xlim([mid[0] - max_range, mid[0] + max_range])
    ax.set_ylim([mid[1] - max_range, mid[1] + max_range])
    ax.set_zlim([mid[2] - max_range, mid[2] + max_range])
        
    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('Z')
    ax.legend()
        
    # Generate appropriate filename and title based on sampling method
    grid_type = 'Fibonacci'
    output_filename = "fibonacci_point_cloud.png"
        
    ax.set_title(f'{grid_type} Sampling Grid ({n_sampling_vis} points)')
        
    plt.savefig(t_Hertzian_contact.fullsavepath + output_filename, dpi=300)
    print(f'Saved {grid_type} point cloud visualization')
    print(f'  → File: {output_filename}')
    print(f'  → Location: {os.path.abspath(t_Hertzian_contact.fullsavepath)}')
    print(f'  → Visualized {n_sampling_vis} actual {grid_type} sampling points')
    plt.close()
    
    # Visualize roughness if enabled
    if ENABLE_SURFACE_ROUGHNESS and roughness_coeffs is not None:
        print('\nVisualizing surface roughness...')
        # Get the roughness field that was applied
        delta_glq_viz = roughness_coeffs.expand(grid='GLQ', lmax=lmax_plot, extend=True).data
        delta_glq_viz = _normalize_rms(delta_glq_viz, ROUGHNESS_RMS)
    
        if VISUALIZE_ROUGHNESS_FIELD:
            _visualize_roughness_field(
                delta_glq_viz, 
                lmax=lmax_plot,
                save_path=t_Hertzian_contact.fullsavepath,
                dpi=ROUGHNESS_VIZ_DPI
            )
    
        if VISUALIZE_ROUGHENED_SURFACE:
            # Get optimization-grid surface coordinates using GLQ at lmax_low
            lat_opt, lon_opt = np.meshgrid(
                np.linspace(-np.pi/2, np.pi/2, lJmax+1),
                np.linspace(0, 2*np.pi, 2*lJmax+2, endpoint=False),
                indexing='ij'
            )
            
            # Convert to Cartesian (unit sphere)
            Vp0_opt = np.zeros((lat_opt.size, 3))
            Vp0_opt[:, 0] = np.cos(lat_opt.flatten()) * np.cos(lon_opt.flatten())
            Vp0_opt[:, 1] = np.cos(lat_opt.flatten()) * np.sin(lon_opt.flatten())
            Vp0_opt[:, 2] = np.sin(lat_opt.flatten())
            
            _visualize_roughened_surface(
                Vp0_opt,
                delta_glq_viz,
                r0,
                lmax_plot=lmax_plot,
                lJmax=lJmax,
                save_path=t_Hertzian_contact.fullsavepath,
                dpi=ROUGHNESS_VIZ_DPI
            )
    
if volume_method:
    # =====================================================================
    # VOLUME METHOD (FIDVC) - OPTIONAL EVALUATION
    # =====================================================================
    
    print('\n' + '='*60)
    print('VOLUME METHOD - FIDVC')
    print('='*60)
    
    # Load and prepare FIDVC data
    print('Loading and preparing FIDVC data...')
    try:
        t_Hertzian_contact.load_prep_FIDVC()
        has_corr = hasattr(t_Hertzian_contact, 'u_corr_full_cart')
        has_true = hasattr(t_Hertzian_contact, 'u_true_full_cart')
        if not (has_corr or has_true):
            raise RuntimeError(
                'FIDVC arrays were not initialized (missing u_corr_full_cart/u_true_full_cart). '
                'Check that u_profile.mat and mesh_profile.mat exist in the output directory.'
            )
        print('Successfully loaded FIDVC data')
        print('  u_corr_full_cart:', has_corr)
        print('  u_true_full_cart:', has_true)
        print("Attribute N:", getattr(t_Hertzian_contact, 'N', 'N not found'))
    except Exception as e:
        print(f'ERROR loading FIDVC data: {e}')
        # traceback is imported at module level
        traceback.print_exc()
        sys.exit('Could not load FIDVC data. Synthetic image data has been created; analyze it with FIDVC first to generate u_profile.mat and mesh_profile.mat in the output directory.')
    
    # Calculate derivatives
    print('Computing derivatives...')
    stepsize = 4        # This is "dm" in Matlab. We changed the standard 8 to the 4 defined here.
    t_Hertzian_contact.calc_all_derivatives(stepsize)
    
    
    eval_r = 0.8        # Percentage of radius at which to evaluate derivatives to recover tractions (0.8 means 80% of the radius)
    nmaxplot = lmax_plot
    print(f'Evaluating tractions at r = {eval_r}...')
    t_Hertzian_contact.T_spheresurface_VC(nmaxplot, eval_r=eval_r)
    
    # Calculate errors for volume method
    t_Hertzian_contact.T_spheresurface_true(nmaxplot)
    
    # Mark that tractions are ready for plotting (already in Pa)
    t_Hertzian_contact._plot_in_Pa = True
    
    # Configure plotting
    SHOW_ANALYTIC_VC = True
    SHOW_DEVIATION_VC = True
    
    # Generate 3D traction plots for volume method
    print('\nGenerating 3D traction plots for volume method...')
    mpl.use('Agg')
    
    try:
        t_Hertzian_contact.plot_T_3D(
            view='bottom', samescale='no',
            show_analytic=SHOW_ANALYTIC_VC, show_deviation=SHOW_DEVIATION_VC
        )
        
        t_Hertzian_contact.plot_T_3D(
            view='sidexz', samescale='no',
            show_analytic=SHOW_ANALYTIC_VC, show_deviation=SHOW_DEVIATION_VC
        )
        
        t_Hertzian_contact.plot_T_3D(
            view='bottom', samescale='no', coord='cart',
            show_analytic=SHOW_ANALYTIC_VC, show_deviation=SHOW_DEVIATION_VC
        )
        
        t_Hertzian_contact.plot_T_3D(
            view='sidexz', samescale='no', coord='cart',
            show_analytic=SHOW_ANALYTIC_VC, show_deviation=SHOW_DEVIATION_VC
        )
        
        print('✓ Generated 3D traction plots for volume method')
    except Exception as e:
        print(f'Warning: Could not create 3D plots: {e}')
        # traceback is imported at module level
        traceback.print_exc()
    
    # Calculate and save errors
    t_Hertzian_contact.calc_save_errors_VC(printimportant=False)
    
    df_vc = pd.DataFrame(t_Hertzian_contact.Errors_VC)
    print('\n' + '='*60)
    print('VOLUME METHOD - ERROR METRICS')
    print('='*60)
    print('\nFull Sphere Error Metrics (NRMSE2 highlighted column set):')
    print(df_vc.iloc[:, 2:9])
    
    # Compute contact-region-specific errors for VC method using theta bands
    try:
        T_true_vc_sph = np.asarray(t_Hertzian_contact.T_true_full_sph)
        T_recon_vc_sph = np.asarray(t_Hertzian_contact.T_corr_full_sph)
        
        latsdeg = np.array(t_Hertzian_contact.latsdeg)
        lonsdeg = np.array(t_Hertzian_contact.lonsdeg)
        lon_grid, lat_grid = np.meshgrid(lonsdeg, latsdeg)
        theta_flat = lat_grid.flatten()
        
        # Hertzian contact: θ ∈ [0°, 9°] ∪ [171°, 180°]
        contact_mask_vc = (theta_flat <= 9) | (theta_flat >= 171)
        traction_free_mask_vc = ~contact_mask_vc
        
        n_contact_vc = int(contact_mask_vc.sum())
        n_free_vc = int(traction_free_mask_vc.sum())
        n_total_vc = theta_flat.size
        
        aad2_contact_vc = np.sqrt(np.sum([
            np.mean(np.abs(T_true_vc_sph[i].flatten()[contact_mask_vc]))**2 for i in range(3)
        ]))
        metrics_contact_vc = compute_component_metrics(
            T_true_vc_sph,
            T_recon_vc_sph,
            contact_mask_vc,
            aad2_ref=aad2_contact_vc
        )
        metrics_free_vc = compute_component_metrics(T_true_vc_sph, T_recon_vc_sph, traction_free_mask_vc)
        
        print('='*60)
        print('NMRE = mean(|T_recon - T_true| / (|T_true| + eps)) over region')
        print('NMAE = mean(|T_recon - T_true|) / mean(|T_true|) over region')
        print(f'\nContact Region Only (θ ∈ [0°, 9°] ∪ [171°, 180°], {100*n_contact_vc/n_total_vc:.1f}% of grid):')
        for comp_name, m in metrics_contact_vc.items():
            print(f'  {comp_name} NMRE  = {m["nmre"]:.4f} ({100*m["nmre"]:.2f}%)')
            print(f'  {comp_name} NMAE  = {m["nmae"]:.4f} ({100*m["nmae"]:.2f}%)')
            print(f'  {comp_name} NRMSE = {m["nrmse"]:.4f} ({100*m["nrmse"]:.2f}%)')
            print(f'  {comp_name} NRMSE2 = {m["nrmse2"]:.4f} ({100*m["nrmse2"]:.2f}%)')
            print(f'  {comp_name} RMSE  = {m["rmse"]:.3f} Pa')
            print(f'  {comp_name} MAE   = {m["mae"]:.3f} Pa')
            print(f'  {comp_name} MBE   = {m["mbe"]:.3f} Pa')
        print(f'\nTraction-Free Region (spurious):')
        for comp_name, m in metrics_free_vc.items():
            print(f'  {comp_name} MAE   = {m["mae"]:.3f} Pa')
            print(f'  {comp_name} RMSE  = {m["rmse"]:.3f} Pa')
        
    except Exception as e:
        print(f'[DEBUG] Failed to compute VC contact-region errors: {e}')
        n_contact_vc = 0
        n_free_vc = 0
        metrics_contact_vc = {
            'T_r': {'rmse': 0, 'mae': 0, 'mbe': 0, 'nrmse': 0, 'nmae': 0, 'nmre': 0, 'nrmse2': 0, 'nmae2': 0, 'nmbe2': 0, 'mean_true_abs': 0, 'mean_recon_abs': 0},
            'T_theta': {'rmse': 0, 'mae': 0, 'mbe': 0, 'nrmse': 0, 'nmae': 0, 'nmre': 0, 'nrmse2': 0, 'nmae2': 0, 'nmbe2': 0, 'mean_true_abs': 0, 'mean_recon_abs': 0},
            'T_phi': {'rmse': 0, 'mae': 0, 'mbe': 0, 'nrmse': 0, 'nmae': 0, 'nmre': 0, 'nrmse2': 0, 'nmae2': 0, 'nmbe2': 0, 'mean_true_abs': 0, 'mean_recon_abs': 0},
        }
        metrics_free_vc = metrics_contact_vc
    
    print('='*60 + '\n')
    
    # Save errors to text file with parameters
    try:
        error_summary = []
        error_summary.append('='*60)
        error_summary.append('HERTZIAN CONTACT - VOLUME METHOD ERROR ANALYSIS')
        error_summary.append('='*60)
        error_summary.append('')
        
        # Add scenario and parameters
        error_summary.append('CONTACT SCENARIO: Hertzian Contact (FIDVC Volume Method)')
        error_summary.append('')
        error_summary.append('MATERIAL PARAMETERS:')
        error_summary.append(f'  E (Young\'s modulus): {E0} Pa')
        error_summary.append(f'  nu (Poisson\'s ratio): {nu0}')
        error_summary.append(f'  Microparticle radius: {microparticle_radius_meter*1e6} μm')
        error_summary.append('')
        
        error_summary.append('COMPUTATIONAL PARAMETERS:')
        error_summary.append(f'  nmax (spherical harmonics truncation): {t_Hertzian_contact.nmax}')
        error_summary.append(f'  lmax_plot: {lmax_plot}')
        error_summary.append(f'  Contact radius a: {t_Hertzian_contact.a}')
        error_summary.append(f'  Penetration depth d: {t_Hertzian_contact.d}')
        error_summary.append(f'  Applied force F: {t_Hertzian_contact.F}')
        error_summary.append(f'  Peak pressure p0: {t_Hertzian_contact.p0}')
        error_summary.append(f'  Image resolution: {t_Hertzian_contact.res}')
        error_summary.append(f'  Evaluation radius: eval_r = 0.8')
        error_summary.append(f'  Derivative stepsize: {stepsize}')
        error_summary.append('')
        
        error_summary.append('SYNTHETIC NANOPARTICLE DISTRIBUTION PARAMETERS:')
        error_summary.append(f'  Bead density: {beaddensity}')
        error_summary.append(f'  Bead diameter (code units): {beaddiameter}')
        error_summary.append(f'  Bead diameter (physical): {beaddiameter*microparticle_radius_meter*1e9:.1f} nm')
        error_summary.append(f'  Box size: {boxsize}')
        error_summary.append(f'  Displacement noise amplitude: {disp_noise_amp}')
        error_summary.append('')
        
        error_summary.append('ERROR METRICS:')
        error_summary.append('-'*60)
        error_summary.append('FULL SPHERE (includes traction-free region):')
        error_summary.append(df_vc.iloc[:, 2:9].to_string())
        error_summary.append('-'*60)
        error_summary.append('')
        error_summary.append('NMRE = mean(|T_recon - T_true| / (|T_true| + eps)) over region')
        error_summary.append('NMAE = mean(|T_recon - T_true|) / mean(|T_true|) over region')
        error_summary.append('CONTACT REGION ONLY Error Metrics:')
        if n_contact_vc + n_free_vc > 0:
            error_summary.append(f'  Contact region: {n_contact_vc} points ({100*n_contact_vc/(n_contact_vc+n_free_vc):.1f}% of grid)')
        else:
            error_summary.append('  Contact region: 0 points (0.0% of grid)')
        for comp_name, m in metrics_contact_vc.items():
            error_summary.append(f'  {comp_name} NMRE  = {m["nmre"]:.4f} ({100*m["nmre"]:.2f}%)')
            error_summary.append(f'  {comp_name} NMAE  = {m["nmae"]:.4f} ({100*m["nmae"]:.2f}%)')
            error_summary.append(f'  {comp_name} NRMSE = {m["nrmse"]:.4f} ({100*m["nrmse"]:.2f}%)')
            error_summary.append(f'  {comp_name} NRMSE2 = {m["nrmse2"]:.4f} ({100*m["nrmse2"]:.2f}%)')
            error_summary.append(f'  {comp_name} RMSE  = {m["rmse"]:.3f} Pa')
            error_summary.append(f'  {comp_name} MAE   = {m["mae"]:.3f} Pa')
            error_summary.append(f'  {comp_name} MBE   = {m["mbe"]:.3f} Pa')
            error_summary.append(f'  Mean |T_true| in contact = {m["mean_true_abs"]:.3f} Pa')
        error_summary.append('')
        error_summary.append('TRACTION-FREE REGION (spurious tractions):')
        for comp_name, m in metrics_free_vc.items():
            error_summary.append(f'  {comp_name} RMSE = {m["rmse"]:.3f} Pa')
            error_summary.append(f'  {comp_name} MAE  = {m["mae"]:.3f} Pa')
            error_summary.append(f'  Mean |T_recon| = {m["mean_recon_abs"]:.3f} Pa (should be ~0)')
        error_summary.append('')
        error_summary.append('INTERPRETATION:')
        error_summary.append('  Contact region reconstruction quality (NMRE):')
        for comp_name, m in metrics_contact_vc.items():
            error_summary.append(f'    {comp_name}: {100*m["nmre"]:.1f}%')
        error_summary.append('')
        error_summary.append('  Spurious traction amplitude in traction-free region:')
        for comp_name, m in metrics_free_vc.items():
            if m['mean_true_abs'] > 0:
                percent_of_contact = 100 * m["mae"] / (m['mean_true_abs'] + 1e-10)
            else:
                percent_of_contact = 0
            error_summary.append(f'    {comp_name}: {m["mae"]:.1f} Pa (~{percent_of_contact:.1f}% of contact pressure)')
        error_summary.append('')
        error_summary.append('  Overall NRMSE2 may be inflated by Gibbs phenomenon at contact boundaries.')
        error_summary.append('-'*60)
        
        # Persist error summary to a text file with structured fields (surface-style + volume params)
        error_file = os.path.join(t_Hertzian_contact.fullsavepath, 'volume_method_errors.txt')
        try:
            with open(error_file, 'w') as f:
                f.write('='*70 + '\n')
                f.write('HERTZIAN CONTACT - VOLUME METHOD ERROR ANALYSIS\n')
                f.write('='*70 + '\n\n')
                f.write(f'Runtime: {time.time()-start_time:.2f}s\n')
                f.write(f'E = {E0} Pa, nu = {nu0}\n')
                f.write(f'Microparticle radius = {microparticle_radius_meter*1e6} μm\n')
                f.write('\n')
                f.write('COMPUTATIONAL PARAMETERS:\n')
                f.write(f'  nmax (spherical harmonics truncation): {t_Hertzian_contact.nmax}\n')
                f.write(f'  lmax_plot: {lmax_plot}\n')
                f.write(f'  Contact radius a: {t_Hertzian_contact.a}\n')
                f.write(f'  Penetration depth d: {t_Hertzian_contact.d}\n')
                f.write(f'  Applied force F: {t_Hertzian_contact.F}\n')
                f.write(f'  Peak pressure p0: {t_Hertzian_contact.p0}\n')
                f.write(f'  Image resolution: {t_Hertzian_contact.res}\n')
                f.write(f'  Derivative stepsize: {stepsize}\n')
                f.write('\n')
                f.write('SYNTHETIC NANOPARTICLE DISTRIBUTION PARAMETERS:\n')
                f.write(f'  Bead density: {beaddensity}\n')
                f.write(f'  Bead diameter (code units): {beaddiameter}\n')
                try:
                    f.write(f'  Bead diameter (physical): {beaddiameter*microparticle_radius_meter*1e9:.1f} nm\n')
                except Exception:
                    f.write('  Bead diameter (physical): NA\n')
                f.write(f'  Box size: {boxsize}\n')
                f.write(f'  Displacement noise amplitude: {disp_noise_amp}\n')
                f.write('\n')
                f.write('ERROR METRICS:\n')
                f.write('-'*60 + '\n')
                f.write('FULL SPHERE (includes traction-free region):\n')
                f.write(df_vc.iloc[:, 2:9].to_string() + '\n')
                f.write('-'*60 + '\n\n')
                f.write('NMRE = mean(|T_recon - T_true| / (|T_true| + eps)) over region\n')
                f.write('NMAE = mean(|T_recon - T_true|) / mean(|T_true|) over region\n')
                f.write('CONTACT REGION ONLY Error Metrics:\n')
                for comp_name, m in metrics_contact_vc.items():
                    f.write(f'  {comp_name} NMRE  = {m["nmre"]:.4f} ({100*m["nmre"]:.2f}%)\n')
                    f.write(f'  {comp_name} NMAE  = {m["nmae"]:.4f} ({100*m["nmae"]:.2f}%)\n')
                    f.write(f'  {comp_name} NRMSE = {m["nrmse"]:.4f} ({100*m["nrmse"]:.2f}%)\n')
                    f.write(f'  {comp_name} NRMSE2 = {m["nrmse2"]:.4f} ({100*m["nrmse2"]:.2f}%)\n')
                    f.write(f'  {comp_name} RMSE  = {m["rmse"]:.3f} Pa\n')
                    f.write(f'  {comp_name} MAE   = {m["mae"]:.3f} Pa\n')
                    f.write(f'  {comp_name} MBE   = {m["mbe"]:.3f} Pa\n')
                    f.write(f'  Mean |T_true| in contact = {m["mean_true_abs"]:.3f} Pa\n')
                f.write('\n')
                f.write('TRACTION-FREE REGION (spurious tractions):\n')
                for comp_name, m in metrics_free_vc.items():
                    f.write(f'  {comp_name} RMSE = {m["rmse"]:.3f} Pa\n')
                    f.write(f'  {comp_name} MAE  = {m["mae"]:.3f} Pa\n')
                    f.write(f'  Mean |T_recon| = {m["mean_recon_abs"]:.3f} Pa (should be ~0)\n')
                f.write('\n')
                f.write('INTERPRETATION:\n')
                f.write('  Contact region reconstruction quality (NMRE):\n')
                for comp_name, m in metrics_contact_vc.items():
                    f.write(f'    {comp_name}: {100*m["nmre"]:.1f}%\n')
                f.write('\n')
                f.write('  Spurious traction amplitude in traction-free region:\n')
                for comp_name, m in metrics_free_vc.items():
                    if m['mean_true_abs'] > 0:
                        percent_of_contact = 100 * m['mae'] / (m['mean_true_abs'] + 1e-10)
                    else:
                        percent_of_contact = 0
                    f.write(f'    {comp_name}: {m["mae"]:.1f} Pa (~{percent_of_contact:.1f}% of contact pressure)\n')
                f.write('\n')
                f.write('  Overall NRMSE2 may be inflated by Gibbs phenomenon at contact boundaries.\n')
                f.write('-'*60 + '\n')
            print(f'Error summary saved to: {error_file}')
        except Exception as e:
            print(f'Warning: could not save error file: {e}')
    except Exception as e:
        print(f'Warning: could not save error file: {e}')
    
    volume_time = time.time() - start_time
    print(f'Volume method completed in {volume_time:.2f}s')

# =====================================================================
# SURFACE METHOD - OPTIMIZATION SETUP (Fibonacci or GLQ)
# =====================================================================

if surface_method:
    print('\n' + '='*60)
    print('SURFACE METHOD OPTIMIZATION - FIBONACCI INITIALIZATION')
    print('='*60)

    # Vp is already set correctly in the GLQ/Fibonacci branches above
    # For GLQ: Vp = Vp0 * r0 (already in microns)
    # For Fibonacci: Vp_deformed is set (also in microns)
    print(f'\nSurface coordinates ready: Vp shape = {Vp.shape}')
    print(f'  Vp range: [{Vp.min():.3e}, {Vp.max():.3e}] microns')

    print('\nPreparing for surface method optimization...')
    # Load coefficient matrices with explicit path
    Cmat, Dmat = loadCoeffs(mu0, nu0, lJmax, 'reg', coeff_dir=COEFF_DIR)
    print('Loaded C and D matrices')

    # Get Jacobian matrices
    def get_damping():
        lmax = lJmax
        ldamp_hi = lmax
        ldamp_lo = lmax - 5
        lv, _ = LM_list(lmax)
        lv_ones = np.ones_like(lv)
        lv_lim = np.minimum(np.maximum(lv, ldamp_lo), ldamp_hi)
        ldamp = (np.maximum(lv_lim - ldamp_lo, 0) / (ldamp_hi - ldamp_lo))**1
        Q = spm.csr_matrix(np.diag(np.tile(ldamp, 3)).astype(complex))
        return Q

    def get_Jacobians(printtimes=True):
        Q = get_damping()
        tic = time.time()
        Lmat = genLmat(lJmax, Cmat=Cmat, Dmat=Dmat)
        if printtimes:
            print('Time for generating L matrix: %.2fs' % (time.time() - tic))
        tic = time.time()
        Smat = genSmat(lJmax, Cmat=Cmat, Dmat=Dmat)
        if printtimes:
            print('Time for generating S matrix: %.2fs' % (time.time() - tic))
        Dinv = spm.linalg.inv(Dmat)
        CDmat = Cmat.dot(Dinv)
        tic = time.time()
        CDSmat = np.asmatrix(CDmat.dot(Smat))
        SHCDS = np.asmatrix(Smat).H.dot(CDSmat)
        if printtimes:
            print('Time for generating S^HCD^{-1}S matrix: %.2fs' % (time.time() - tic))
        tic = time.time()
        QCDSmat = Q.dot(CDSmat)
        QCDSHQCDS = QCDSmat.H.dot(QCDSmat).real
        if printtimes:
            print('Time for generating QCDSHQCDS matrix: %.2fs' % (time.time() - tic))
        tic = time.time()
        
        # Ensure isTfv is properly shaped for the mesh
        if isTfv.ndim == 1:
            isTfv_grid = isTfv.reshape(lJmax+1, 2*lJmax+2)
        else:
            isTfv_grid = isTfv
        
        P = np.diag(np.stack([isTfv_grid]*3, axis=-1).flatten())
        
        TresJac = myalpha*2*np.asmatrix(np.dot(np.dot(Lmat.T, P), Lmat))/(lJmax+1)/(lJmax*2+1)
        EelJac = mybeta*2*np.pi*(SHCDS+SHCDS.H)
        penJac = mygamma*(QCDSHQCDS+QCDSHQCDS.H)
        if printtimes:
            print('time of matrix build:', time.time() - tic)
        return TresJac, EelJac, penJac, Q

    TresJac, EelJac, penJac, Q = get_Jacobians()

    # Get interpolation function for surface method
    print('\nCreating interpolation function for surface method optimization...')
    
    # Define Fibonacci grid for optimization
    latsdeg, lonsdeg = _psh.expand.GLQGridCoord(lJmax, extend=True)
    lon0, lat0 = np.meshgrid(lonsdeg, latsdeg)
    X0surf = np.stack([lat0, lon0], axis=-1)
    X0 = GLQCartCoord(lJmax)
    
    # =====================================================
    # FIBONACCI-BASED INITIALIZATION OF SAMPLED SURFACE (avoids oversampling at poles, unlike GLQ-based sampling below)
    # =====================================================
    print('\n  *** FIBONACCI GRID INITIALIZATION ***')
    print('  Creating displacement interpolator from Fibonacci-sampled deformed surface...')
    print('  Using Ur_interp to create radial displacement function from deformed coordinates')
        
    # Vp contains the deformed surface coordinates in code units (microns)
    # Create the interpolation function that maps (θ, φ) → u_r(θ, φ)
    f_interp = Ur_interp(Vp/r0, lmax=lJmax+20, plot_figure=plot_figure)
        
    # Standard zero initialization for Fibonacci (same as GLQ)
    u0_surf = np.zeros((lJmax+1, lJmax*2+2, 2))
    print(f'  ✓ Initialized u0_surf with zeros (Fibonacci method): shape={u0_surf.shape}')
        
    # Prepare for optimization
    target_args = (f_interp, lJmax, mybeta, myord, X0surf, X0, isTfv_grid, Cmat, Dmat,
                   mu0, nu0, np.array([1]), np.array([1]), eps, '2-point', 
                   (TresJac, EelJac, penJac), mygamma)
    u_surf = u0_surf.flatten().copy()
    u_surf_list = [u_surf]

    print('\n' + '='*60)
    print('Starting optimization...')
    print('='*60)

    # Run optimization iterations
    N_iterations = 15  # Number of optimization iterations - standard 15
    for i in range(N_iterations):
        print(f'\nPeriod {i:4d}  {"Tr":<13}{"Eel":<13}{"f0":<13}{"f1":<13}{"f2":<13}{"f"}')
        tic = time.time()
        
        def print_iter(xk):
            Uvec, aK, Tvec = usurf2vec(xk, f_interp, lJmax, X0surf=X0surf, X0=X0, Cmat=Cmat, Dmat=Dmat)
            Tdist = Tvec2Tres(Tvec, lJmax, isTfv=isTfv_grid, norm_order=myord)
            Eel = np.vdot(Uvec, Tvec).real*2*np.pi
            pen = np.vdot(Q.dot(Tvec), Q.dot(Tvec)).real
            dr = usurf2dr2(xk, *target_args)
            print('%13.4ePa%13.4epJ%13.4e%13.4e%13.4e%13.4e' % (
                np.sqrt(Tdist)*mu0, Eel*(r0/1e6)**3*mu0*1e12, Tdist, Eel, pen, dr))
        
        u_res = minimize(usurf2dr2, u_surf.flatten(), args=target_args, jac=grad_usurf2dr2,
                        method=minimizer, options=minimizer_config, callback=print_iter)
        
        print(f'Iteration Time: {time.time() - tic:.2f}s')
        u_surf = u_res.x.copy()
        u_surf_list.append(u_surf)
        
        if u_res.success:
            print('\nOptimization converged successfully!')
            break

    # ============================================================================
    # POST-OPTIMIZATION ANALYSIS AND PLOTTING
    # ============================================================================

    print('\n' + '='*60)
    print('POST-OPTIMIZATION ANALYSIS')
    print('='*60)

    # Get final optimized result
    u_surf_final = u_surf
    Uvec_recon, aK_recon, Tvec_recon = usurf2vec(u_surf_final, f_interp, lJmax, 
                                                   X0surf=X0surf, X0=X0, 
                                                   Cmat=Cmat, Dmat=Dmat)

    # Convert reconstructed tractions to mesh for plotting and analysis
    print('Converting traction coefficients to mesh representation...')
    t_Hertzian_contact.mu = mu0
    t_Hertzian_contact.T_spheresurface_MF(Tvec_recon, lJmax)

    # Get analytical tractions for comparison
    print('Computing analytical traction field...')
    t_Hertzian_contact.T_spheresurface_true(lJmax)

    # ============================================================================
    # PLOTTING
    # ============================================================================

    print('\nGenerating traction plots...')
    mpl.use('Agg')

    # Configure plotting
    SHOW_ANALYTIC_MF = True
    SHOW_DEVIATION_MF = True

    # Set _plot_in_Pa = True to prevent double-scaling
    t_Hertzian_contact._plot_in_Pa = True

    # Ensure reconstructed and analytic tractions are evaluated at the
    # plotting resolution so the 3D/2D visualizations are smooth even
    # when the reconstruction cutoff `lJmax` is much smaller.
    try:
        t_Hertzian_contact.T_spheresurface_MF(Tvec_recon, lmax_plot)
        t_Hertzian_contact.T_spheresurface_true(lmax_plot)
        print(f'  Note: upsampled MF and analytic tractions to lmax_plot={lmax_plot} for plotting')
    except Exception as _e:
        print(f'  Warning: could not upsample tractions for plotting: {_e}')

    # Generate 3D plots (multiple views)
    try:
        t_Hertzian_contact.plot_T_3D(
            view='bottom', samescale='no',
            show_analytic=SHOW_ANALYTIC_MF, show_deviation=SHOW_DEVIATION_MF, method_='MF'
        )
        
        t_Hertzian_contact.plot_T_3D(
            view='sidexz', samescale='no',
            show_analytic=SHOW_ANALYTIC_MF, show_deviation=SHOW_DEVIATION_MF, method_='MF'
        )
        
        t_Hertzian_contact.plot_T_3D(
            view='bottom', samescale='no', coord='cart',
            show_analytic=SHOW_ANALYTIC_MF, show_deviation=SHOW_DEVIATION_MF, method_='MF'
        )
        
        t_Hertzian_contact.plot_T_3D(
            view='sidexz', samescale='no', coord='cart',
            show_analytic=SHOW_ANALYTIC_MF, show_deviation=SHOW_DEVIATION_MF, method_='MF'
        )
        
        print('✓ Generated 3D traction plots')
    except Exception as e:
        print(f'Warning: Could not create 3D plots: {e}')

    # ============================================================================
    # ERROR ANALYSIS: Contact region vs traction-free errors
    # ============================================================================

    print('\n' + '='*60)
    print('SURFACE METHOD - ERROR ANALYSIS')
    print('='*60)

    # Reconstruct at high-res by setting mu=mu0 and calling T_spheresurface_MF
    t_Hertzian_contact.mu = mu0
    t_Hertzian_contact.T_spheresurface_MF(Tvec_recon, lmax_plot)

    # Get analytical tractions at high-res for comparison
    t_Hertzian_contact.T_spheresurface_true(lmax_plot)

    # === DIAGNOSTIC: Print contact radius, radius, and analytical traction stats ===
    print(f"[DIAG] Contact radius a: {t_Hertzian_contact.a}")
    print(f"[DIAG] Sphere radius: {t_Hertzian_contact.radius}")
    T_true_sph_diag = np.asarray(t_Hertzian_contact.T_true_full_sph)
    print(f"[DIAG] Analytical traction shape: {T_true_sph_diag.shape}")
    print(f"[DIAG] Analytical traction min: {T_true_sph_diag.min():.3e}, max: {T_true_sph_diag.max():.3e}, mean: {T_true_sph_diag.mean():.3e}")
    print(f"[DIAG] Nonzero elements: {(np.abs(T_true_sph_diag) > 1e-10).sum()} / {T_true_sph_diag.size}")

    # Extract spherical traction fields
    T_recon_sph = np.asarray(t_Hertzian_contact.T_MF_full_sph)  # (3, n_points)
    T_true_sph = np.asarray(t_Hertzian_contact.T_true_full_sph)  # (3, n_points)

    # Define contact region using theta bands (paper definition)
    latsdeg = np.array(t_Hertzian_contact.latsdeg)
    lonsdeg = np.array(t_Hertzian_contact.lonsdeg)
    lon_grid, lat_grid = np.meshgrid(lonsdeg, latsdeg)
    theta_flat = lat_grid.flatten()

    # Hertzian contact: θ ∈ [0°, 9°] ∪ [171°, 180°]
    contact_mask = (theta_flat <= 9) | (theta_flat >= 171)
    traction_free_mask = ~contact_mask

    n_contact = int(contact_mask.sum())
    n_free = int(traction_free_mask.sum())
    n_total = theta_flat.size

    # Compute per-component error metrics for surface method
    aad2_contact = np.sqrt(np.sum([
        np.mean(np.abs(T_true_sph[i].flatten()[contact_mask]))**2 for i in range(3)
    ]))
    metrics_contact = compute_component_metrics(
        T_true_sph,
        T_recon_sph,
        contact_mask,
        aad2_ref=aad2_contact
    )
    metrics_free = compute_component_metrics(T_true_sph, T_recon_sph, traction_free_mask)



    # Calculate errors using the standard function
    try:
        t_Hertzian_contact.calc_save_errors_MF(printfull=False, hist=False)
        df = pd.DataFrame(t_Hertzian_contact.Errors_MF)
        
        # Print error metrics with contact-region summary
        print('\n' + '='*60)
        print('SURFACE METHOD - ERROR ANALYSIS')
        print('='*60)
        print('\nFull Sphere Error Metrics (NRMSE2 highlighted column set):')
        print(df.iloc[:, 2:9])
        print('='*60)
        
        if n_contact > 0:
            print('NMRE = mean(|T_recon - T_true| / (|T_true| + eps)) over region')
            print('NMAE = mean(|T_recon - T_true|) / mean(|T_true|) over region')
            print(f'\nContact Region Only (θ ∈ [0°, 9°] ∪ [171°, 180°], {100*n_contact/n_total:.1f}% of grid):')
            for comp_name, m in metrics_contact.items():
                print(f'  {comp_name} NMRE  = {m["nmre"]:.4f} ({100*m["nmre"]:.2f}%)')
                print(f'  {comp_name} NMAE  = {m["nmae"]:.4f} ({100*m["nmae"]:.2f}%)')
                print(f'  {comp_name} NRMSE = {m["nrmse"]:.4f} ({100*m["nrmse"]:.2f}%)')
                print(f'  {comp_name} NRMSE2 = {m["nrmse2"]:.4f} ({100*m["nrmse2"]:.2f}%)')
                print(f'  {comp_name} RMSE  = {m["rmse"]:.3f} Pa')
                print(f'  {comp_name} MAE   = {m["mae"]:.3f} Pa')
                print(f'  {comp_name} MBE   = {m["mbe"]:.3f} Pa')
        
        if n_free > 0:
            print(f'\nTraction-Free Region (spurious from Gibbs phenomenon):')
            for comp_name, m in metrics_free.items():
                print(f'  {comp_name} MAE   = {m["mae"]:.3f} Pa')
                print(f'  {comp_name} RMSE  = {m["rmse"]:.3f} Pa')
        
        print('='*60 + '\n')
        
        # Persist error summary to a text file
        error_file_txt = os.path.join(t_Hertzian_contact.fullsavepath, 'surface_method_errors.txt')
        with open(error_file_txt, 'w') as f:
            f.write('='*70 + '\n')
            f.write('SURFACE METHOD - ERROR ANALYSIS\n')
            f.write('='*70 + '\n\n')
            f.write(f'Runtime: {time.time()-start_time:.2f}s\n')
            f.write(f'E = {E0} Pa, nu = {nu0}\n')
            f.write(f'lJmax = {lJmax}, lmax_plot = {lmax_plot}\n')
            try:
                f.write(f'ROUGHNESS_RMS = {ROUGHNESS_RMS}\n')
            except Exception:
                # ROUGHNESS_RMS may not be defined in all calling contexts
                f.write('ROUGHNESS_RMS = NA\n')
            try:
                f.write(f'FIBONACCI_SAMPLES = {FIBONACCI_SAMPLES}\n')
            except Exception:
                f.write('FIBONACCI_SAMPLES = NA\n')
            f.write(f'Contact radius a = {t_Hertzian_contact.a} (code units)\n')
            f.write(f'Contact region: θ ∈ [0°, 9°] ∪ [171°, 180°], {n_contact} points ({100*n_contact/n_total:.1f}% of grid)\n\n')
            f.write(f'Contact region AAD2 (vector magnitude) = {aad2_contact:.3f} Pa\n\n')
            
            f.write('FULL SPHERE Error Metrics (includes traction-free region):\n')
            f.write(str(df.iloc[:, 2:9]))
            f.write('\n\n')
            
            f.write('NMRE = mean(|T_recon - T_true| / (|T_true| + eps)) over region\n')
            f.write('NMAE = mean(|T_recon - T_true|) / mean(|T_true|) over region\n')
            f.write('CONTACT REGION ONLY Error Metrics:\n')
            for comp_name, m in metrics_contact.items():
                f.write(f'  {comp_name} NMRE  = {m["nmre"]:.4f} ({100*m["nmre"]:.2f}%)\n')
                f.write(f'  {comp_name} NMAE  = {m["nmae"]:.4f} ({100*m["nmae"]:.2f}%)\n')
                f.write(f'  {comp_name} NRMSE = {m["nrmse"]:.4f} ({100*m["nrmse"]:.2f}%)\n')
                f.write(f'  {comp_name} NRMSE2 = {m["nrmse2"]:.4f} ({100*m["nrmse2"]:.2f}%)\n')
                f.write(f'  {comp_name} RMSE  = {m["rmse"]:.3f} Pa\n')
                f.write(f'  {comp_name} MAE   = {m["mae"]:.3f} Pa\n')
                f.write(f'  {comp_name} MBE   = {m["mbe"]:.3f} Pa\n')
                f.write(f'  Mean |T_true| in contact = {m["mean_true_abs"]:.3f} Pa\n')
            f.write('\n')
            
            f.write('TRACTION-FREE REGION (spurious tractions):\n')
            for comp_name, m in metrics_free.items():
                f.write(f'  {comp_name} RMSE = {m["rmse"]:.3f} Pa\n')
                f.write(f'  {comp_name} MAE  = {m["mae"]:.3f} Pa\n')
                f.write(f'  Mean |T_recon| = {m["mean_recon_abs"]:.3f} Pa (should be ~0)\n')
            f.write('\n')
            
            f.write('INTERPRETATION:\n')
            f.write('  Contact region reconstruction quality (NMRE):\n')
            for comp_name, m in metrics_contact.items():
                f.write(f'    {comp_name}: {100*m["nmre"]:.1f}%\n')
            f.write(f'  Overall NRMSE2 inflated by Gibbs phenomenon at contact boundary.\n')
        
        print(f'✓ Error analysis complete')
        print(f'  - Error summary: {error_file_txt}')
    except Exception as e:
        print(f'Warning: Could not compute error metrics: {e}')

    print(f'\nTotal time: {time.time() - start_time:.2f}s')
    print('='*60)
    print('Analysis complete!')
    print('='*60)

