import numpy as np
import scipy as sp
import scipy.sparse as spm
from scipy.sparse.linalg import lsqr, spsolve
from scipy.interpolate import RectBivariateSpline
from scipy.optimize import minimize
from scipy.io import loadmat, savemat

import matplotlib.pyplot as plt
from matplotlib import cm, colors
from mpl_toolkits.mplot3d import Axes3D
from mpl_toolkits.mplot3d.art3d import Line3DCollection
import sys, time, os.path
from itertools import permutations
import pyshtools

sys.path.append('..')
from shelastic.shutil import SphCoord_to_CartCoord, CartCoord_to_SphCoord, GLQCartCoord
from shelastic.shutil import SHCilmToVector, SHVectorToCilm, SHVec2mesh, SHmesh2Vec
from shelastic.shutil import TransMat, l_coeffs, m_coeffs, LM_list, lmk2K, K2lmk
from shelastic.shbv import generate_submat, Uvec2Tvec, Tvec2Uvec
from shelastic.shvis  import plotfv, vismesh, visSHVec, visSH3d

def loadCoeffs(mu0, nu0, lmax, shtype, coeff_dir=None, verbose=True):
    mu = 1.; nu = nu0; 
    lJmax = lKmax = lmax; 
    if coeff_dir is None:
        coeff_dir = os.path.join('..', 'shelastic', 'default_modes')
    Dmat = generate_submat(loadmat(os.path.join(coeff_dir, 'Umodes.mat')),
                           mu, nu, lKmax, lJmax, shtype=shtype, verbose=verbose).tocsc()
    Cmat = generate_submat(loadmat(os.path.join(coeff_dir, 'Tmodes.mat')),
                           mu, nu, lKmax, lJmax, shtype=shtype, verbose=verbose).tocsc()
    return Cmat, Dmat

def arbitrary_force(rcut_r0, mu0 = 300/3, nu0 = 0.499, r0 = 5, pF=1, noise_level=0, plot_figure=True,
                    lJmax=20, lmax_plot=60, mag = 4, dilation = 2, Cmat=None, Dmat=None):
    if Cmat is None and Dmat is None:
        Cmat, Dmat = loadCoeffs(mu0, nu0, lJmax, 'reg')
    N = GLQCartCoord(lJmax); X0 = N*r0;
    lKmax = lJmax; rcut = r0 * rcut_r0;

    # Select contact region
    pLeft = np.array([0, -r0, 0]); pRight = np.array([0, r0, 0]);
    distLeft = rcut - np.linalg.norm(X0 - pLeft.reshape(1, 1, 3), axis=-1)
    distRight= rcut - np.linalg.norm(X0 - pRight.reshape(1, 1, 3),axis=-1)
    regionLeft = 1/(1 + np.exp(-distLeft*mag))
    regionRight = 1/(1 + np.exp(-distRight*mag))

    # Add opposing forces on both sides
    T_usr_mesh_init = np.zeros_like(X0, dtype=complex)
    F_mesh = np.zeros_like(X0, dtype=complex)
    F_mesh[..., 1] = (pF+0j)*regionLeft - (pF+0j)*regionRight
    T_usr_mesh_init = F_mesh

    # decompose the force field into spherical harmonics
    Tvec = SHmesh2Vec(T_usr_mesh_init)
    
    # Reconstruct the shape
    Uvec = Tvec2Uvec(Tvec, Cmat, Dmat)
    umesh_fine = SHVec2mesh(Uvec*r0, lmax=lmax_plot, SphCoord=False, Complex=True)
    VX0 = GLQCartCoord(lmax_plot)*r0
    Vp = (VX0 + umesh_fine).reshape(-1, 3)
    
    # add noise to the shape
    npts, ndim = Vp.shape
    Vr, Vtheta, Vphi = CartCoord_to_SphCoord(Vp[...,0], Vp[...,1], Vp[...,2])
    noise = np.random.normal(scale=noise_level, size=npts)
    Vp = np.stack(SphCoord_to_CartCoord(Vr + noise, Vtheta, Vphi), axis=-1)

    # obtain the traction-free region
    TcLeft = (rcut+dilation) > np.linalg.norm(VX0 - pLeft.reshape(1, 1, 3), axis=-1)
    TcRight= (rcut+dilation) > np.linalg.norm(VX0 - pRight.reshape(1, 1, 3), axis=-1)
    TfRegion = np.logical_not(np.logical_or(TcLeft, TcRight))
    if plot_figure:
        fig, axs = visSHVec(Tvec*mu0, lmax_plot=lmax_plot, SphCoord=True, Complex=True,
                            config_quiver=(2, 3, 'k', 1000), lonshift=180, figsize=(6,3), 
                            n_vrange=(-100, 100), s_vrange=(0, 50), show=False)
        if len(Vphi) == (lmax_plot+1)*(2*lmax_plot+1):
            Vshape = (lmax_plot+1, 2*lmax_plot+1)
        else:
            Vshape = (lmax_plot+1, 2*lmax_plot+2)
        LONS = np.rad2deg(Vphi).reshape(Vshape)
        LATS = 90 - np.rad2deg(Vtheta).reshape(Vshape)
        axs[0].contour(LONS, LATS, TfRegion, [0.5,], colors='k', linewidths=1)
        axs[1].contour(LONS, LATS, TfRegion, [0.5,], colors='k', linewidths=1)
    
    Tfv = TfRegion.flatten()
    return Tvec, Uvec, Vp, Tfv

def Ur_interp(Vp, lmax=30, plot_figure=False):
    Vr, Vthe, Vphi = CartCoord_to_SphCoord(Vp[...,0], Vp[...,1], Vp[...,2])
    Vphi[Vphi < 0] += 2*np.pi
    Vlat = 90-np.rad2deg(Vthe)
    Vlon = np.rad2deg(Vphi)
    ur = (Vr - 1)

    urcilm_interp, chi2 = pyshtools.expand.SHExpandLSQ(ur, Vlat, Vlon, lmax=lmax)
    print('shape fitting accuracy:', chi2)

    ucoeff_interp = pyshtools.SHCoeffs.from_array(urcilm_interp)
    urgrid_interp = ucoeff_interp.expand('GLQ')

    lats = urgrid_interp.lats(); lons = urgrid_interp.lons();
    lats_circular = np.hstack(([90.], lats, [-90.]))
    xmesh = urgrid_interp.to_array().copy()
    if lons[-1] != 360.0:
        lons_circular = np.append(lons, 360)
    else:
        lons_circular = lons
        xmesh = xmesh[:, :-1]
    LONS, LATS = np.meshgrid(lons_circular, lats_circular)

    fpoints = np.zeros_like(LONS)
    fpoints[1:-1, :-1] = xmesh
    fpoints[ 0, :] = np.mean(xmesh[ 0, :], axis=0)  # not exact !
    fpoints[-1, :] = np.mean(xmesh[-1, :], axis=0)  # not exact !
    fpoints[1:-1, -1] = xmesh[:, 0]
    f_interp = RectBivariateSpline(lats_circular[::-1], lons_circular, fpoints[::-1, ], kx=1, ky=1)
    
    if plot_figure:
        fig, axs = plt.subplots(1,1,figsize=(6,2.5))
        im = axs.tripcolor(Vlon, Vlat, ur)
        fig.colorbar(im)
        axs.axis('equal')
        axs.set_xlim(0, 360)
        axs.set_ylim(-90, 90)
        fig, axs = plt.subplots(1,1,figsize=(6,2.5))
        ur_interp = f_interp.ev(Vlat, Vlon)
        im = axs.tripcolor(Vlon, Vlat, (ur_interp-ur))
        fig.colorbar(im)
        axs.axis('equal')
        axs.set_xlim(0, 360)
        axs.set_ylim(-90, 90)
        plt.show()
    
    return f_interp

def calculateTfv(Uvec, lJmax, Vp, Tfv, lat_weight=False):
    Xt = GLQCartCoord(lJmax) + SHVec2mesh(Uvec, lmax=lJmax, SphCoord=False, Complex=True)
    dist2mat = np.linalg.norm((Xt[..., np.newaxis, :] - Vp), axis=-1)
    arg_list_x = dist2mat.argmin(axis=-1)
    if lat_weight:
        latsdeg, lonsdeg = pyshtools.expand.GLQGridCoord(lJmax, extend=True)
        phi, theta = np.meshgrid(np.deg2rad(lonsdeg), np.deg2rad(90 - latsdeg))
        lat_weight = np.sin(theta)
    else:
        lat_weight = 1
    return Tfv[arg_list_x] * lat_weight

def usurf2umesh(u_surf, f_interp, lmax, X0surf=None, X0=None):
    if lmax is None:
        lmax = (np.sqrt(8*u_surf.size/2+1)-3)//4
    if X0surf is None:
        latsdeg, lonsdeg = pyshtools.expand.GLQGridCoord(lmax, extend=True)
        lon0, lat0 = np.meshgrid(lonsdeg, latsdeg)
        X0surf = np.stack([lat0, lon0], axis=-1)
    if X0 is None:
        X0 = GLQCartCoord(lmax)
    if u_surf.size == (lmax+1)*(2*lmax+1)*2:
        u_surf_reshape = u_surf.reshape(lmax+1, 2*lmax+1, 2)
    else:
        u_surf_reshape = u_surf.reshape(lmax+1, 2*lmax+2, 2)
    x_surf = X0surf + u_surf_reshape
    lat_x = x_surf[..., 0]; lon_x = x_surf[...,1];

    Theta_x = np.deg2rad(90 - lat_x); Phi_x = np.deg2rad(lon_x);
    R_x = f_interp.ev(lat_x, lon_x)+1
    x = np.stack(SphCoord_to_CartCoord(R_x, Theta_x, Phi_x), axis=-1)
    return (x - X0).flatten()

def dumesh_dus(u_surf, f_interp, lmax, *args, eps=1e-5, mode='forward'):
    u_surf = u_surf.reshape(lmax+1, 2*lmax+2, 2)
    ptr = np.array([+eps, 0])
    ptu = np.array([0, +eps])
    u_mesh_r = usurf2umesh(u_surf + ptr, f_interp, lmax, *args)
    u_mesh_u = usurf2umesh(u_surf + ptu, f_interp, lmax, *args)
    if mode == 'forward':
        pt0 = np.array([0, 0])
        u_mesh_0 = usurf2umesh(u_surf + pt0, f_interp, lmax, *args)
        dumesh_1 = (u_mesh_r - u_mesh_0)/eps
        dumesh_2 = (u_mesh_u - u_mesh_0)/eps
    elif mode == '2-points':
        ptl = np.array([-eps, 0])
        ptd = np.array([0, -eps])
        u_mesh_l = usurf2umesh(u_surf + ptl, f_interp, lmax, *args)
        u_mesh_d = usurf2umesh(u_surf + ptd, f_interp, lmax, *args)
        dumesh_1 = (u_mesh_r - u_mesh_l)/eps/2
        dumesh_2 = (u_mesh_u - u_mesh_d)/eps/2

    return sp.linalg.block_diag(*np.stack([dumesh_1, dumesh_2], axis=-1).reshape(-1, 3, 2))

def usurf2vec(u_surf, f_interp, lmax, X0surf=None, X0=None, Cmat=None, Dmat=None, mu0=300/3, nu0=0.499):
    if Cmat is None and Dmat is None:
        Cmat, Dmat = loadCoeffs(mu0, nu0, lmax, 'reg')
    umesh = usurf2umesh(u_surf, f_interp, lmax, X0surf=X0surf, X0=X0)
    if umesh.size == (lmax+1)*(2*lmax+1)*3:
        umesh_reshape = umesh.reshape(lmax+1, 2*lmax+1, 3)
    else:
        umesh_reshape = umesh.reshape(lmax+1, 2*lmax+2, 3)
    Uvec  = SHmesh2Vec(umesh_reshape, lmax=lmax)
    aK    = spsolve(Dmat, Uvec.T)
    Tvec  = Cmat.dot(aK)
    return Uvec, aK, Tvec

def usurf2Eel(u_surf, f_interp, lmax, X0surf=None, X0=None, Cmat=None, Dmat=None, mu0=300/3, nu0=0.499):
    Uvec, aK, Tvec = usurf2vec(u_surf, f_interp, lmax, X0surf=X0surf, X0=X0, Cmat=Cmat, Dmat=Dmat, mu0=mu0, nu0=nu0)
    return np.vdot(Uvec, Tvec).real*2*np.pi

def Tvec2Tres(Tvec, lmax, isTfv=None, lat_weights=np.array([1]), vert_weight=np.array([1]), norm_order=1):
    tmesh = SHVec2mesh(Tvec, lmax=lmax, SphCoord=False, Complex=True)
    tvalues = np.sum((tmesh*vert_weight[..., None]*lat_weights[..., None])**2, axis=-1) * isTfv
    if norm_order > 1:
        Tdist = np.linalg.norm(tvalues, ord=norm_order)
    else:
        Tdist = np.mean(tvalues)
    return Tdist

def genLmat(lmax, Cmat=None, Dmat=None, mu0=300/3, nu0=0.499):
    '''Tmesh = Lmat.dot(Umesh)'''
    if Cmat is None and Dmat is None:
        Cmat, Dmat = loadCoeffs(mu0, nu0, lmax, 'reg')
    meshsize = (lmax+1)*(2*lmax+2)*3
    du= np.identity(meshsize)
    L = np.empty((meshsize, meshsize))
    for i in range(meshsize):
        Uvec = SHmesh2Vec(du[:, i].reshape(lmax+1,2*lmax+2,3), lmax)
        Tvec = Uvec2Tvec(Uvec, Cmat, Dmat)
        L[:, i] = SHVec2mesh(Tvec, lmax=lmax, SphCoord=False, Complex=True).flatten()
    return L

def genSmat(lmax, Cmat=None, Dmat=None, mu0=300/3, nu0=0.499):
    '''Uvec = Smat.dot(Umesh)'''
    if Cmat is None and Dmat is None:
        Cmat, Dmat = loadCoeffs(mu0, nu0, lmax, 'reg')
    meshsize = (lmax+1)*(2*lmax+2)*3
    coefsize = (lmax+1)**2 * 3
    du= np.identity(meshsize)
    S = np.empty((coefsize, meshsize), dtype=complex)
    for i in range(meshsize):
        Uvec = SHmesh2Vec(du[:, i].reshape(lmax+1, 2*lmax+2, 3), lmax)
        S[:, i] = Uvec.flatten()
    return S

def usurf2dr(u_surf, f_interp, lmax, beta=1, norm_order=1, 
             X0surf=None, X0=None, isTfv=None, Cmat=None, Dmat=None, 
             mu0=300/3, nu0=0.499, lat_weights=np.array([1]), vert_weight=np.array([1]),
             eps=1e-5, mode='forward', JacMat=None):
    Uvec, aK, Tvec = usurf2vec(u_surf, f_interp, lmax, Cmat=Cmat, Dmat=Dmat, mu0=mu0, nu0=nu0)
    Tdist = Tvec2Tres(Tvec, lmax, isTfv=isTfv, lat_weights=lat_weights, vert_weight=vert_weight, norm_order=norm_order)
    Eel = np.vdot(Uvec, Tvec).real*2*np.pi
    return beta*Tdist + Eel

def grad_usurf2dr(u_surf, f_interp, lmax, beta=1, norm_order=1, 
                  X0surf=None, X0=None, isTfv=None, Cmat=None, Dmat=None, 
                  mu0=300/3, nu0=0.499, lat_weights=np.array([1]), vert_weight=np.array([1]), 
                  eps=1e-5, mode='forward', JacMat=None):
    umesh = usurf2umesh(u_surf, f_interp, lmax, X0surf=X0surf, X0=X0)
    dum_dus = dumesh_dus(u_surf, f_interp, lmax, X0surf, X0, eps=eps, mode='forward')
    if JacMat is None:
        raise TypeError('JacMat must be provided')
    JacMat = beta*JacMat[0] + JacMat[1]
    return np.ravel(np.dot(JacMat.dot(umesh).real, dum_dus))

def usurf2dr2(u_surf, f_interp, lmax, beta=1, norm_order=1, 
             X0surf=None, X0=None, isTfv=None, Cmat=None, Dmat=None, 
             mu0=300/3, nu0=0.499, lat_weights=np.array([1]), vert_weight=np.array([1]),
             eps=1e-5, mode='forward', JacMat=None, gamma=1):
    Uvec, aK, Tvec = usurf2vec(u_surf, f_interp, lmax, Cmat=Cmat, Dmat=Dmat, mu0=mu0, nu0=nu0)
    Tdist = Tvec2Tres(Tvec, lmax, isTfv=isTfv, lat_weights=lat_weights, vert_weight=vert_weight, norm_order=norm_order)
    Eel = np.vdot(Uvec, Tvec).real*2*np.pi
    ldamp_hi = lmax; ldamp_lo = lmax - 5;
    lv, _ = LM_list(lmax); lv_ones = np.ones_like(lv);
    lv_lim = np.minimum(np.maximum(lv, ldamp_lo), ldamp_hi)
    ldamp = (np.maximum(lv_lim-ldamp_lo, 0) / (ldamp_hi - ldamp_lo))**1
    Tvec_mod = (Tvec.reshape(3, -1) * ldamp).flatten()
    penalty = np.vdot(Tvec_mod, Tvec_mod).real
    return beta * Tdist + Eel + gamma * penalty

def grad_usurf2dr2(u_surf, f_interp, lmax, beta=1, norm_order=1, 
                  X0surf=None, X0=None, isTfv=None, Cmat=None, Dmat=None, 
                  mu0=300/3, nu0=0.499, lat_weights=np.array([1]), vert_weight=np.array([1]), 
                  eps=1e-5, mode='forward', JacMat=None, gamma=1):
    umesh = usurf2umesh(u_surf, f_interp, lmax, X0surf=X0surf, X0=X0)
    dum_dus = dumesh_dus(u_surf, f_interp, lmax, X0surf, X0, eps=eps, mode='forward')
    if JacMat is None:
        raise TypeError('JacMat must be provided')
    JacMat = beta * JacMat[0] + JacMat[1] + gamma*JacMat[2]
    return np.ravel(np.dot(JacMat.dot(umesh).real, dum_dus))
