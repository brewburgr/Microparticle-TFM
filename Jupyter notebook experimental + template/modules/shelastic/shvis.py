"""
shvis
=====
Functions for visualizing spherical harmonic solutions
"""

import numpy as _np
import scipy.sparse as _spm
import matplotlib.pyplot as _plt
from mpl_toolkits.mplot3d import Axes3D
from matplotlib import cm
import pyshtools as _psh
from shelastic.shutil import SHVec2mesh

def plotfv(fv, figsize=(10,5), colorbar=True, show=True, vrange=None, cmap='viridis', lonshift=0):
    """Plot a color map of a 2d function

    Usage
    -----
    fig, ax = plotfv(fv, [figsize, colorbar, show, vrange, cmap, lonshift])

    Returns
    -------
    fig,ax : matplotlib figure and axis instances

    Parameters
    ----------
    fv : ndarray, shape (nlat, nlon)
        2-D numpy array of the gridded data, where nlat and nlon are the
        number of latitudinal and longitudinal bands, respectively.
    figsize : size of the figure, optional, default = (10, 5)
    colorbar : bool, optional, default = True
        If True (default), plot the colorbar along with the map
    show : bool, optional, default = True
        If True, plot the image to the screen.
    vrange : 2-element tuple, optional
        The range of the colormap, default is (min, max)
    cmap : string, default = 'viridis'
        Name of the colormap, see matplotlib
    lonshift : float, in degree, default = 0
        Shift the map along longitude direction by lonshift degree

    """
    if lonshift is not None:
        fv = _np.roll(fv, _np.round(fv.shape[1]*lonshift/360).astype(int), axis=1)
    if vrange is None:
        fmax, fmin = fv.max(), fv.min()
    else:
        fmin, fmax = vrange
    fcolors = (fv - fmin)/(fmax - fmin)    # normalize the values into range [0, 1]
    fcolors[fcolors<0]=0
    fig0 = _plt.figure(figsize=figsize)
    ax0 = fig0.add_subplot(111)
    cax0 = ax0.imshow(fv, extent=(0, 360, -90, 90), cmap=cmap, vmin=fmin, vmax=fmax, interpolation='nearest')
    ax0.set(xlabel='longitude', ylabel='latitude')
    if colorbar:
        fig0.colorbar(cax0)
    if show:
        _plt.show()
    return fig0, ax0

def vismesh(xmesh, cmap='viridis', show=False, SphCoord=True, 
            config_quiver=(2, 4, 'k', 1), n_vrange=None, s_vrange=None,
            lonshift=0, figsize=(10, 5)):
    """Plot color map of mesh representation of SH vectors

    Usage
    -----
    fig, ax = vismesh(xmesh, [cmap, show, SphCoord, config_quiver, 
                              n_vrange, s_vrange, lonshift, figsize])

    Returns
    -------
    fig,ax : matplotlib figure and axis instances

    Parameters
    ----------
    xmesh : ndarray, dimension (lmax+1, 2*lmax+1, nd)
        Mesh point representation of SH vector
    cmap : string, default = 'viridis'
        Name of the colormap, see matplotlib
    show : bool, optional, default = True
        If True, plot the image to the screen.
    SphCoord : bool, optional, default = True
        If True, xmesh is in spherical coordinates, plot normal and shear
        If False, xmesh is in Cartesian coordinates, plot x, y, z
    config_quiver : (st, dq, color, scale)
        Configuration settings for quiver on the shear plot.
        The mesh point is down sampled as xmesh[::dq, st::dq]
        color is the color of the quiver
        scale is the scale of the quiver, larger scale value corresponds to shorter arrows
    n_vrange,s_vrange : 2-element tuple, optional
        The range of the colormap for normal and shear, only being used in spherical coordinates.
    lonshift : float, in degree, default = 0
        Shift the map along longitude direction by lonshift degree
    figsize : tuple, optional, default = (10, 5)
        Size of the figure

    """
    nlat, nlon, nd = xmesh.shape
    lmax_plot = nlat - 1
    if SphCoord:
        fig = [None for _ in range(2)]
        ax = [None for _ in range(2)]
        xshear= _np.linalg.norm(xmesh[...,1:], axis=-1)
        
        fig[0], ax[0] = plotfv(xmesh[...,0], show=show, cmap=cmap,vrange=n_vrange,
                               lonshift=lonshift, figsize=figsize)
        ax[0].set_title('norm')
        
        fig[1], ax[1] = plotfv(xshear, show=show, cmap='Reds', lonshift=lonshift, figsize=figsize, vrange=s_vrange)
        latsdeg, lonsdeg = _psh.expand.GLQGridCoord(lmax_plot, extend=True)
        lons, lats = _np.meshgrid(lonsdeg, latsdeg)
        xshift = _np.roll(xmesh, _np.round(lons.shape[1]*lonshift/360).astype(int), axis=1)
        st, dq, color, scale = config_quiver
        ax[1].quiver(lons[::dq,st::dq], lats[::dq,st::dq], 
                     xshift[::dq,st::dq,2], -xshift[::dq,st::dq,1], 
                     color=color, scale=scale)
        ax[1].set_title('shear')
    else:
        fig = [None for _ in range(3)]
        ax = [None for _ in range(3)]
        titlestr = ('x', 'y', 'z')
        for k in range(3):
            fig[k], ax[k] = plotfv(xmesh[...,k], show=show, cmap=cmap, lonshift=lonshift, figsize=figsize)
            ax[k].set_title('$'+titlestr[k]+'$')
    return fig, ax

def visSHVec(xvec, lmax_plot=None, cmap='viridis', show=True, 
             SphCoord=True, config_quiver=(2, 4, 'k', 1), n_vrange=None, s_vrange=None,
             lonshift=0, Complex=False, figsize=(10, 5)):
    """Plot color map of coefficient representation of SH vectors

    Returns
    -------
    fig,ax : matplotlib figure and axis instances

    Parameters
    ----------
    xvec : ndarray, complex, dimension (nd*(lmax+1)^2, )
        SH coefficient representation of the SH vector
        For now the function only supports nd = 3;
    cmap : string, default = 'viridis'
        Name of the colormap, see matplotlib
    show : bool, optional, default = True
        If True, plot the image to the screen.
    SphCoord : bool, optional, default = True
        If True, xmesh is in spherical coordinates, plot normal and shear
        If False, xmesh is in Cartesian coordinates, plot x, y, z
    config_quiver : (st, dq, color, scale)
        Configuration settings for quiver on the shear plot.
        The mesh point is down sampled as xmesh[::dq, st::dq]
        color is the color of the quiver
        scale is the scale of the quiver, larger scale value corresponds to shorter arrows
    n_vrange,s_vrange : 2-element tuple, optional
        The range of the colormap for normal and shear, only being used in spherical coordinates.
    lonshift : float, in degree, default = 0
        Shift the map along longitude direction by lonshift degree
    Complex : bool, only Complex=True is supported
        Whether the output is real or complex spherical harmonics
    figsize : tuple, optional, default = (10, 5)
        Size of the figure

    """
    if lmax_plot is None:
        lmax_plot = _np.sqrt(xvec.size//3).astype(np.int) - 1
    xmesh = SHVec2mesh(xvec, lmax=lmax_plot, SphCoord=SphCoord, Complex=Complex)
    fig, ax = vismesh(xmesh, cmap=cmap, show=False, SphCoord=SphCoord, config_quiver=config_quiver,
                      n_vrange=n_vrange, s_vrange=s_vrange, lonshift=lonshift, figsize=figsize)
    if show:
        _plt.show()
    return fig, ax

def visSH3d(xmesh, cmesh=None, r0=1, lmax_plot=None, cmap='RdBu', colorbar=False,
            figsize=(16,16), show=True, filename=None, vmin=None, vmax=None,
            elevation=0, azimuth=0, surface=False, color=None, fig=None, ax=None, alpha=None, return_data=False):
    """Plot reconstructed spherical shape and traction colored 3d plot

    Returns
    -------
    fig,ax : matplotlib figure and axis instances

    Parameters
    ----------
    xmesh : ndarray, dimension (lmax+1, 2*lmax+1, nd)
        Mesh point representation of displacement SH vector
    cmesh : ndarray, dimension (lmax+1, 2*lmax+1, nd), optional
        If used, color the 3d shape with the mesh point representation
    r0 : float
        Radius of the original spherical shape
    lmax_plot : int, optional
        If used, the mesh is truncated to the given lmax;
        If None, determined by the mesh size
    figsize : tuple, optional, default = (16, 16)
        Size of the figure
    show : bool
        If True, plt.show() is called
    filename : string, optional
        If used, the figure will be saved into a file
    elevation,azimuth : float
        Elevation and azimuth of the 3d view
    surface : bool
        If True, the surface of the sphere will be plotted;
        otherwise, only the grid points will be plotted.
    color : color object, optional
        Surface color. If None, use cmesh to determine the colors.
    """
    if lmax_plot is None:
        lmax_plot = xmesh.shape[0] - 1
    lats, lons = _psh.expand.GLQGridCoord(lmax_plot, extend=True)
    nlat = lats.size; nlon = lons.size;

    lats_circular = _np.hstack(([90.], lats, [-90.]))
    lons_circular = _np.append(lons, [lons[0]])
    u = _np.radians(lons_circular)
    v = _np.radians(90. - lats_circular)
    normvec = _np.zeros((nlat+2, nlon+1, 3))
    normvec[...,0] = _np.sin(v)[:, None] * _np.cos(u)[None, :]
    normvec[...,1] = _np.sin(v)[:, None] * _np.sin(u)[None, :]
    normvec[...,2] = _np.cos(v)[:, None] * _np.ones_like(lons_circular)[None, :]

    upoints = _np.zeros((nlat + 2, nlon + 1, 3))
    upoints[1:-1, :-1, :] = xmesh
    upoints[0, :, :] = _np.mean(xmesh[0,:,:], axis=0)  # not exact !
    upoints[-1, :, :] = _np.mean(xmesh[-1,:,:], axis=0)  # not exact !
    upoints[1:-1, -1, :] = xmesh[:, 0, :]
    upoints *= r0
    
    x = r0 * _np.sin(v)[:, None] * _np.cos(u)[None, :]  + upoints[..., 0]
    y = r0 * _np.sin(v)[:, None] * _np.sin(u)[None, :] + upoints[..., 1]
    z = r0 * _np.cos(v)[:, None] * _np.ones_like(lons_circular)[None, :] + upoints[...,2]

    if color is None:
        if cmesh is None:
            magn_point = _np.sum(normvec * upoints, axis=-1)
        else:
            tpoints = _np.zeros((nlat + 2, nlon + 1, 3))
            tpoints[1:-1, :-1, :] = cmesh
            tpoints[0, :, :] = _np.mean(cmesh[0,:,:], axis=0)  # not exact !
            tpoints[-1, :, :] = _np.mean(cmesh[-1,:,:], axis=0)  # not exact !
            tpoints[1:-1, -1, :] = cmesh[:, 0, :]
            magn_point = _np.sum(normvec * tpoints, axis=-1)
        magn_face = 1./4. * (magn_point[1:, 1:] + magn_point[:-1, 1:] +
                             magn_point[1:, :-1] + magn_point[:-1, :-1])
        magnmax_face = _np.max(_np.abs(magn_face))
        magnmax_point = _np.max(_np.abs(magn_point))
        if vmin is None:
            vmin = -magnmax_face/2.
        if vmax is None:
            vmax =  magnmax_face/2.
        norm = _plt.Normalize(vmin, vmax, clip=True)
        cmap = _plt.get_cmap(cmap)
        colors = cmap(norm(magn_face.flatten()))
        colors = colors.reshape(nlat + 1, nlon, 4)

    if fig is None and ax is None:
        fig = _plt.figure(figsize=figsize)
        ax3d = fig.add_subplot(1, 1, 1, projection='3d')
    else:
        ax3d = ax
    if surface:
        if color is None:
            surf = ax3d.plot_surface(x, y, z, rstride=1, cstride=1, facecolors=colors, alpha=alpha)
        else:
            surf = ax3d.plot_surface(x, y, z, rstride=1, cstride=1, color=color, alpha=alpha)
        if colorbar:
            fig.colorbar(cm.ScalarMappable(norm=norm, cmap=cmap), ax=ax3d, shrink=0.6)
    else:
        ax3d.scatter(x, y, z, alpha=alpha)
    ax3d.view_init(elev=elevation, azim=azimuth)
    ax3d.tick_params(labelsize=16)

    if filename is not None:
        _plt.tight_layout()
        fig.savefig(filename,transparent=True)
    if show:
        _plt.show(block=True)
    if return_data:
        return _np.stack([x, y, z], axis=-1), tpoints, [lats_circular, lons_circular]
    else:
        return fig, ax3d

def visualize_Cmat(Csub, precision=1e-8, m_max=3):
    '''Visualizing coefficient matrix
    
    Parameters
    ----------
    Csub : lil_matrix, complex
        Complex spherical harmonic coefficient matrix
    precision : float
        below which the values are considered noise
    m_max : deprecated, always use 3

    Returns
    -------
    fig,ax : matplotlib figure and axis instances
    
    '''
    fig, ax = _plt.subplots(1, 1)
    ax.spy(Csub, precision=precision, markersize = 3)
    mode_sub = int(_np.sqrt(Csub.shape[1]/m_max))-1
    lmax_sub = int(_np.sqrt(Csub.shape[0]/3))-1
    print(mode_sub, lmax_sub)
    lsy = _np.arange(0, lmax_sub+1)
    lsx = _np.arange(0, mode_sub+1)

    x_range = [0, Csub.shape[1]]
    y_range = [0, Csub.shape[0]]
    # traction direction dividing line:
    for i in range(1, 3+1):
        y_divide = y_range[1]*i/3
        ax.plot(x_range, _np.ones(2)*y_divide-0.5)
        y_text = y_divide - y_range[1]/m_max/2
        #_plt.text(-mode_sub-5, y_text, '$T_'+str(i)+'$', fontsize=24)
    # mode dividing line:
    for i in range(1, m_max+1):
        x_divide = x_range[1]*i/m_max
        ax.plot(_np.ones(2)*x_divide-0.5, y_range)
        x_text = x_divide - x_range[1]/3/2 - 1
        #_plt.text(x_text, -lmax_sub, '$\\psi_'+str(i)+'$', fontsize=24)

    # ticks for different traction directions
    ticks_y = _np.array([])
    for i in range(3):
        ticks_Ti = i*(lmax_sub+1)**2+lsy**2
        ticks_y = _np.hstack((ticks_y, ticks_Ti))
    ax.yticks(ticks_y, _np.tile(lsy, 3))
    # ticks for different modes
    ticks_x = _np.array([])
    for i in range(m_max):
        ticks_Psi_i = i*(mode_sub+1)**2+lsx**2
        ticks_x = _np.hstack((ticks_x, ticks_Psi_i))
    ax.xticks(ticks_x, _np.tile(lsx, m_max))
    if m_max == 4:
        ax.title('Solution A+B', fontsize=24)
    else:
        ax.title('Solution B', fontsize=24)
    return fig, ax
