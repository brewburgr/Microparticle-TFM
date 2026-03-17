"""
shutil
======
utility functions
"""

import numpy as _np
import scipy.sparse as _spm
import pyshtools as _psh

# Functions for converting between Cartesian coordinates and spherical coordinates
def CartCoord_to_SphCoord(X, Y, Z):
    """translate Cartesian coordinates into spherical coordinates
    
        Parameters
        ----------
        X,Y,Z : ndarray
            make sure X, Y, Z have the same dimensions
        
        Returns
        -------
        R,THETA,PHI : ndarray
            same size as X, Y, Z.
    
    """
    R = _np.sqrt(X**2 + Y**2 + Z**2)
    THETA = _np.arccos(Z/R)
    PHI = _np.arctan2(Y, X)
    PHI[PHI < 0] += 2*_np.pi
    return (R, THETA, PHI)

def SphCoord_to_CartCoord(R, THETA, PHI):
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
    Z = R * _np.cos(THETA)
    X = R * _np.sin(THETA) * _np.cos(PHI)
    Y = R * _np.sin(THETA) * _np.sin(PHI)
    return (X, Y, Z)

def TransMat(t_mesh=None, p_mesh=None, lJmax=None):
    """coordinate translation matrix for vectors and tensors
    
        Given theta, phi mesh, or lJmax value, return the translation
        matrix for vector and tensor. i.e. f = Q.dot(x)
        where x is in Cartesian coordinates and f is in spherical
        coordinates. If you select to provide lJmax, the generated
        coordinate is [GLQ coordinates](https://shtools.oca.eu/shtools/pyglqgridcoord.html)

        Parameters
        ----------
        t_mesh,p_mesh : ndarray
            theta and phi mesh with same dimension.
        lJmax : optional, int
            if t_mesh and p_mesh not provided, generate theta and
            phi mesh based on GLQ gridding.
        
        Returns
        -------
        Q : ndarray, dimension t_mesh.shape + (3, 3)
            3x3 matrix corresponding to the given mesh.

    """
    # if t_mesh and p_mesh not provide, use lJmax to generate the mesh
    if (t_mesh is None) and (p_mesh is None) and (lJmax is not None):
        latglq, longlq = _psh.expand.GLQGridCoord(lJmax, extend=True)
        theta = _np.radians(90-latglq)
        phi = _np.radians(longlq)
        p_mesh, t_mesh = _np.meshgrid(phi, theta)
    
    Q = _np.zeros(p_mesh.shape+(3,3))
    Q[...,0,0] = _np.sin(t_mesh)*_np.cos(p_mesh)
    Q[...,0,1] = _np.sin(t_mesh)*_np.sin(p_mesh)
    Q[...,0,2] = _np.cos(t_mesh)
    Q[...,1,0] = _np.cos(t_mesh)*_np.cos(p_mesh)
    Q[...,1,1] = _np.cos(t_mesh)*_np.sin(p_mesh)
    Q[...,1,2] =-_np.sin(t_mesh)
    Q[...,2,0] =-_np.sin(p_mesh)
    Q[...,2,1] = _np.cos(p_mesh) 

    return Q

def GLQCartCoord(lmax):
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
    latsdeg, lonsdeg = _psh.expand.GLQGridCoord(lmax, extend=True)
    lon = _np.deg2rad(lonsdeg)
    colat = _np.deg2rad(90-latsdeg)
    PHI, THETA = _np.meshgrid(lon, colat)
    R = _np.ones_like(PHI)
    X,Y,Z = SphCoord_to_CartCoord(R, THETA, PHI)
    return _np.stack([X,Y,Z], axis=-1)

# Routines to calculate index for translating between cilm and vector
def lm2L(l, m):
    """translate 2d indices (l, m) to 1d index L"""
    return l**2 + (l + m)

def L2lm(L):
    """translate 1d index L to 2d indices (l, m)"""
    l = _np.floor(_np.sqrt(L)).astype(int)
    m = L - l**2 - l
    return (l, m)

def LM_list(lmax):
    """return l and m indices for all SH modes below lmax
    
        Parameters
        ----------
        lmax : int
            maximum SH l-order
        
        Returns
        -------
        l : int, dimension ((lmax+1)^2, )
            a list of l indices for all the SH indices below lmax
        m : int, dimension ((lmax+1)^2, )
            a list of m indices for all the SH indices below lmax

    """
    return L2lm(_np.arange((lmax+1)**2, dtype=int))

def ILM_list(lmax):
    """return (i, l, m) indices of Cilm array for all SH modes below lmax"""
    l, m = LM_list(lmax)
    i = (m < 0).astype(int)
    return (i, l, _np.abs(m).astype(int))

# Given lmax, convert between (l,m,k) degrees and 1d mode index K
def lmk2K(l, m, k, lmax):
    """convert spherical harmonic vector indices to vector index
    
        Examples
        --------
        >>> lmk2K(2, 1, 1, 3)
        23
        
    """
    n = (lmax+1)**2
    return lm2L(l, m) + k*n
def K2lmk(K, lmax):
    """convert vector index to spherical harmonic vector indices
    
        Examples
        --------
        >>> lmk2K(18, 3)
        (1, 0, 1)
    """
    n = (lmax+1)**2
    k = _np.floor(K/n).astype(int)
    l, m = L2lm(K%n)
    return l, m, k

def l_coeffs(lmax):
    """This function creates a coefficient array with their l order

        Examples
        --------
        >>> l_coeffs(3)
        array([[[0, 0, 0, 0],
                [1, 1, 1, 1],
                [2, 2, 2, 2],
                [3, 3, 3, 3]],
               [[0, 0, 0, 0],
                [1, 1, 1, 1],
                [2, 2, 2, 2],
                [3, 3, 3, 3]]])

    """
    l_list = _np.arange(lmax + 1)
    return _np.broadcast_to(l_list[_np.newaxis,:,_np.newaxis], (2, lmax+1, lmax+1))
def m_coeffs(lmax):
    """This function creates a coefficient array with their m order

        Examples
        --------
        >>> m_coeffs(3)
        array([[[0, 1, 2, 3],
                [0, 1, 2, 3],
                [0, 1, 2, 3],
                [0, 1, 2, 3]],
               [[0, 1, 2, 3],
                [0, 1, 2, 3],
                [0, 1, 2, 3],
                [0, 1, 2, 3]]])


    """
    m_list = _np.arange(lmax + 1)
    return _np.broadcast_to(m_list[_np.newaxis,_np.newaxis, :], (2, lmax+1, lmax+1))

# Routines for matrix shape transformation
def SHCilmToVector(cilm, lmax=None):
    """This function convert a SH array to an 1d vector

        Examples
        --------
        >>> SHCilmToVector(np.array([[[1+1j, 0, 0, 0],
                                      [2+2j, 3+3j, 0, 0],
                                      [4+4j, 5+5j, 6+6j, 0],
                                      [7+7j, 8+8j, 9+9j, 10+10j]],
                                     [[0, 0, 0, 0],
                                      [0, 3-3j, 0, 0],
                                      [0, 5-5j, 6-6j, 0],
                                      [0, 8-8j, 9-9j, 10-10j]]]))
        array([ 1. +1.j,  3. -3.j,  2. +2.j,  3. +3.j,  6. -6.j,  5. -5.j,
        4. +4.j,  5. +5.j,  6. +6.j, 10.-10.j,  9. -9.j,  8. -8.j,
        7. +7.j,  8. +8.j,  9. +9.j, 10.+10.j])

    """
    clmax = cilm.shape[1] - 1
    if lmax is None:
        lmax = clmax
    i, l, m = ILM_list(clmax)
    vec = _np.zeros((lmax+1)**2, dtype=cilm.dtype)
    vec[:(clmax+1)**2] = cilm[i,l,m]
    return vec

def SHVectorToCilm(vec, lmax=None):
    """This function convert an 1d vector to a SH array

        Examples
        --------
        >>> SHVectorToCilm(np.array([1+1j, 2+2j, 3+3j, 4+4j, 5+5j, 6+6j, 7+7j, 8+8j, 9+9j]))
        array([[[1.+1.j, 0.+0.j, 0.+0.j],
                [3.+3.j, 4.+4.j, 0.+0.j],
                [7.+7.j, 8.+8.j, 9.+9.j]],
               [[0.+0.j, 0.+0.j, 0.+0.j],
                [0.+0.j, 2.+2.j, 0.+0.j],
                [0.+0.j, 6.+6.j, 5.+5.j]]])

    """
    # check whether vec has a valid length
    if lmax is None:
        lmax = _np.around(_np.sqrt(len(vec))).astype(int) - 1
        if (lmax+1)**2 != len(vec):
            print('invalid length')
            return
    Lmax = (lmax+1)**2
    cilm = _np.zeros((2, lmax+1, lmax+1), dtype=vec.dtype)
    i, l, m = ILM_list(lmax)
    cilm[i, l, m] = vec[:Lmax]
    return cilm

def sparse_mode(M_K, lmax=None, etol=1e-8):
    """create sparse representation of SH vector or tensor
    
        compress a vector or tensor of SH coefficients into 1d sparse vector
    
        Parameters
        ----------
        M_K : complex, dimension (2, lmax+1, lmax+1), (2, lmax+1, lmax+1, 3) or (2, lmax+1, lmax+1, 3, 3)
            SH scalar, vector or tensor, every element is a SH-coefficient array
            of maximum l-order
        lmax : int
            maximum SH l-order
        etol : float
            coefficients below etol are considered 0

        Returns
        -------
        scipy lil sparse matrix, dimension (d*(lmax+1)^2, 1)
            a compressed sparse matrix representation of d-element SH vector

    """
    clmax = M_K.shape[1] - 1
    if lmax is None:
        lmax = clmax
    MK_mod = M_K.reshape(*M_K.shape[:3], -1)
    d = MK_mod.shape[-1]
    mode = _np.empty((d, (lmax+1)**2), dtype=M_K.dtype)
    for i in range(d):
        mode[i, :] = SHCilmToVector(MK_mod[..., i], lmax=lmax)
    mode = mode.reshape((mode.size, 1))
    SPM = _spm.lil_matrix((mode.size, 1), dtype=mode.dtype)
    filter_idx = _np.abs(mode)>etol
    SPM[filter_idx] = mode[filter_idx]
    return SPM

def dense_mode(mode, d, lmax):
    """extract SH tensor from 1d sparse representation
    
        extract a SH scalar(1x1), vector(3x1) or SH tensor(3x3) from the sparse
        representation.
    
        Parameters
        ----------
        mode : scipy lil sparse matrix, dimension ((3^d)*(lmax+1)^2, 1)
            a compressed sparse matrix representation of SH vector or tensor
        d : int
            value from 0, 1, or 2. representing scalar, vector and tensor correspondingly
        lmax : int
            maximum SH l-order

        Returns
        -------
        M : complex, dimension (2, lmax+1, lmax+1), (2, lmax+1, lmax+1, 3) or (2, lmax+1, lmax+1, 3, 3)
            SH scalar, vector or tensor, every element is a SH-coefficient array
            of maximum l-order

    """
    Lmax = (lmax+1)**2
    if d==0:
        M = SHVectorToCilm(mode)
    elif d==1:
        M = []
        for i in range(3):
            M.append(SHVectorToCilm(mode[Lmax*i:Lmax*(i+1)]))
        M = _np.stack(M, axis=-1)
    elif d==2:
        M = []
        for i in range(3):
            Mi = []
            for j in range(3):
                Mi.append(SHVectorToCilm(mode[Lmax*(i*3+j):Lmax*(i*3+j+1)]))
            M.append(_np.stack(Mi, axis=-1))
        M = _np.stack(M, axis=-2)

    return M

# Converting between SH coefficients and values on mesh points for SH vector (3x1)

def SHmesh2Vec(xmesh, lmax=None, Complex=True):
    """Transform a mesh point representation of SH vector to SH coefficients
    
        Spherical harmonics transform components of a SH 3-vector from GLQ mesh grid
        to SH coefficients, for now it only supports GLQ-mesh.

        Parameters
        ----------
        xmesh : ndarray, dimension (nlat, nlon, nd)
            The mesh is in Cartesian coordinates
            For GLQ mesh: nlat = lmax + 1; nlon = 2*lmax + 1;
            For now the function only supports nd = 3;
        lmax : int
            maximum SH l-order
        Complex : bool, only Complex=True is supported
            Whether the output is real or complex spherical harmonics.

        Returns
        -------
        ndarray, complex, dimension (nd*(lmax+1)^2, )
            Coefficient representation of SH vector

    """
    nlat, nlon, nd = xmesh.shape
    if lmax is None:
        lmax = nlat - 1
    if not Complex:
        print('SHmesh2Vec: real spherical harmonic not supported!')
    else:
        xmesh = xmesh.astype(complex)
        xvec = _np.empty((nd, (lmax+1)**2), dtype=complex)
    for k in range(nd):
        xgrid = _psh.SHGrid.from_array(xmesh[...,k], grid='GLQ')
        xcilm = xgrid.expand()
        xvec[k, :] = SHCilmToVector(xcilm.to_array(), lmax=lmax)
    return xvec.flatten()

def SHVec2mesh(xvec, lat=None, lon=None, lmax=None, SphCoord=True, Complex=False):
    """Transform Coefficients of SH vector to mesh point representation
    
        Spherical harmonics transform components of a SH 3-vector from SH coefficients
        to mesh grids (inCartesian or spherical coordinates), for now it supports
        GLQ-mesh or irregular mesh (given lat and lon values).

        Parameters
        ----------
        xvec : ndarray, complex, dimension (nd*(lmax+1)^2, )
            SH coefficient representation of the SH vector
            For now the function only supports nd = 3;
        lat,lon : array-like, degree
            A list of lattitude and longitude location of irregular mesh points
        lmax : int
            maximum SH l-order
        SphCoord : bool
            Whether the mesh grid is in spherical or Cartesian coordinates
        Complex : bool, only Complex=True is supported
            Whether the output is real or complex spherical harmonics

        Returns
        -------
        ndarray, dimension (lmax+1, 2*lmax+1, nd)
            Mesh point representation of SH vector

    """
    if lmax is None:
        lmax = (_np.sqrt(xvec.size/3) - 1).astype(int)
    cvec = xvec.reshape(3, -1)
    nvec = cvec.shape[1]
    xmesh= [None for _ in range(3)]
    for k in range(3):
        if Complex:
            cext = _np.zeros((lmax+1)**2, dtype=complex)
            cext[:nvec] = cvec[k, :(lmax+1)**2]
            cilm = SHVectorToCilm(cext)
        else:
            cext = _np.zeros((lmax+1)**2)
            cext[:nvec] = cvec[k, :(lmax+1)**2]
            cilm = _psh.shio.SHVectorToCilm(cext)
        coeffs = _psh.SHCoeffs.from_array(cilm)
        if (lat is None) and (lon is None):
            grid = coeffs.expand('GLQ')
            xmesh[k] = grid.to_array().real
        else:
            grid = coeffs.expand(lon=lon, lat=lat)
            xmesh[k] = grid.real
    xmesh = _np.stack(xmesh, axis=-1)
    if SphCoord:
        if (lat is None) and (lon is None):
            Q = TransMat(lJmax=lmax)
        else:
            Q = TransMat(t_mesh=_np.radian(90-lat), p_mesh=_np.radian(lon))
        xmesh = _np.sum(Q*xmesh[...,_np.newaxis,:], axis=-1)
    return xmesh.real

# Temporary alternative for SHComplexCoeffs.expand(lat=lat, lon=lon)
def eval_GridC(coeff, latin, lonin, rin=1.0, lmax_calc=None, norm=None, shtype=None):
    """Evaluate values of solid spherical harmonics on grids
    
        evaluate Ilm and Rlm given lattitude and longitude list of the grid points
    
        Parameters
        ----------
        coeff : SHCoeffs
            SHTOOLS coefficient class
        latin,lonin : array_like, same size
            lattitude and longitude list of grids for evaluating values
        rin : float or array_like (same size as latin and lonin)
            If float, evaluate spherical harmonics on sphere surface with radius rin;
            If array_like, evaluate on grid points
        lmax_calc : int
            Maximum l order to evaluate, the orders greater lmax_calc are truncated.
        norm : string, ['4pi', 'schmidt', 'ortho'], default same as coeff
            Normalization convention
        shtype : string, ['irr', 'reg']
            Type of solid spherical harmonics

        Returns
        -------
        values : complex, same dimension as latin and lonin
            Evaluated values on grids

    """
    if shtype == None:
        C_n = 0.0
    elif shtype == 'irr':
        C_n = -l_coeffs(coeff.lmax)-1
    elif shtype == 'reg':
        C_n = l_coeffs(coeff.lmax)
    elif type(shtype) is int:
        C_n = shtype
    else:
        print('invalid shtype (irr, reg, float)')
    if norm == None: # normalization not given, use the coefficient norm convention
        if coeff.normalization == '4pi':
            norm = 1
        elif coeff.normalization == 'schmidt':
            norm = 2
        elif coeff.normalization == 'ortho':
            norm = 4
    if lmax_calc == None:
        lmax_calc = coeff.lmax
    if (type(rin) is _np.ndarray) or (type(rin) is list):
        values = _np.empty_like(latin, dtype=complex)
        for v, latitude, longitude, radius in _np.nditer([values, latin, lonin, rin], op_flags=['readwrite']):
            C_r = radius**C_n
            v[...] = _psh.expand.MakeGridPointC(C_r*coeff.to_array(),
                                                lat=latitude, lon=longitude, 
                                                lmax=lmax_calc, norm=norm, 
                                                csphase=coeff.csphase)
    else:
        values = _np.empty_like(latin, dtype=complex)
        C_r = rin**C_n
        for v, latitude, longitude in _np.nditer([values, latin, lonin], op_flags=['readwrite']):
            v[...] = _psh.expand.MakeGridPointC(C_r*coeff.to_array(),
                                                lat=latitude, lon=longitude, 
                                                lmax=lmax_calc, norm=norm, 
                                                csphase=coeff.csphase)
    return values
