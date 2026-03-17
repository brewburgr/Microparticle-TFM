"""
shgrad
======
Calculate gradient of a set of spherical harmonic coefficients in terms of complex 
spherical harmonic coefficients
"""

import numpy as _np
from scipy.special import factorial
import pyshtools as _psh
from shelastic.shutil import l_coeffs, LM_list

def SHMultiplyC(sh1, sh2, lmax_calc=100, norm='4pi', csphase=1):
    """Calculate the product of two sets of complex spherical harmonic coefficients
    
    The function will take two sets of complex spherical harmonic coefficients `sh1`
    and `sh2`, return the product of the two functions represented in spherical 
    harmonics. The returned coefficient array is the spherical harmonic representation
    of the product of `sh1` and `sh2`.

    Parameters
    ----------

    sh1,sh2 : complex, dimension [2, lmax + 1, lmax + 1]
        2 set of SH coefficients with the same size, following SHTOOLS conventions.
    lmax_calc : int
        not used
    norm : optional, ['4pi', 'ortho', 'schmidt', 'unnorm'], default='4pi'
        Normalization of the spherical harmonic functions
    csphase : optional, integer, [1, -1], default=1
        Condon-Shortley phase
    
    Returns
    -------
    complex, dimension (2, lmax + 1, lmax + 1)
        Spherical harmonic representation of the product.
        Same dimension as sh1 and sh2
    
    Notes
    -----
    multiply the functions in the space domain, and expand the resulting
    field in spherical harmonics using SHExpandGLQ. The spherical harmonic
    bandwidth of the resulting field is lmax, which is also the bandwidths
    of the input fields.
    Conventions can be set by the optional arguments norm and csphase;
    if not set, the default is to use geodesy 4-pi normalized harmonics
    and exclude the Condon-Shortley phase of :math:`(-1)^m`. See 
    `Complex spherical harmonics <https://shtools.oca.eu/shtools/complex-spherical-harmonics.html>`_
    in SHTOOLS

    """
    SH1 = _psh.SHCoeffs.from_array(sh1, normalization=norm, csphase=csphase)
    SH2 = _psh.SHCoeffs.from_array(sh2, normalization=norm, csphase=csphase)
    grid1 = SH1.expand('GLQ')
    grid2 = SH2.expand('GLQ')
    GridProd = grid1 * grid2
    SHProd = GridProd.expand()
    return SHProd.to_array()

def DiffNormCoeffs(lmax, norm=None, csphase=1, shtype='irr'):
    """Calculate the normalization coefficients for VSH2"""
    
    if csphase == -1:
        print('csphase == -1 is not implemented!')

    # first we create a list of l, m degrees lower than lmax
    l_list, m_list = LM_list(lmax - 1)
    if shtype == 'irr':
        l_d = l_list + 1
    elif shtype == 'reg':
        l_d = l_list - 1
    else:
        print('invalid shtype (irr, reg)')
        return (0, 0, 0)

    # calculate the normalization coefficient from the Ilm definition
    C_d_p = factorial(l_list + _np.abs(m_list))
    C_d_m = factorial(l_list - _np.abs(m_list))
    C_d_norm_4pi = _np.sqrt((2*l_list + 1)/(2*l_d + 1))
    C_dz_p = factorial(l_d + _np.abs(m_list))
    C_d1_p = factorial(l_d + _np.abs(m_list-1))
    C_d2_p = factorial(l_d + _np.abs(m_list+1))
    C_dz_m = factorial(l_d - _np.abs(m_list))
    C_d1_m = factorial(l_d - _np.abs(m_list-1))
    C_d2_m = factorial(l_d - _np.abs(m_list+1))

    # remove the zeros in the denominator (for regular solid harmonics only)
    C_dz_m_i = _np.zeros(C_dz_m.shape)
    C_d1_m_i = _np.zeros(C_d1_m.shape)
    C_d2_m_i = _np.zeros(C_d2_m.shape)
    C_dz_m_i[C_dz_m != 0.0] = 1.0/C_dz_m[C_dz_m != 0.0]
    C_d1_m_i[C_d1_m != 0.0] = 1.0/C_d1_m[C_d1_m != 0.0]
    C_d2_m_i[C_d2_m != 0.0] = 1.0/C_d2_m[C_d2_m != 0.0]

    C_dz_norm = _np.sqrt(C_dz_p*C_dz_m_i*C_d_m/C_d_p)
    C_d1_norm = _np.sqrt(C_d1_p*C_d1_m_i*C_d_m/C_d_p)
    C_d2_norm = _np.sqrt(C_d2_p*C_d2_m_i*C_d_m/C_d_p)

    # calculate the normalization coefficients of the spherical harmonics
    if csphase==1:
        if norm == '4pi' or norm == 'ortho':
            C_dz_list = C_dz_m/C_d_m * C_dz_norm * C_d_norm_4pi
            C_d1_list = C_d1_m/C_d_m * C_d1_norm * C_d_norm_4pi
            C_d2_list = C_d2_m/C_d_m * C_d2_norm * C_d_norm_4pi
        elif norm == 'schmidt':
            C_dz_list = C_dz_m/C_d_m * C_dz_norm
            C_d1_list = C_d1_m/C_d_m * C_d1_norm
            C_d2_list = C_d2_m/C_d_m * C_d2_norm
        else: # unnormalized
            C_dz_list = C_dz_m/C_d_m
            C_d1_list = C_d1_m/C_d_m
            C_d2_list = C_d2_m/C_d_m
    else: # not implemented
        print('DiffNormCoeffs: csphase = -1 not implemented')
        return (0, 0, 0)

    C_dz = _psh.SHCoeffs.from_zeros(lmax=lmax,kind='complex')
    C_d1 = _psh.SHCoeffs.from_zeros(lmax=lmax,kind='complex')
    C_d2 = _psh.SHCoeffs.from_zeros(lmax=lmax,kind='complex')
    C_dz.set_coeffs(-C_dz_list, l_list + 1, m_list)
    C_d1.set_coeffs(C_d1_list, l_list + 1, m_list - 1)
    C_d2.set_coeffs(-C_d2_list, l_list + 1, m_list + 1)

    return (C_d1.to_array(), C_d2.to_array(), C_dz.to_array())

def ISHgrad(coeffs, norm='4pi', csphase=1, r=1.0):
    """Calculate the gradient of irregular solid harmonics (given unit radius)"""
    # normalize the coefficients by radius
    lmax = coeffs.shape[2] - 1
    C_d1, C_d2, C_dz = DiffNormCoeffs(lmax, norm=norm, csphase=csphase)
    lp1 = l_coeffs(lmax) + 1
    coeffs = coeffs / (r**lp1)

    # Dz(I_l^m) = C_dz * I_(l+1)^m
    clm_dz = _np.zeros_like(coeffs, dtype=complex) 
    clm_dz[:,1:,:] = coeffs[:,:-1,:]
    clm_dz = clm_dz * C_dz

    # D1(I_l^m) = C_d1 * I_(l+1)^(m-1)
    clm_d1 = _np.zeros_like(coeffs, dtype=complex) 
    clm_d1[0,1:,:-1] = coeffs[0,:-1,1:]
    clm_d1[1,1:,1:] = coeffs[1,:-1,:-1]
    clm_d1[1,1:,1] = coeffs[0,:-1,0]
    clm_d1 = clm_d1 * C_d1

    # D2(I_l^m) = C_d2 * I_(l+1)^(m+1)
    clm_d2 = _np.zeros_like(coeffs, dtype=complex) 
    clm_d2[0,1:,1:] = coeffs[0,:-1,:-1]
    clm_d2[1,1:,:-1] = coeffs[1,:-1,1:]
    clm_d2[0,1:,0] = coeffs[1,:-1,1]
    clm_d2 = clm_d2 * C_d2

    # Dx = (D1 + D2)/2; Dy = (D2 - D1)/2j;
    clm_dx = (clm_d1 + clm_d2)/2.0
    clm_dy = (clm_d2 - clm_d1)/2.0j

    return _np.stack([clm_dx, clm_dy, clm_dz], axis=-1)
    
def VSH1(coeffs, norm='4pi', csphase=1):
    """Calculate the first vector spherical harmonics (directional)

    The function will take a spherical harmonic coefficient array,
    multiply by the normal vector, and return the first vector
    spherical harmonics in Cartesian coordinates.

    Parameters
    ----------

    coeffs : complex, dimension [2, lmax + 1, lmax + 1]
        one set of SH coefficients representing a function
        following SHTOOLS conventions.
    norm : optional, ['4pi', 'ortho', 'schmidt', 'unnorm'], default='4pi'
        Normalization of the spherical harmonic functions
    csphase : optional, integer, [1, -1], default=1
        Condon-Shortley phase
    
    Returns
    -------
    complex, dimension (2, lmax + 1, lmax + 1, 3)
        Spherical harmonic representation of the first vector
        spherical harmonics.
    
    Notes
    -----
    The first vector spherical harmonics is defined as:
    
    .. math:: \mathbf{Y}_l^m(\\theta, \\varphi) = Y_l^m(\\theta, \\varphi) \hat{r}
    
    where :math:`\hat{r}=\\left[x/r, y/r, z/r\\right]^T` is the unit normal vector in radial
    direction, in Cartesian coordinates:
    
    .. math:: \mathbf{Y}_l^m = Y_l^m \\left[\\frac{x}{r}, \\frac{y}{r}, \\frac{z}{r}\\right]^T

    """
    clm = _psh.SHCoeffs.from_array(coeffs, normalization=norm, csphase=csphase)
    grid = clm.expand('GLQ')
    grid_data = grid.data
    latglq, longlq = _psh.expand.GLQGridCoord(clm.lmax, extend=True)
    LON, LAT = _np.meshgrid(longlq, latglq)
    THETA = _np.deg2rad(90-LAT)
    PHI = _np.deg2rad(LON)

    grid.data = grid_data * _np.sin(THETA) * _np.cos(PHI)
    Y_x = grid.expand()
    grid.data = grid_data * _np.sin(THETA) * _np.sin(PHI)
    Y_y = grid.expand()
    grid.data = grid_data * _np.cos(THETA)
    Y_z = grid.expand()

    Y = _np.stack((Y_x.to_array(), Y_y.to_array(), Y_z.to_array()), axis=-1)
    return Y

def VSH2(coeffs, norm='4pi', csphase=1):
    """Calculate the second vector spherical harmonics (gradient) 

    The function will take a spherical harmonic coefficient array,
    take the derivative, and return the second vector
    spherical harmonics in Cartesian coordinates.

    Parameters
    ----------

    coeffs : complex, dimension [2, lmax + 1, lmax + 1]
        one set of SH coefficients representing a function 
        following SHTOOLS conventions.
    norm : optional, ['4pi', 'ortho', 'schmidt', 'unnorm'], default='4pi'
        Normalization of the spherical harmonic functions
    csphase : optional, integer, [1, -1], default=1
        Condon-Shortley phase
    
    Returns
    -------
    complex, dimension (2, lmax + 1, lmax + 1, 3)
        Spherical harmonic representation of the second vector
        spherical harmonics.
    
    Notes
    -----
    The second vector spherical harmonics is defined as:
    
    .. math:: \mathbf{\Psi}_l^m(\\theta, \\varphi) = r\\nabla Y_l^m(\\theta, \\varphi)

    where :math:`r` is the radius. The 2nd VSH represents the gradient of a
    spherical harmonic function in Cartesian coordinates.
    
    .. math:: \mathbf{\Psi}_l^m = r\\left[\\frac{\\partial Y_l^m}{\\partial x}, \\frac{\\partial Y_l^m}{\\partial y}, \\frac{\\partial Y_l^m}{\\partial z}\\right]^T

    """
    lmax = coeffs.shape[2] - 1
    dcoeffs = ISHgrad(coeffs)

    # the first spherical harmonic part
    l_list, m_list = LM_list(lmax+1)
    lp1 = _np.swapaxes(_np.broadcast_to(_np.arange(lmax+1)+1, (2, lmax+1, lmax+1)), 1,2)
    lp1_Y = VSH1(coeffs*lp1, norm=norm, csphase=csphase)

    return dcoeffs + lp1_Y
