"""
shelastic
=========
Generating the spherical harmonic representation of displacement basis :math:`u^{(K)}`,
stress basis :math:`\mathbf{\sigma}^{(K)}`, and traction basis :math:`\mathbf{T}^{(K)}` 
with index K (k,l,m).
"""

import numpy as _np
import scipy.sparse as _spm
from scipy.io import savemat
import pyshtools as _psh
import sys, os
from shelastic.shutil import SHCilmToVector, sparse_mode, lmk2K
from shelastic.shgrad import VSH1, VSH2

def genUmode(l, m, k, shtype='irr', lmax=None):
    '''generate displacement mode (K) = (l,m,k)
    
    Calculate the displacement vector basis of indices (l,m,k) at unit
    sphere (r = 1), and unit shear modulus (mu = 1). This function 
    returns U_nu and U_0 which do not depend on Poisson's ratio nu.
    use calUmode((U_nu, U_0), mu, nu) to obtain the displacement
    vector given shear modulus mu and Poisson's ratio nu.

    Parameters
    ----------
    l,m,k : int
        indices of the basis, l=0,1,2,...; m=-l,-l+1,...,l-1,l; k=0,1,2.
    shtype : str, ['irr' or 'reg']
        'irr' represents irregular spherical harmonics for spherical void
        'reg' represents regular spherical harmonics for solid sphere
    lmax : int, l+3 by default
        maximum l order to save the spherical harmonic coefficients.
    
    Returns
    -------
    U_nu,U_0 : complex, dimension (2, lmax + 1, lmax + 1, 3)
        Spherical harmonic representation of the two parts of the 
        displacement basis.
    
    Notes
    -----
    See (A.10) in the paper for the equation and derivation of the displacement
    solution

    '''
    if shtype == 'irr':
        C_vsh1 = -l
        if lmax is None:
            lmax = l + 3
    elif shtype == 'reg':
        C_vsh1 = l+1
        if lmax is None:
            lmax = l + 3
    else:
        print('U_mode: invalid shtype (irr, reg)')
        return 0
    Ylm = _psh.SHCoeffs.from_zeros(lmax=lmax, kind='complex')
    Ylm.set_coeffs(1.0, l, m)
    U_nu = _np.zeros(Ylm.to_array().shape+(3,), dtype=complex)
    U_nu[..., k] = Ylm.to_array()
    rdotPsi = VSH1(Ylm.to_array())[...,k]
    U_0 = C_vsh1*VSH1(rdotPsi) + VSH2(rdotPsi)
    return (U_nu, U_0)

def calUmode(Umode, mu, nu, c_omega=1.0):
    '''Calculate spherical harmonic vector representation of displacement given shear modulus mu and Poisson's ratio nu
    
    Parameters
    ----------
    Umode : tuple, (U_nu, U_0)
        displacement mode calculated by genUmode
    mu : float
        shear modulus
    nu : float
        Poisson's ratio
    c_omega : float, optional
        pre-factor, not used
    
    Returns
    -------
    complex, dimension (2, lmax + 1, lmax + 1, 3)
        Spherical harmonic vector representation ofthe displacement basis given mu and nu.

    '''
    U_nu, U_0 = Umode
    C_nu = -4*(1-nu)
    return c_omega*(U_nu * C_nu + U_0)/2.0/mu

'''def genGradU(l, m, k, shtype='irr', lmax=None, returnU=False):
    generate displacement gradient tensor of a spherical-harmonic displacement vector
    
    Calculate the displacement gradient tensor grad(U) given a spherical 
    harmonic vector representation of displacement U. This function 
    returns a spherical harmonic tensor gradU given the type of vector spherical harmonics
    (regular or irregular)
    
    Parameters
    ----------
    l,m,k : int
        indices of the basis, l=0,1,2,...; m=-l,-l+1,...,l-1,l; k=0,1,2.
    shtype : str, ['irr' or 'reg']
        'irr' represents irregular spherical harmonics for spherical void
        'reg' represents regular spherical harmonics for solid sphere
    lmax : int, l+3 by default
        maximum l order to save the spherical harmonic coefficients.
    returnU : bool
        If True, the displacement basis will also be returned.

    Returns
    -------
    gradu : complex, dimension (2, lmax + 1, lmax + 1, 3, 3)
        Spherical harmonic representation of the displacement gradient tensor.

    Notes
    -----
    See (A.16) in the paper for the equation and derivation of the stress
    solution


    if lmax is None:
        lmax = l + 3
    Umodes = genUmode(l, m, k, shtype=shtype, lmax=lmax)
    gradUmodes = []
    for U in Umodes:
        gradu = [None for _ in range(3)]
        for i in range(3):
            gradu[i] = c1 * VSH1(U[...,i]) + c2 * VSH2(U[...,i])
        gradUmodes.append(_np.stack(gradu, axis=-1))
'''
    
def genSmode(l, m, k, shtype='irr', lmax=None, returnU=False):
    '''generate stress mode (K) = (l,m,k)
    
    Calculate the stress tensor basis of indices (l,m,k) at unit
    sphere (r = 1), and unit shear modulus (mu = 1). This function 
    returns S_nu1, S_nu2, S_nu3, and S_0 which do not depend on
    Poisson's ratio nu.
    
    Use calSmode((S_nu1, S_nu2, S_nu3, S_0), mu, nu) to obtain the 
    stress tensor given shear modulus mu and Poisson's ratio nu.

    Parameters
    ----------
    l,m,k : int
        indices of the basis, l=0,1,2,...; m=-l,-l+1,...,l-1,l; k=0,1,2.
    shtype : str, ['irr' or 'reg']
        'irr' represents irregular spherical harmonics for spherical void
        'reg' represents regular spherical harmonics for solid sphere
    lmax : int, l+3 by default
        maximum l order to save the spherical harmonic coefficients.
    returnU : bool
        If True, the displacement basis will also be returned.
    
    Returns
    -------
    U_nu,U_0 : complex, dimension (2, lmax + 1, lmax + 1, 3)
        Spherical harmonic representation of the two parts of the 
        displacement basis. only returned when returnU is True.
    S_nu1,S_nu2,S_nu3,S_0 : complex, dimension (2, lmax + 1, lmax + 1, 3, 3)
        Spherical harmonic representation of the four parts of the 
        stress tensor basis.

    Notes
    -----
    See (A.17) in the paper for the equation and derivation of the stress
    solution

    '''
    if shtype == 'irr':
        c1 = -(l+1)
    elif shtype == 'reg':
        c1 = l
    else: # if (shtype != 'irr') and (shtype != 'reg'):
        print('genGradU: invalid shtype (irr, reg)')
        return 0
    c2 = 1.0
    if lmax is None:
        lmax = l + 3

    Umodes = genUmode(l, m, k, shtype=shtype, lmax=lmax)
    gradUmodes = []
    for U in Umodes:
        gradu = [None for _ in range(3)]
        for i in range(3):
            gradu[i] = c1 * VSH1(U[...,i]) + c2 * VSH2(U[...,i])
        gradUmodes.append(_np.stack(gradu, axis=-1))
    gradU_nu, gradU_0 = tuple(gradUmodes)

    ukk_nu = _np.trace(gradU_nu, axis1=-1, axis2=-2)
    ukk_0 = _np.trace(gradU_0, axis1=-1, axis2=-2)
    S_nu1 = ukk_nu[...,_np.newaxis,_np.newaxis]*_np.eye(3)
    S_nu2 = 0.5*(gradU_nu+_np.swapaxes(gradU_nu, -1, -2))
    S_nu3 = ukk_0[...,_np.newaxis,_np.newaxis] *_np.eye(3)
    S_0 = 0.5*(gradU_0+_np.swapaxes(gradU_0, -1, -2))
    if returnU:
        U_nu, U_0 = Umodes
        return (U_nu, U_0, S_nu1, S_nu2, S_nu3, S_0)
    else:
        return (S_nu1, S_nu2, S_nu3, S_0)

def calSmode(Smode, mu, nu, c_omega=1.0):
    '''Calculate spherical harmonic tensor representation of stress given shear modulus mu and Poisson's ratio nu
    
    Parameters
    ----------
    Smode : tuple, (S_nu1, S_nu2, S_nu3, S_0)
        stress mode calculated by genSmode
    mu : float
        shear modulus
    nu : float
        Poisson's ratio
    c_omega : float, optional
        pre-factor, not used
    
    Returns
    -------
    complex, dimension (2, lmax + 1, lmax + 1, 3, 3)
        Spherical harmonic vector representation of the displacement basis given mu and nu.

    '''
    S_nu1, S_nu2, S_nu3, S_0 = Smode
    c_nu1 = -4*(1-nu)
    c_nu2 = nu/(1-2*nu)
    return c_omega*(c_nu1*c_nu2*S_nu1 + c_nu1*S_nu2 + c_nu2*S_nu3 + S_0)

def calTmode(S_K):
    '''Calculate spherical harmonic vector representation of traction given the stress tensor
    
    Parameters
    ----------
    S_K : complex, dimension (2, lmax + 1, lmax + 1, 3, 3)
        stress spherical-harmonic tensor
    
    Returns
    -------
    T_K : complex, dimension (2, lmax + 1, lmax + 1, 3)
        Spherical harmonic vector representation of the traction.

    '''
    T_K = _np.zeros(S_K.shape[:-1], dtype=complex)
    for i in range(3):
        T_K[...,i] = VSH1(S_K[...,i,0])[...,0]+VSH1(S_K[...,i,1])[...,1]+VSH1(S_K[...,i,2])[...,2]
    return T_K

def generate_modes(lmax, etol=1e-8, save_lmax=None, path=None):
    '''procedure for generating spherical harmonic modes in sparse matrix format

    The procedure saves .mat file that includes all the elasticity bases with 
    l <= lmax.
    
    Umodes.mat: 
    
    'U0irr', 'U1irr': U_0 and U_nu for irregular spherical harmonics;
    'U0reg', 'U1reg': U_0 and U_nu for regular spherical harmonics;
    Every element is a sparse matrix of size (3x(save_lmax+1)^2, 3x(lmax+1)^2)
    
    Smodes.mat: 
    
    'S0irr', 'S1irr', 'S2irr', 'S3irr': S_0, S_nu1, S_nu2, S_nu3 for irregular spherical harmonics;
    'S0reg', 'S1reg', 'S2reg', 'S3reg': S_0, S_nu1, S_nu2, S_nu3 for regular spherical harmonics;
    Every element is a sparse matrix of size (9x(save_lmax+1)^2, 3x(lmax+1)^2)
    
    Tmodes.mat: 
    
    'T0irr', 'T1irr', 'T2irr', 'T3irr': T_0, T_nu1, T_nu2, T_nu3 for irregular spherical harmonics;
    'T0reg', 'T1reg', 'T2reg', 'T3reg': T_0, T_nu1, T_nu2, T_nu3 for regular spherical harmonics;
    Every element is a sparse matrix of size (3x(save_lmax+1)^2, 3x(lmax+1)^2)
    
    To calculate traction given mu and nu, use calSmode((T_0, T_nu1, T_nu2, T_nu3), mu, nu)

    Parameters
    ----------
    lmax : int
        generating all the spherical harmonic modes of l <= lmax
    etol : float
        Threshold below which is considered as noise (save space for sparse matrix)
    save_lmax : int, by default lmax+3
        maximum lmax needed for saving all the elasticity solutions
    path : string
        path for saving the modes

    '''
    # Todo: append modes given lmax and existing mode
    if save_lmax is None:
        save_lmax = lmax + 3
    M = 3*(save_lmax+1)**2
    N = 3*(lmax+1)**2
    U1irr = _spm.lil_matrix((M,N), dtype=complex); U1reg = _spm.lil_matrix((M,N), dtype=complex);
    U0irr = _spm.lil_matrix((M,N), dtype=complex); U0reg = _spm.lil_matrix((M,N), dtype=complex);
    S1irr = _spm.lil_matrix((3*M,N), dtype=complex); S1reg = _spm.lil_matrix((3*M,N), dtype=complex);
    S2irr = _spm.lil_matrix((3*M,N), dtype=complex); S2reg = _spm.lil_matrix((3*M,N), dtype=complex);
    S3irr = _spm.lil_matrix((3*M,N), dtype=complex); S3reg = _spm.lil_matrix((3*M,N), dtype=complex);
    S0irr = _spm.lil_matrix((3*M,N), dtype=complex); S0reg = _spm.lil_matrix((3*M,N), dtype=complex);
    T1irr = _spm.lil_matrix((M,N), dtype=complex); T1reg = _spm.lil_matrix((M,N), dtype=complex);
    T2irr = _spm.lil_matrix((M,N), dtype=complex); T2reg = _spm.lil_matrix((M,N), dtype=complex);
    T3irr = _spm.lil_matrix((M,N), dtype=complex); T3reg = _spm.lil_matrix((M,N), dtype=complex);
    T0irr = _spm.lil_matrix((M,N), dtype=complex); T0reg = _spm.lil_matrix((M,N), dtype=complex);
    for l in range(lmax+1):
        for m in range(-l, l+1):
            for k in range(3):
                K = lmk2K(l,m,k,lmax=lmax)
                print(l, m, k)
                Unu, U0, Snu1, Snu2, Snu3, S0 = genSmode(l, m, k, shtype='irr',returnU=True)
                Tnu1 = calTmode(Snu1); Tnu2 = calTmode(Snu2); Tnu3 = calTmode(Snu3); T0 = calTmode(S0)
                print('irregular solid harmonic modes...')
                U1irr[:, K] = sparse_mode(Unu, lmax=save_lmax, etol=etol); 
                U0irr[:, K] = sparse_mode(U0, lmax=save_lmax, etol=etol);
                S1irr[:, K] = sparse_mode(Snu1, lmax=save_lmax, etol=etol); 
                S2irr[:, K] = sparse_mode(Snu2, lmax=save_lmax, etol=etol);
                S3irr[:, K] = sparse_mode(Snu3, lmax=save_lmax, etol=etol); 
                S0irr[:, K] = sparse_mode(S0, lmax=save_lmax, etol=etol);
                T1irr[:, K] = sparse_mode(Tnu1, lmax=save_lmax, etol=etol); 
                T2irr[:, K] = sparse_mode(Tnu2, lmax=save_lmax, etol=etol);
                T3irr[:, K] = sparse_mode(Tnu3, lmax=save_lmax, etol=etol); 
                T0irr[:, K] = sparse_mode(T0, lmax=save_lmax, etol=etol);
                Unu, U0, Snu1, Snu2, Snu3, S0 = genSmode(l, m, k, shtype='reg',returnU=True)
                Tnu1 = -calTmode(Snu1); Tnu2 = -calTmode(Snu2); Tnu3 = -calTmode(Snu3); T0 = -calTmode(S0)
                print('regular solid harmonic modes...')
                U1reg[:, K] = sparse_mode(Unu, lmax=save_lmax, etol=etol); 
                U0reg[:, K] = sparse_mode(U0, lmax=save_lmax, etol=etol);
                S1reg[:, K] = sparse_mode(Snu1, lmax=save_lmax, etol=etol); 
                S2reg[:, K] = sparse_mode(Snu2, lmax=save_lmax, etol=etol);
                S3reg[:, K] = sparse_mode(Snu3, lmax=save_lmax, etol=etol); 
                S0reg[:, K] = sparse_mode(S0, lmax=save_lmax, etol=etol);
                T1reg[:, K] = sparse_mode(Tnu1, lmax=save_lmax, etol=etol); 
                T2reg[:, K] = sparse_mode(Tnu2, lmax=save_lmax, etol=etol);
                T3reg[:, K] = sparse_mode(Tnu3, lmax=save_lmax, etol=etol); 
                T0reg[:, K] = sparse_mode(T0, lmax=save_lmax, etol=etol);
    Umodes = {'U1irr': U1irr, 'U1reg': U1reg, 'U0irr': U0irr, 'U0reg': U0reg}
    Smodes = {'S1irr': S1irr, 'S1reg': S1reg, 'S2irr': S2irr, 'S2reg': S2reg,
              'S3irr': S3irr, 'S3reg': S3reg, 'S0irr': S0irr, 'S0reg': S0reg}
    Tmodes = {'T1irr': T1irr, 'T1reg': T1reg, 'T2irr': T2irr, 'T2reg': T2reg,
              'T3irr': T3irr, 'T3reg': T3reg, 'T0irr': T0irr, 'T0reg': T0reg}
    if path is None:
        return Umodes, Smodes, Tmodes
    else:
        savemat(os.path.join(path, 'Umodes.mat'), Umodes)
        savemat(os.path.join(path, 'Smodes.mat'), Smodes)
        savemat(os.path.join(path, 'Tmodes.mat'), Tmodes)
        print('save success')
