"""
ShElastic
=========

Provides
  1. Linear elasticity solver for boundary conditions on spherical interfaces
  2. Spherical harmonics representation of tensor based on SHTOOLS [1]
  3. Represent tensor operation (multiplication and gradient) in spherical harmonic space

Available modules
-----------------
shgrad
    Class for calculating first and second vector spherical harmonics
shelastic
    Class for generating elastic solutions based on Papkovich-Neuber solutions
shutil
    Routines for operating spherical harmonic coefficients
shbv
    Routines for solving spherical boundary-value problems
shtest
    Routines for importing solutions to classical elasticity problems
"""

__version__ = '1.0'
__author__ = 'Yifan Wang'

import os
import numpy as _np
import pyshtools as _psh

# ---- Define __all__ for use with: from shelastic import * ----
__all__ = ['shgrad', 'shelastic', 'shutil', 'shbv', 'shtest']

def _sanity_check():
    """
    Quick sanity checks for common bugs caused by environment.
    """
    pass
    # try:
        # x = ones(2, dtype=float32)
        # if not abs(x.dot(x) - 2.0) < 1e-5:
            # raise AssertionError()
    # except AssertionError:
        # msg = ("The current Numpy installation ({!r}) fails to "
               # "pass simple sanity checks. This can be caused for example "
               # "by incorrect BLAS library being linked in.")
        # raise RuntimeError(msg.format(__file__))

_sanity_check()
del _sanity_check
