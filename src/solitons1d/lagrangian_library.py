from numba import njit
import numpy as np

from .lagrangian import Lagrangian

@njit
def phi4_V(x):
    return 0.5*np.pow(1-np.pow(x,2),2)

@njit
def phi4_dV(x):
    return 2*np.pow(x,3) - 2*x

lagrangians = {
    "phi_4": {"V": phi4_V, "dV": phi4_dV, "vacua": [-1,1]},
}

list_of_lagrangians = list(lagrangians.keys())

def make_lagrangian_from_library(lagrangian_string):
    
    lagrangian_dict = lagrangians.get(lagrangian_string)
    assert lagrangian_dict is not None, f"{lagrangian_string} is not in the Lagrangian"\
    "Library. You can view the supported Lagrangians in `solitons1d.list_of_lagrangians`"

    lagrangian = Lagrangian(**lagrangian_dict)
    return lagrangian

