import numpy as np
from solitons1d.soliton import get_first_derivative
from solitons1d.grid import Grid

def test_first_derivative():
    """
    Computes the first derivative of the function f(x) = 2x and checks
    that it is 2.
    """

    lattice_spacing = 0.1
    linear_function = 2*np.arange(-1,1,lattice_spacing)
    first_derivative = get_first_derivative(linear_function, len(linear_function), lattice_spacing)

    assert np.allclose(first_derivative[2:-2], 2.0)


def test_grid_construction():
    """
    Makes a grid object and checks that data is propagated to the object.
    Then checks that an error is raised the number of lattice points is given as
    a float.
    """

    num_grid_points = 100
    grid_spacing = 0.2

    grid = Grid(num_grid_points=num_grid_points, grid_spacing=grid_spacing)

    # Check that arguments become attributes of the object
    assert grid.num_grid_points == num_grid_points
    assert grid.grid_spacing == grid_spacing
