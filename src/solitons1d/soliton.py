import json
from typing import Callable, Literal
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from .grid import Grid
from .lagrangian import Lagrangian
from .lagrangian_library import make_lagrangian_from_library

from numba import njit, prange

def print_hello():
    print("hello")

class Soliton:
    """
    A class describing a Soliton.

    Parameters
    ----------
    grid : Grid
        The grid underpinning the soliton.
    lagrangian : Lagrangian | str
        The Lagrangian of the theory supporting the soliton.
    initial_profile_function : None | function
        The initial profile function, must be from R -> R. Optional.
    initial_profile : None | array-like
        The initial profile function as an array. Optional.
    """

    def __init__(
        self,
        grid: Grid,
        lagrangian: Lagrangian | Literal["phi_4", "sine_Gordon"],
        initial_profile_function: Callable[[float], float] | None = None,
        initial_profile: np.ndarray | None = None,
    ):
        self.grid = grid

        if isinstance(lagrangian, str):
            self.lagrangian = make_lagrangian_from_library(lagrangian)
        elif isinstance(lagrangian, Lagrangian):
            self.lagrangian = lagrangian
        else:
            raise TypeError("`lagrangian` must be a string or a Lagrangian object")

        self.profile = np.zeros(grid.num_grid_points)

        assert (initial_profile_function is None) or (initial_profile is None), (
            "Please only specify `initial_profile_function` or `profile_function`"
        )

        if initial_profile_function is not None:
            self.profile = create_profile(self.grid.grid_points, initial_profile_function)
        else:
            self.profile = initial_profile

        self.compute_energy()

    def compute_energy(self):
        """Computes the energy of a soliton, and stores this in the `energy` attribute."""

        energy = compute_energy_fast(
            self.lagrangian.V,
            self.profile,
            self.grid.num_grid_points,
            self.grid.grid_spacing,
        )
        self.energy = energy

    def plot_soliton(self):
        """Makes a plot of the profile function of your soliton"""

        fig, ax = plt.subplots()
        ax.plot(self.grid.grid_points, self.profile)
        ax.set_title(f"Profile function. Energy = {self.energy:.4f}")

        return fig

    def save(
        self,
        folder_name: str | Path,
    ):
        """
        Saves a `Soliton` object at `folder_name`.
        """

        folder = Path(folder_name)
        folder.mkdir(exist_ok=True)

        grid_folder = "./grid"
        lagrangian_folder = "./lagrangian"

        metadata = {
            "grid_folder": str(grid_folder),
            "lagrangian_folder": str(lagrangian_folder),
        }

        properties = {"energy": self.energy}

        self.grid.save(grid_folder)
        self.lagrangian.save(lagrangian_folder)

        with open(folder / "metadata.json", "w") as f:
            json.dump(metadata, f)

        with open(folder / "properties.json", "w") as f:
            json.dump(properties, f)

        # use `numpy`s save function to save the profile array
        np.save("profile", self.profile)

    def grad_flow(self, dt, num_steps):
        self.profile = grad_flow_fast(
            self.lagrangian.dV,
            self.profile,
            self.grid.num_grid_points,
            self.grid.grid_spacing,
            dt,
            num_steps,
        )
        self.compute_energy()


def load_soliton(folder_name):
    """
    Loads the `Lagrangian` object at `folder_name`.
    """
    folder = Path(folder_name)
    metadata_path = folder / "metadata.json"

    assert metadata_path.is_file(), (
        f"Could not find Grid `metadata.json` file in {folder}."
    )

    with open(metadata_path, "r") as f:
        soliton_metadata = json.load(f)

    grid_folder = soliton_metadata.get("grid_folder")
    grid = load_grid(grid_folder)

    lagrangian_folder = soliton_metadata.get("lagrangian_folder")
    lagrangian = load_lagrangian(lagrangian_folder)

    profile = np.load("profile.npy")

    soliton = Soliton(grid=grid, lagrangian=lagrangian, initial_profile=profile)

    return soliton


@njit
def compute_energy_fast(
    V: Callable[[float], float],
    profile: np.array,
    num_grid_points: int,
    grid_spacing: float,
) -> float:
    """
    Computes the energy of a Lagrangian of the form
        E = 1/2 (d_phi)^2 + V(phi)

    Parameters
    ----------
    V: function
        The potential energy function
    profile: np.ndarray
        The profile function of the soliton
    num_grid_points: int
        Length of `profile`
    grid_spacing: float
        Grid spacing of underlying grid
    """
    dx_profile = get_first_derivative(profile, num_grid_points, grid_spacing)

    kin_eng = 0.5 * np.pow(dx_profile, 2)
    pot_eng = V(profile)

    tot_eng = np.sum(kin_eng + pot_eng) * grid_spacing

    return tot_eng


@njit
def grad_flow_fast(dV, profile, num_grid_points, grid_spacing, dt, num_steps):
    for _ in range(num_steps):
        dE = compute_dE(dV, profile, num_grid_points, grid_spacing)
        profile -= dt * dE

    return profile


@njit(parallel=True)
def get_first_derivative(
    phi: np.ndarray,
    num_grid_points: int,
    grid_spacing: float,
) -> np.ndarray:
    """
    For a given array, computes the first derivative of that array.

    Parameters
    ----------
    phi: np.ndarray
        Array to get the first derivative of
    num_grid_points: int
        Length of the array
    grid_spacing: float
        Grid spacing of underlying grid

    Returns
    -------
    d_phi: np.ndarray
        The first derivative of `phi`.

    """
    d_phi = np.zeros(num_grid_points)
    for i in prange(num_grid_points - 4):
        d_phi[i + 2] = (phi[i] - 8 * phi[i + 1] + 8 * phi[i + 3] - phi[i + 4]) / (
            12.0 * grid_spacing
        )

    return d_phi


@njit
def get_second_derivative(
    phi: np.ndarray,
    num_grid_points: int,
    grid_spacing: float,
) -> np.ndarray:
    """
    For a given array, computes the first derivative of that array.

    Parameters
    ----------
    phi: np.ndarray
        Array to get the first derivative of
    num_grid_points: int
        Length of the array
    grid_spacing: float
        Grid spacing of underlying grid

    Returns
    -------
    d_phi: np.ndarray
        The first derivative of `phi`.

    """
    ddV = np.zeros(num_grid_points)
    for i in prange(num_grid_points - 4):
        ddV[i + 2] = (
            -phi[i] + 16 * phi[i + 1] - 30 * phi[i + 2] + 16 * phi[i + 3] - phi[i + 4]
        ) / (12.0 * np.pow(grid_spacing, 2))

    return ddV


@njit
def compute_dE(dV, profile, num_grid_points, grid_spacing):
    dd_phi = get_second_derivative(profile, num_grid_points, grid_spacing)
    dV_array = dV(profile)

    return dV_array - dd_phi


def create_profile(
    grid_points: np.array,
    initial_profile_function: Callable[[np.array], np.array] | None = None,
) -> np.array:
    """
    Creates a profile function on a grid, from profile function `initial_profile_function`.

    Parameters
    ----------
    grid_points: Grid
        The x-values of a grid.
    initial_profile_function: function
        A function which accepts and returns a 1D numpy array

    Returns
    -------
    profile: np.array
        Generated profile function
    """

    profile = initial_profile_function(grid_points)
    return profile
