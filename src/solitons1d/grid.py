import json
from pathlib import Path
import numpy as np


class Grid:
    """
    A 1D grid.

    Parameters
    ----------
    num_grid_points : int
        Number of grid points used in grid.
    grid_spacing : float
        Spacing between lattice points.

    Attributes
    ----------
    grid_length : float
        Total length of grid.
    grid_points : np.array[float]
        An array of grid points, of the grid.

    """

    def __init__(
        self,
        num_grid_points: int,
        grid_spacing: float,
    ):
        self.num_grid_points = num_grid_points
        self.grid_spacing = grid_spacing
        self.grid_length = (num_grid_points) * grid_spacing

        self.grid_points = np.arange(
            -self.grid_length / 2, self.grid_length / 2, grid_spacing
        )

    def save(
        self,
        folder_name: str | Path,
    ):
        """
        Saves a `Grid` object at `folder_name`.
        """
        metadata = {
            "num_grid_points": self.num_grid_points,
            "grid_spacing": self.grid_spacing,
        }

        # make the folder a Path if it is a string
        folder = Path(folder_name)
        folder.mkdir(exist_ok=True)

        # this overwrites any existing metadata.json file
        with open(folder / "metadata.json", "w") as f:
            json.dump(metadata, f)


def load_grid(folder_name: str | Path):
    """
    Loads the `Grid` object at `folder_name`.
    """
    folder = Path(folder_name)
    metadata_path = folder / "metadata.json"

    assert metadata_path.is_file(), (
        f"Could not find Grid `metadata.json` file in {folder}."
    )

    with open(metadata_path, "r") as f:
        grid_metadata = json.load(f)

    # the ** "unpacks" the dictionary into a series of arguments
    grid = Grid(**grid_metadata)
    return grid
