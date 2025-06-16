import pickle as pkl
from pathlib import Path
from typing import Callable
import numpy as np


class Lagrangian:
    """
    Used to represent Lagrangians of the form:
        L = - 1/2(dx_phi)^2 - V(phi)

    Parameters
    ----------
    V : function
        The potential energy function, must be a map from R -> R
    dV : function
        The derivative of the potential energy function, must be a map from R -> R
    vacua : list-like or None
        List of vacua of the potential energy.
    """

    def __init__(
        self,
        V: Callable[[float], float],
        dV: Callable[[float], float],
        vacua: list | np.ndarray | None = None,  # np.ndarray is the type of a numpy array
    ):
        self.V = V
        self.dV = dV
        self.vacua = vacua

        if vacua is not None:
            for vacuum in vacua:
                # np.isclose does what it sounds like: are the values close?
                # That f"" is called an f-string, allowing you to add parameters to strings
                assert np.isclose(dV(vacuum), 0), (
                    f"The given vacua do not satisfy dV({vacuum}) = 0"
                )

    def save(
        self,
        folder_name: str | Path,
    ):
        """
        Saves a `Lagrangian` object at `folder_name`.
        """
        metadata = {
            "V": self.V,
            "dV": self.dV,
            "vacua": self.vacua,
        }

        # make the folder a Path if it is a string
        folder = Path(folder_name)
        folder.mkdir(exist_ok=True)

        with open(folder / "metadata.pkl", "wb") as f:
            pkl.dump(metadata, f)


def load_lagrangian(folder_name: str | Path):
    """
    Loads the `Lagrangian` object at `folder_name`.
    """
    folder = Path(folder_name)
    metadata_path = folder / "metadata.pkl"

    assert metadata_path.is_file(), (
        f"Could not find Lagrangian `metadata.json` file in {folder}."
    )

    b = a + 1
    



    with open(metadata_path, "rb") as f:
        lagrangian_metadata = pkl.load(f)

    # the ** "unpacks" the dictionary into a series of arguments
    lagrangian = Lagrangian(**lagrangian_metadata)
    return lagrangian
