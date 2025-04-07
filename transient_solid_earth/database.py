"""
RAM to file conversions.
"""

from json import JSONEncoder, dump, load
from pathlib import Path
from typing import Any, Optional, TypedDict

import numpy
from pydantic import BaseModel


class JSONSerialize(JSONEncoder):
    """
    Handmade JSON encoder that correctly encodes special structures.
    """

    def default(self, o):
        if isinstance(o, numpy.ndarray):
            return o.tolist()
        if isinstance(o, BaseModel):
            return o.__dict__
        return JSONEncoder().default(o)


def save_base_model(obj: Any, name: str, path: Path):
    """
    Saves a JSON serializable type.
    """

    # Eventually considers subpath.
    while len(name.split("/")) > 1:
        path = path.joinpath(name.split("/")[0])
        name = "".join(name.split("/")[1:])

    # May create the directory.
    path.mkdir(exist_ok=True, parents=True)

    # Saves the object.
    with open(path.joinpath(name + ".json"), "w", encoding="utf-8") as file:
        dump(obj, fp=file, cls=JSONSerialize, indent=4)


def load_base_model(
    name: str,
    path: Path,
    base_model_type: Optional[Any] = None,
) -> Any:
    """
    Loads a JSON serializable type.
    """

    filepath = path.joinpath(name + ("" if ".json" in name else ".json"))
    with open(filepath, "r", encoding="utf-8") as file:
        loaded_content = load(fp=file)
    return loaded_content if not base_model_type else base_model_type(**loaded_content)


class ComplexDict(TypedDict):
    """
    Ensures that a dictionary contains a real part and an imaginary part.
    """

    real: numpy.ndarray[float]
    imag: numpy.ndarray[float]


def complex_array_to_dict(array: numpy.ndarray[complex]) -> ComplexDict:
    """
    Transforms a complex numpy array into a dictionary compatible wih (.JSON).
    """
    return {"real": array.real, "imag": array.imag}


def complex_dict_to_array(dictionary: ComplexDict) -> numpy.ndarray[complex]:
    """
    Transforms a complex dictionary into a numpy array.
    """
    return dictionary["real"] + dictionary["imag"] * 1.0j


def generate_degrees_list(
    degree_thresholds: list[int],
    degree_steps: list[int],
) -> list[int]:
    """
    Generates the list of degrees for which to compute Love numbers, given a list of thresholds and
    a list of steps.
    """

    return numpy.concatenate(
        [
            numpy.arange(degree_thresholds[i], degree_thresholds[i + 1], degree_step, dtype=int)
            for i, degree_step in enumerate(degree_steps)
        ],
    ).tolist()
