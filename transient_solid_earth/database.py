"""
RAM to file conversions.
"""

import json
from pathlib import Path
from time import sleep
from typing import Any, Optional

import numpy
from pydantic import BaseModel


class JSONSerialize(json.JSONEncoder):
    """
    Handmade JSON encoder that correctly encodes special structures.
    """

    def default(self, o):
        if isinstance(o, numpy.ndarray):
            return o.tolist()
        if isinstance(o, BaseModel):
            return o.__dict__
        return json.JSONEncoder().default(o)


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
        json.dump(obj, fp=file, cls=JSONSerialize, indent=4)


def load_base_model(
    name: str,
    path: Path,
    base_model_type: Optional[Any] = None,
) -> Any:
    """
    Loads a JSON serializable type.
    """

    filepath = path.joinpath(name + ("" if ".json" in name else ".json"))
    try:
        with open(filepath, "r", encoding="utf-8") as file:
            loaded_content = json.load(fp=file)
    except json.decoder.JSONDecodeError:
        # Waits to avoid concurrent reading/writing.
        sleep(1e-3)
        # Then retries.
        return load_base_model(name=name, path=path, base_model_type=base_model_type)
    return loaded_content if not base_model_type else base_model_type(**loaded_content)


def load_complex_array(path: Path, name: Optional[str] = None) -> numpy.ndarray:
    """
    Loads a complex array.
    """

    path = path if not name else path.joinpath(name)
    return numpy.array(object=load_base_model(name="real", path=path)) + 1.0j * numpy.array(
        object=load_base_model(name="imag", path=path)
    )


def save_complex_array(
    obj: dict[str, numpy.ndarray] | numpy.ndarray, path: Path, name: Optional[str] = None
) -> None:
    """
    Saves a complex array.
    """

    if isinstance(obj, numpy.ndarray):
        obj = {
            "real": numpy.real(obj),
            "imag": numpy.imag(obj),
        }
    path = path if not name else path.joinpath(name)
    save_base_model(obj=obj["real"], name="real", path=path)
    save_base_model(obj=obj["imag"], name="imag", path=path)


def generate_degrees_list(
    degree_thresholds: list[int],
    degree_steps: list[int],
    n_max: Optional[int] = None,
) -> list[int]:
    """
    Generates the list of degrees for which to compute Love numbers, given a list of thresholds and
    a list of steps.
    """

    if n_max:
        degree_thresholds = [threshold for threshold in degree_thresholds if threshold <= n_max]
        degree_steps = degree_steps[: len(degree_thresholds) - 1]
        degree_thresholds += [n_max + degree_steps[-1]]

    return numpy.concatenate(
        [
            numpy.arange(degree_thresholds[i], degree_thresholds[i + 1], degree_step, dtype=int)
            for i, degree_step in enumerate(degree_steps)
        ],
    ).tolist()
