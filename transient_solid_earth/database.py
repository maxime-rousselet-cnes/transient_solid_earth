"""
RAM to file conversions.
"""

import json
from csv import DictWriter
from pathlib import Path
from time import sleep
from typing import Any, Optional, Union

import numpy
from pandas import read_csv
from pydantic import BaseModel

from .paths import tables_path


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
    return numpy.array(object=load_base_model(name="real", path=path)).astype(
        numpy.complex64
    ) + 1.0j * numpy.array(object=load_base_model(name="imag", path=path)).astype(numpy.complex64)


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
        degree_thresholds += [n_max + degree_steps[-1]]
        degree_steps = degree_steps[: len(degree_thresholds) - 1]

    return numpy.concatenate(
        [
            numpy.arange(degree_thresholds[i], degree_thresholds[i + 1], degree_step, dtype=int)
            for i, degree_step in enumerate(degree_steps)
        ],
    ).tolist()


def extract_terminal_attributes(obj: Any, prefix: str = "") -> Union[dict, Any]:
    """
    Recursively extracts terminal attributes from a nested object structure,
    preserving the attribute path as keys in the result dictionary.
    Terminal attributes are those that do not have a __dict__ or are not collections of objects.
    """

    if isinstance(obj, (list, tuple)):
        # Handle sequences by extracting each element with the same prefix
        result = {}
        for i, item in enumerate(obj):
            nested = extract_terminal_attributes(item, f"{prefix}{i}:" if prefix else f"{i}:")
            if isinstance(nested, dict):
                result.update(nested)
        return result

    if hasattr(obj, "__dict__"):
        result = {}
        attributes: dict = vars(obj)

        for attr, value in attributes.items():
            full_key = f"{prefix}{attr}" if not prefix else f"{prefix}:{attr}"
            if hasattr(value, "__dict__") or isinstance(value, (list, tuple)):
                nested = extract_terminal_attributes(value, full_key)
                if isinstance(nested, dict):
                    result.update(nested)
            else:
                result[full_key] = value
        return result

    return obj


def is_in_table(table_name: str, id_to_check: str) -> bool:
    """
    Verify if a given ID is in a (.CSV) table.
    """

    file = tables_path.joinpath(table_name + ".csv")

    if file.exists():

        df = read_csv(file)

        return id_to_check in df["ID"].values

    return False


def add_result_to_table(table_name: str, dictionary: dict[str, str | bool | float]) -> None:
    """
    Adds a line to the wanted result table with a result informations and filename.
    """

    table_filepath = tables_path.joinpath(table_name + ".csv")
    tables_path.mkdir(exist_ok=True, parents=True)
    write = not table_filepath.exists()

    # Adds a line to the table (whether it exists or not)..
    with open(file=table_filepath, mode="a+", encoding="utf-8", newline="") as file:
        writer = DictWriter(file, dictionary.keys())
        if write:
            writer.writeheader()
        writer.writerow(dictionary)
