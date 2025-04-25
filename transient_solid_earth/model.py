"""
Because all models have to be identified.
"""

import dataclasses
from typing import Optional, TypeVar


@dataclasses.dataclass
class Model:
    """
    Abstract class for every model to have an ID.
    """

    model_id: Optional[str]


MODEL = TypeVar("MODEL", bound=Model)
