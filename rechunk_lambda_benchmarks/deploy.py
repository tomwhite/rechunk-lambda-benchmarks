"""The lambdas' deploy spec."""

from pathlib import Path

import map_lambda.deploy
from map_lambda.deploy import spec

SPEC = Path(__file__).with_name("deploy.toml")

PROJECT = map_lambda.deploy.project(SPEC)


def memory_sizes(function: str) -> tuple[int, ...]:
    """The sizes in MB `function` is deployed at, its default first."""
    (found,) = (f for f in spec.load(SPEC).functions if f.name == function)
    return (found.memory, *(size for size in found.sizes if size != found.memory))
