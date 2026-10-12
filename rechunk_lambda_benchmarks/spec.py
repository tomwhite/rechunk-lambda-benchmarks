"""
The array "spec": the subset of Zarr array metadata sent to the lambdas,
so an invocation need not GET zarr.json, and conversions to and from full
Zarr v3 metadata.

Only shape, chunks, data_type, fill_value and compressors are sent (plus
`shards` for a sharded array), rather than the whole document, which
would assume zarr-python and zarrs agree on all of it. Every other field
must take a default hardcoded identically here and in
`ArraySpec::to_metadata` in rust/common/src/lib.rs - keep the two in
sync. tests/fixtures/array_metadata.json is checked against both.
"""

import json
from typing import Any

import zarr

ArraySpec = dict[str, Any]

# The array-to-bytes codec: codecs[0], ahead of the compressors.
_SERIALIZER = {"name": "bytes", "configuration": {"endian": "little"}}

# Sharded, this is codecs[0] with the serializer inside it; the chunk
# grid then describes the shard and the codec's `chunk_shape` the chunk.
_SHARDING_CODEC = "sharding_indexed"

# zarr-python's defaults, hardcoded on both sides like `_SHARED_DEFAULTS`.
_SHARD_INDEX_CODECS = [_SERIALIZER, {"name": "crc32c"}]
_SHARD_INDEX_LOCATION = "end"

# Not sent: both sides hardcode these values, and any other is an error.
_SHARED_DEFAULTS: dict[str, Any] = {
    "zarr_format": 3,
    "node_type": "array",
    "chunk_key_encoding": {"name": "default", "configuration": {"separator": "/"}},
    "storage_transformers": [],
}

# Neither sent nor checked: they do not affect how chunks are encoded or
# keyed, and the lambda never writes zarr.json.
_IGNORED_FIELDS = frozenset({"attributes", "dimension_names"})

_SPEC_SOURCE_FIELDS = frozenset({"shape", "chunk_grid", "data_type", "fill_value", "codecs"})

_KNOWN_FIELDS = frozenset(_SHARED_DEFAULTS) | _IGNORED_FIELDS | _SPEC_SOURCE_FIELDS


def array_spec(array: zarr.Array) -> ArraySpec:
    """
    The wire spec for `array`. A sharded array has `chunks` and `shards`,
    as in zarr-python; an unsharded one has no `shards` key.

    Raises ValueError for anything the lambdas cannot faithfully
    reproduce: a non-"/" chunk key separator, a non-regular chunk grid,
    storage transformers, a sharding codec configured otherwise than both
    sides hardcode, or an unknown metadata field.
    """
    # through JSON: to_dict renders sequences as tuples, and JSON is what goes on the wire
    metadata = json.loads(json.dumps(array.metadata.to_dict()))

    if unknown := sorted(set(metadata) - _KNOWN_FIELDS):
        raise ValueError(
            f"unsupported Zarr array metadata field(s) {unknown}: this project only supports "
            f"arrays whose metadata is limited to {sorted(_KNOWN_FIELDS)}"
        )

    for field, expected in _SHARED_DEFAULTS.items():
        actual = metadata.get(field)
        if actual != expected:
            raise ValueError(
                f"Zarr array metadata {field}={actual!r}, but this project requires the shared "
                f"default {expected!r} (hardcoded in both rechunker_lambda/spec.py and "
                f"rust/common/src/lib.rs, so it isn't sent with every invocation)"
            )

    chunk_grid = metadata["chunk_grid"]
    if chunk_grid.get("name") != "regular":
        raise ValueError(
            f"unsupported chunk grid {chunk_grid.get('name')!r}: only 'regular' is supported"
        )

    grid_shape = list(chunk_grid["configuration"]["chunk_shape"])
    codecs = metadata["codecs"]
    sharded = bool(codecs) and codecs[0].get("name") == _SHARDING_CODEC
    if sharded:
        chunks, compressors = _sharding(codecs)
        shards: dict[str, Any] = {"shards": grid_shape}
    elif codecs and codecs[0] == _SERIALIZER:
        # verbatim: codecs are spec-defined on both sides, so a
        # representation of our own would drift more, not less
        chunks, compressors, shards = grid_shape, list(codecs[1:]), {}
    else:
        serializer = codecs[0] if codecs else None
        raise ValueError(
            f"unsupported serializer (codecs[0]) {serializer!r}: only {_SERIALIZER!r} and "
            f"{_SHARDING_CODEC!r} are supported"
        )

    return {
        "shape": list(metadata["shape"]),
        "chunks": chunks,
        **shards,
        "data_type": metadata["data_type"],
        "fill_value": metadata["fill_value"],
        "compressors": compressors,
    }


def _sharding(codecs: list[dict[str, Any]]) -> tuple[list[int], list[dict[str, Any]]]:
    """
    The inner chunk shape and compressors of a sharded array, checking
    the rest matches what both sides hardcode. The index must agree
    exactly: written differently, each side would read the other's
    shards as the wrong bytes rather than fail.
    """
    if len(codecs) > 1:
        raise ValueError(
            f"unsupported codecs after {_SHARDING_CODEC!r}: "
            f"{[codec.get('name') for codec in codecs[1:]]!r}. Compression belongs inside "
            f"the shard, where zarr-python puts it"
        )
    config = codecs[0].get("configuration", {})
    inner = config.get("codecs") or []
    if not inner or inner[0] != _SERIALIZER:
        found = inner[0] if inner else None
        raise ValueError(
            f"unsupported serializer inside the shard {found!r}: only {_SERIALIZER!r} is supported"
        )
    for field, expected in (
        ("index_codecs", _SHARD_INDEX_CODECS),
        ("index_location", _SHARD_INDEX_LOCATION),
    ):
        if config.get(field) != expected:
            raise ValueError(
                f"sharding {field}={config.get(field)!r}, but this project requires "
                f"{expected!r} (hardcoded in both rechunker_lambda/spec.py and "
                f"rust/common/src/lib.rs, so it isn't sent with every invocation)"
            )
    return list(config["chunk_shape"]), list(inner[1:])


def metadata_from_spec(spec: ArraySpec) -> dict[str, Any]:
    """
    The inverse of `array_spec`: full Zarr v3 metadata, with the shared
    defaults filled in. The counterpart of `ArraySpec::to_metadata` in
    rust/common/src/lib.rs, so in-process test workers build an array
    exactly as a real lambda does.
    """
    if shards := spec.get("shards"):
        grid_shape = list(shards)
        codecs = [
            {
                "name": _SHARDING_CODEC,
                "configuration": {
                    "chunk_shape": list(spec["chunks"]),
                    "codecs": [_SERIALIZER, *spec["compressors"]],
                    "index_codecs": _SHARD_INDEX_CODECS,
                    "index_location": _SHARD_INDEX_LOCATION,
                },
            }
        ]
    else:
        grid_shape = list(spec["chunks"])
        codecs = [_SERIALIZER, *spec["compressors"]]

    return {
        "shape": list(spec["shape"]),
        "data_type": spec["data_type"],
        "chunk_grid": {
            "name": "regular",
            "configuration": {"chunk_shape": grid_shape},
        },
        "fill_value": spec["fill_value"],
        "codecs": codecs,
        "attributes": {},
        **_SHARED_DEFAULTS,
    }
