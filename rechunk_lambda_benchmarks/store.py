import functools
from pathlib import Path
from typing import Any
from urllib.parse import urlparse
from urllib.request import url2pathname

import boto3
import obstore as obs
from botocore.utils import InstanceMetadataRegionFetcher

IMDS_TIMEOUT_SECONDS = 1
IMDS_ATTEMPTS = 1


@functools.cache
def s3_region() -> str | None:
    """
    The AWS region for S3, as botocore resolves it, then from EC2
    instance metadata.
    """
    region: str | None = boto3.Session().region_name
    if region:
        return region
    try:
        return InstanceMetadataRegionFetcher(
            timeout=IMDS_TIMEOUT_SECONDS, num_attempts=IMDS_ATTEMPTS
        ).retrieve_region()
    except Exception:
        # not on EC2: leave it to obstore rather than fail here
        return None


@functools.cache
def _credentials() -> Any | None:
    """
    botocore's credential object, or None if it found nothing.
    """
    return boto3.Session().get_credentials()


def s3_credentials() -> dict[str, str]:
    """
    Credentials for obstore, resolved the way botocore resolves them,
    or empty to leave obstore to its own resolution.
    """
    credentials = _credentials()
    if credentials is None:
        return {}
    frozen = credentials.get_frozen_credentials()
    resolved = {
        "access_key_id": frozen.access_key,
        "secret_access_key": frozen.secret_key,
    }
    if frozen.token:
        resolved["token"] = frozen.token
    return resolved


def store_url(path: str | Path, *, mkdir: bool = False) -> str:
    """
    Normalise a store argument to a URL, since the lambdas parse it as
    one. A bare path becomes a `file://` URL, resolved first because
    `Path.as_uri` refuses a relative one.
    """
    if not isinstance(path, str | Path):
        raise TypeError(f"path must be a str or Path, got {type(path)!r}")
    if isinstance(path, Path) or "://" not in path:
        local = Path(path).resolve()
        if mkdir:
            local.mkdir(parents=True, exist_ok=True)
        return local.as_uri()
    if mkdir and urlparse(path).scheme == "file":
        Path(url2pathname(urlparse(path).path)).mkdir(parents=True, exist_ok=True)
    return path


def path_to_store(path: str | Path, *, mkdir: bool = False) -> obs.store.ObjectStore:
    """
    An obstore store for a path or URL. A bare path's directory is always
    created; a `file://` URL's only with `mkdir`.
    """
    local = isinstance(path, Path) or (isinstance(path, str) and "://" not in path)
    url = store_url(path, mkdir=mkdir or local)

    if urlparse(url).scheme == "s3":
        options: dict[str, Any] = dict(s3_credentials())
        region = s3_region()
        if region is not None:
            options["region"] = region
        return obs.store.from_url(url, **options)
    return obs.store.from_url(url)
