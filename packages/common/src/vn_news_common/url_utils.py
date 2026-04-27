"""URL canonicalization + hashing for deduplication.

Strategy:

1. Lower-case the scheme and host.
2. Drop the URL fragment.
3. Strip well-known tracking query parameters
   (``utm_*``, ``fbclid``, ``gclid``, ``mc_cid``, ``mc_eid``, ``ref``, …).
4. Sort the remaining query parameters for stability.
5. Remove a trailing slash from the path *unless* the path is just ``/``.

The canonical form is then hashed with SHA-256 (truncated to 16 bytes
hex = 32 chars) for use as a database unique-index column.
"""

from __future__ import annotations

import hashlib
from urllib.parse import parse_qsl, urlencode, urlsplit, urlunsplit

_TRACKING_PREFIXES = (
    "utm_",
    "ga_",
    "yclid",
    "fbclid",
    "gclid",
    "mc_cid",
    "mc_eid",
    "vero_",
    "_ga",
    "ref",
    "ref_src",
    "ref_url",
    "spm",
    "share_source",
    "from",
    "src",
)


def canonicalize_url(url: str) -> str:
    """Return a normalized URL string suitable for deduplication."""
    if not url:
        return url

    parts = urlsplit(url.strip())

    scheme = parts.scheme.lower() or "https"
    netloc = parts.netloc.lower()

    # Drop default ports.
    if (scheme == "http" and netloc.endswith(":80")) or (
        scheme == "https" and netloc.endswith(":443")
    ):
        netloc = netloc.rsplit(":", 1)[0]

    # Strip tracking params, sort the rest.
    pairs = [
        (k, v)
        for k, v in parse_qsl(parts.query, keep_blank_values=False)
        if not _is_tracking_param(k)
    ]
    pairs.sort()
    query = urlencode(pairs, doseq=True)

    # Normalize path: collapse duplicate slashes, drop trailing slash.
    path = parts.path or "/"
    while "//" in path:
        path = path.replace("//", "/")
    if len(path) > 1 and path.endswith("/"):
        path = path[:-1]

    return urlunsplit((scheme, netloc, path, query, ""))


def _is_tracking_param(key: str) -> bool:
    k = key.lower()
    return any(k.startswith(p) for p in _TRACKING_PREFIXES)


def url_hash(url: str) -> str:
    """Return a stable 32-char hex hash of the *canonical* URL."""
    canonical = canonicalize_url(url)
    return hashlib.sha256(canonical.encode("utf-8")).hexdigest()[:32]
