"""Matrix serialization, content hashing and NumPy probability computation."""
from __future__ import annotations

import hashlib
from typing import Any, Dict, List, Sequence, Tuple

import numpy as np

from .errors import ApiError

# On-disk element sizes; a healthy revision's blobs must be exactly these long.
FLOAT64_BYTES = 8
UINT8_BYTES = 1
INT64_BYTES = 8


def to_arrays(
    rain: List[List[float]], missing: List[List[int]] | None, rows: int, cols: int
) -> Tuple[np.ndarray, np.ndarray]:
    """Validate shape against the registered grid and return flat float64/uint8 arrays."""
    if len(rain) != rows or any(len(r) != cols for r in rain):
        raise ApiError(422, "GRID_MISMATCH", f"rain must have shape ({rows}, {cols})")
    rain_arr = np.asarray(rain, dtype=np.float64)
    if missing is None:
        miss_arr = np.zeros((rows, cols), dtype=np.uint8)
    else:
        if len(missing) != rows or any(len(r) != cols for r in missing):
            raise ApiError(422, "GRID_MISMATCH", f"missing must have shape ({rows}, {cols})")
        miss_arr = np.asarray(missing, dtype=np.uint8)
        if np.any((miss_arr != 0) & (miss_arr != 1)):
            raise ApiError(422, "INVALID_MISSING", "missing grid must contain only 0 or 1")
    valid = miss_arr == 0
    if np.any(rain_arr[valid] < 0):
        raise ApiError(422, "NEGATIVE_RAIN", "rain values must be non-negative at valid cells")
    return rain_arr.reshape(-1), miss_arr.reshape(-1)


def content_hash(rain_flat: np.ndarray, miss_flat: np.ndarray) -> str:
    h = hashlib.sha256()
    h.update(b"rain:")
    h.update(rain_flat.astype(np.float64, copy=False).tobytes())
    h.update(b"|miss:")
    h.update(miss_flat.astype(np.uint8, copy=False).tobytes())
    return h.hexdigest()


def compute_probabilities(
    rains: Sequence[np.ndarray],
    misses: Sequence[np.ndarray],
    thresholds: Sequence[float],
) -> Dict[float, Tuple[np.ndarray, np.ndarray]]:
    """Per-cell exceedance probability over valid (non-missing) members.

    p = (#valid members with rain >= threshold) / (#valid members).
    Cells with no valid member at all get denominator 0 and a null probability.
    Returns one (probability_float64, denominator_int64) pair per threshold.
    """
    stacked_rain = np.vstack([r.reshape(1, -1) for r in rains])  # (M, N)
    stacked_miss = np.vstack([m.reshape(1, -1) for m in misses]).astype(bool)
    valid_mask = ~stacked_miss  # (M, N)
    denom = valid_mask.sum(axis=0).astype(np.int64)  # (N,)
    out: Dict[float, Tuple[np.ndarray, np.ndarray]] = {}
    safe_denom = np.where(denom > 0, denom, 1).astype(np.float64)
    for threshold in thresholds:
        hits = np.where(valid_mask, stacked_rain >= float(threshold), False)
        count = hits.sum(axis=0).astype(np.float64)
        prob = count / safe_denom
        # cells with zero valid members are represented as NaN -> JSON null
        prob = np.where(denom > 0, prob, np.nan)
        out[float(threshold)] = (prob.astype(np.float64), denom.copy())
    return out


def matrix_to_json(flat: np.ndarray, rows: int, cols: int) -> List[List[float | int | None]]:
    grid = flat.reshape(rows, cols)
    result: List[List[float | int | None]] = []
    for row in grid:
        out_row: List[float | int | None] = []
        for v in row:
            fv = float(v)
            if np.isnan(fv):
                out_row.append(None)
            else:
                out_row.append(fv)
        result.append(out_row)
    return result


def encode_blob(arr: np.ndarray) -> bytes:
    return np.ascontiguousarray(arr).tobytes()


def expected_blob_bytes(n_cells: int) -> Tuple[int, int]:
    """Expected (rain_blob, miss_blob) byte lengths for a grid of ``n_cells`` cells."""
    return n_cells * FLOAT64_BYTES, n_cells * UINT8_BYTES


def verify_revision_blobs(
    rain_blob: bytes,
    miss_blob: bytes,
    n_cells: int,
    stored_hash: str,
) -> Dict[str, Any]:
    """Check an ingested revision against its warehouse summary.

    Two independent checks are reported, mirroring what a shared-volume
    migration can damage:

    * ``rain_length`` / ``miss_length``: the grid byte length still matches
      ``rows * cols`` times the element width (truncation / padding detection);
    * ``content_hash``: when both blobs have the correct length, recompute the
      SHA-256 over the actual bytes and compare with the summary stored at
      ingestion time (byte-flip detection). The hash check is skipped rather
      than faked when a length check already failed, so an unverified revision
      is never reported as healthy.

    Runs entirely outside any write transaction: hashing a large grid never
    blocks other forecast times from ingesting.
    """
    exp_rain, exp_miss = expected_blob_bytes(n_cells)
    rain_len_ok = len(rain_blob) == exp_rain
    miss_len_ok = len(miss_blob) == exp_miss

    checks: List[Dict[str, Any]] = [
        {
            "check": "rain_length",
            "status": "ok" if rain_len_ok else "damaged",
            "actual_bytes": len(rain_blob),
            "expected_bytes": exp_rain,
        },
        {
            "check": "miss_length",
            "status": "ok" if miss_len_ok else "damaged",
            "actual_bytes": len(miss_blob),
            "expected_bytes": exp_miss,
        },
    ]
    reasons: List[str] = []
    if not rain_len_ok:
        reasons.append(
            f"rain_blob length {len(rain_blob)} != expected {exp_rain}"
        )
    if not miss_len_ok:
        reasons.append(
            f"miss_blob length {len(miss_blob)} != expected {exp_miss}"
        )

    if rain_len_ok and miss_len_ok:
        rain_arr = np.frombuffer(rain_blob, dtype=np.float64, count=n_cells)
        miss_arr = np.frombuffer(miss_blob, dtype=np.uint8, count=n_cells)
        recomputed = content_hash(rain_arr, miss_arr)
        hash_ok = recomputed == stored_hash
        checks.append(
            {
                "check": "content_hash",
                "status": "ok" if hash_ok else "damaged",
                "stored_hash": stored_hash,
                "recomputed_hash": recomputed,
            }
        )
        if not hash_ok:
            reasons.append(
                f"content_hash mismatch: stored {stored_hash} != recomputed {recomputed}"
            )
    else:
        # Cannot recompute a fixed-shape hash over truncated/padded bytes;
        # report the check as skipped instead of claiming a verdict.
        checks.append(
            {
                "check": "content_hash",
                "status": "skipped_due_to_length",
                "stored_hash": stored_hash,
                "recomputed_hash": None,
            }
        )

    healthy = rain_len_ok and miss_len_ok and hash_ok if (rain_len_ok and miss_len_ok) else False
    return {
        "healthy": healthy,
        "checks": checks,
        "reasons": reasons,
    }


def decode_float(blob: bytes, n_cells: int) -> np.ndarray:
    return np.frombuffer(blob, dtype=np.float64, count=n_cells).copy()


def decode_int(blob: bytes, n_cells: int) -> np.ndarray:
    return np.frombuffer(blob, dtype=np.int64, count=n_cells).copy()


def decode_mask(blob: bytes, n_cells: int) -> np.ndarray:
    return np.frombuffer(blob, dtype=np.uint8, count=n_cells).copy()
