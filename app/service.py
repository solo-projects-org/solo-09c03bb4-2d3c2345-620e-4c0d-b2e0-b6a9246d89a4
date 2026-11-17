"""Business logic: registration, member submission and product publishing.

Concurrency model (two API processes sharing one SQLite file)
----------------------------------------------------------------
* Every write happens inside a short ``BEGIN IMMEDIATE`` transaction, so only
  one writer exists at a time. Conflicting writers get ``SQLITE_BUSY`` after
  the busy timeout and receive HTTP 409/503 instead of corrupting state.
* Publishing reads a snapshot, runs the (potentially slow) NumPy computation
  *outside* any write transaction, then re-validates the revision set inside
  an immediate write transaction. The slow CPU work therefore never blocks
  other forecast times from ingesting members.
* A product row and its threshold rows are inserted in one transaction; any
  failure rolls the whole thing back, so no half product can ever be visible.
"""
from __future__ import annotations

import json
import os
import time
import uuid
from datetime import timezone
from typing import Any, Dict, List, Tuple

from .db import get_conn
from .errors import ApiError
from .schemas import RegisterRequest, SubmitRequest
from .storage import (
    compute_probabilities,
    content_hash,
    decode_float,
    decode_int,
    decode_mask,
    encode_blob,
    matrix_to_json,
    to_arrays,
    verify_revision_blobs,
)

# --- test hooks (disabled unless explicitly enabled; never on by default) ---
PUBLISH_DELAY_S = float(os.environ.get("RAIN_TEST_PUBLISH_DELAY_S", "0"))
FAIL_MARKER = os.environ.get("RAIN_TEST_FAIL_MARKER", "")
INSPECT_DELAY_S = float(os.environ.get("RAIN_TEST_INSPECT_DELAY_S", "0"))


def _is_busy(exc: Exception) -> bool:
    return isinstance(exc, Exception) and "SQLITE_BUSY" in str(exc)


# --------------------------------------------------------------------------- #
# Registration
# --------------------------------------------------------------------------- #
def register_group(payload: RegisterRequest) -> Dict[str, Any]:
    group_id = uuid.uuid4().hex
    thresholds_json = json.dumps(payload.thresholds)
    with get_conn() as conn:
        try:
            conn.execute("BEGIN IMMEDIATE")
            conn.execute(
                "INSERT INTO groups (group_id, forecast_time, rows, cols, thresholds)"
                " VALUES (?, ?, ?, ?, ?)",
                (
                    group_id,
                    payload.forecast_time.astimezone(timezone.utc).isoformat(),
                    payload.rows,
                    payload.cols,
                    thresholds_json,
                ),
            )
            conn.executemany(
                "INSERT INTO members (group_id, member_idx, member_id) VALUES (?, ?, ?)",
                [(group_id, idx, name) for idx, name in enumerate(payload.member_names)],
            )
            conn.execute("COMMIT")
        except Exception:
            conn.execute("ROLLBACK")
            raise
    return {
        "group_id": group_id,
        "forecast_time": payload.forecast_time.astimezone(timezone.utc).isoformat(),
        "rows": payload.rows,
        "cols": payload.cols,
        "members": [
            {"member_idx": idx, "member_id": name}
            for idx, name in enumerate(payload.member_names)
        ],
        "thresholds": payload.thresholds,
    }


def _get_group(conn, group_id: str) -> Dict[str, Any]:
    row = conn.execute(
        "SELECT * FROM groups WHERE group_id = ?", (group_id,)
    ).fetchone()
    if row is None:
        raise ApiError(404, "GROUP_NOT_FOUND", f"group {group_id!r} is not registered")
    d = dict(row)
    d["thresholds"] = json.loads(d["thresholds"])
    return d


def _get_member(conn, group_id: str, member_id: str) -> Dict[str, Any]:
    row = conn.execute(
        "SELECT member_idx, member_id FROM members WHERE group_id = ? AND member_id = ?",
        (group_id, member_id),
    ).fetchone()
    if row is None:
        raise ApiError(404, "MEMBER_NOT_FOUND", f"member {member_id!r} is not part of this group")
    return dict(row)


# --------------------------------------------------------------------------- #
# Member submission
# --------------------------------------------------------------------------- #
def submit_member(group_id: str, member_id: str, payload: SubmitRequest) -> Dict[str, Any]:
    with get_conn() as conn:
        group = _get_group(conn, group_id)
        member = _get_member(conn, group_id, member_id)
        rain_flat, miss_flat = to_arrays(
            payload.rain, payload.missing, group["rows"], group["cols"]
        )
        digest = content_hash(rain_flat, miss_flat)
        rain_blob = encode_blob(rain_flat)
        miss_blob = encode_blob(miss_flat)
        idx = member["member_idx"]

        try:
            conn.execute("BEGIN IMMEDIATE")
            prev = conn.execute(
                "SELECT revision, content_hash FROM member_revisions"
                " WHERE group_id = ? AND member_idx = ?"
                " ORDER BY revision DESC LIMIT 1",
                (group_id, idx),
            ).fetchone()
            duplicate = False
            if prev is not None:
                if payload.revision < prev["revision"]:
                    conn.execute("ROLLBACK")
                    raise ApiError(
                        409,
                        "REVISION_CONFLICT",
                        f"revision {payload.revision} is not greater than latest {prev['revision']}",
                    )
                if payload.revision == prev["revision"]:
                    if prev["content_hash"] == digest:
                        duplicate = True  # 同版同文重送：幂等，不重复入库
                    else:
                        conn.execute("ROLLBACK")
                        raise ApiError(
                            409,
                            "REVISION_CONTENT_CONFLICT",
                            f"revision {payload.revision} already exists with different content (异文拒收)",
                        )
            if not duplicate:
                conn.execute(
                    "INSERT INTO member_revisions"
                    " (group_id, member_idx, revision, content_hash, rain_blob, miss_blob)"
                    " VALUES (?, ?, ?, ?, ?, ?)",
                    (group_id, idx, payload.revision, digest, rain_blob, miss_blob),
                )
                # 更正成员：旧产品立即过期（历史行保留可查）
                conn.execute(
                    "UPDATE products SET is_current = 0 WHERE group_id = ? AND is_current = 1",
                    (group_id,),
                )
            conn.execute("COMMIT")
        except ApiError:
            raise
        except Exception:
            conn.execute("ROLLBACK")
            raise

    return {
        "group_id": group_id,
        "member_id": member_id,
        "member_idx": idx,
        "revision": payload.revision,
        "content_hash": digest,
        "duplicate": duplicate,
    }


# --------------------------------------------------------------------------- #
# Publishing
# --------------------------------------------------------------------------- #
def _read_snapshot(conn, group_id: str) -> Tuple[Dict[str, Any], List[Dict[str, Any]], Dict[int, int]]:
    group = _get_group(conn, group_id)
    member_rows = conn.execute(
        "SELECT member_idx, member_id FROM members WHERE group_id = ? ORDER BY member_idx",
        (group_id,),
    ).fetchall()
    latest: List[Dict[str, Any]] = []
    revision_map: Dict[int, int] = {}
    for m in member_rows:
        rev = conn.execute(
            "SELECT revision FROM member_revisions"
            " WHERE group_id = ? AND member_idx = ? ORDER BY revision DESC LIMIT 1",
            (group_id, m["member_idx"]),
        ).fetchone()
        if rev is None:
            raise ApiError(
                409,
                "MEMBERS_INCOMPLETE",
                f"member {m['member_id']!r} has no submission; all members must be present before publishing",
            )
        revision_map[int(m["member_idx"])] = int(rev["revision"])
        latest.append({"member_idx": int(m["member_idx"]), "member_id": m["member_id"]})
    return group, latest, revision_map


def _canonical_revision_map(revision_map: Dict[int, int]) -> str:
    return json.dumps(revision_map, sort_keys=True, separators=(",", ":"))


def publish_product(group_id: str) -> Dict[str, Any]:
    # Bounded outer loop: each iteration is one consistent revision-set attempt.
    for attempt in range(1, 4):
        # 1) snapshot under a read transaction (no write lock held during compute)
        with get_conn() as conn:
            try:
                conn.execute("BEGIN")
                group, members, revision_map = _read_snapshot(conn, group_id)
                revs = conn.execute(
                    "SELECT group_id, member_idx, revision, content_hash,"
                    " rain_blob, miss_blob"
                    " FROM member_revisions WHERE group_id = ?",
                    (group_id,),
                ).fetchall()
                conn.execute("COMMIT")
            except Exception:
                conn.execute("ROLLBACK")
                raise

        n_cells = group["rows"] * group["cols"]
        wanted = {
            (int(r["member_idx"]), int(r["revision"])): (
                r["content_hash"],
                r["rain_blob"],
                r["miss_blob"],
            )
            for r in revs
        }

        # 1b) publish gate: a damaged (e.g. migrated-volume corrupted) revision
        # can never back a warning product. Verification is CPU work done while
        # no write lock is held.
        damaged_publish: List[str] = []
        snapshot_hashes: Dict[Tuple[int, int], str] = {}
        for idx in sorted(revision_map):
            key = (idx, revision_map[idx])
            stored_hash, rain_blob, miss_blob = wanted[key]
            snapshot_hashes[key] = stored_hash
            verdict = verify_revision_blobs(
                rain_blob, miss_blob, n_cells, stored_hash
            )
            if not verdict["healthy"]:
                damaged_publish.append(
                    f"member_idx={idx} revision={revision_map[idx]}"
                )
        if damaged_publish:
            raise ApiError(
                409,
                "REVISION_DAMAGED",
                "refusing to publish: stored grid bytes fail the warehouse"
                " summary check for " + "; ".join(damaged_publish),
            )

        with get_conn() as conn:
            existing = conn.execute(
                "SELECT product_id FROM products"
                " WHERE group_id = ? AND revision_map = ? AND is_current = 1",
                (group_id, _canonical_revision_map(revision_map)),
            ).fetchone()
        if existing is not None:
            return _load_product(existing["product_id"])

        rains = [
            decode_float(wanted[(idx, revision_map[idx])][1], n_cells)
            for idx in sorted(revision_map)
        ]
        misses = [
            decode_mask(wanted[(idx, revision_map[idx])][2], n_cells)
            for idx in sorted(revision_map)
        ]

        # 2) slow compute happens OUTSIDE the write lock: other groups keep ingesting
        if PUBLISH_DELAY_S > 0:
            time.sleep(PUBLISH_DELAY_S)
        if FAIL_MARKER and os.path.exists(FAIL_MARKER):
            # deterministic failure injection for atomicity self-tests
            raise ApiError(500, "COMPUTE_FAILED", "injected computation failure (test hook)")
        try:
            per_threshold = compute_probabilities(rains, misses, group["thresholds"])
        except ApiError:
            raise
        except Exception as exc:  # calculation failure must not leave half a result
            raise ApiError(500, "COMPUTE_FAILED", f"probability computation failed: {exc}") from exc

        canonical = _canonical_revision_map(revision_map)

        # 3) re-validate under the write lock and persist atomically
        write_attempts = 0
        should_retry_snapshot = False
        with get_conn() as conn:
            while True:
                write_attempts += 1
                try:
                    conn.execute("BEGIN IMMEDIATE")
                    latest_map: Dict[int, int] = {}
                    latest_hashes: Dict[int, str] = {}
                    for m in members:
                        rev = conn.execute(
                            "SELECT revision, content_hash FROM member_revisions"
                            " WHERE group_id = ? AND member_idx = ? ORDER BY revision DESC LIMIT 1",
                            (group_id, m["member_idx"]),
                        ).fetchone()
                        if rev is None:
                            conn.execute("ROLLBACK")
                            raise ApiError(409, "MEMBERS_INCOMPLETE", "member disappeared before publish commit")
                        idx = int(m["member_idx"])
                        latest_map[idx] = int(rev["revision"])
                        latest_hashes[idx] = rev["content_hash"]

                    if latest_map != revision_map:
                        # a correction landed while we were computing: this product
                        # would be stale; retry from the new consistent revision set
                        conn.execute("ROLLBACK")
                        if attempt >= 3:
                            raise ApiError(
                                409,
                                "REVISION_CHANGED",
                                "member revisions changed during publish; retry the request",
                            )
                        should_retry_snapshot = True
                        break

                    if any(
                        latest_hashes[idx] != snapshot_hashes[(idx, revision_map[idx])]
                        for idx in revision_map
                    ):
                        # the bytes/summary of a revision we verified changed on
                        # disk before commit: re-snapshot so the gate re-verifies
                        conn.execute("ROLLBACK")
                        if attempt >= 3:
                            raise ApiError(
                                409,
                                "REVISION_DAMAGED",
                                "stored grid bytes changed during publish; retry the request",
                            )
                        should_retry_snapshot = True
                        break

                    row = conn.execute(
                        "SELECT product_id, is_current FROM products"
                        " WHERE group_id = ? AND revision_map = ?",
                        (group_id, canonical),
                    ).fetchone()
                    if row is not None and int(row["is_current"]) == 1:
                        conn.execute("ROLLBACK")
                        return _load_product(int(row["product_id"]))

                    # supersede every previous product of this group (history retained)
                    conn.execute(
                        "UPDATE products SET is_current = 0 WHERE group_id = ? AND is_current = 1",
                        (group_id,),
                    )
                    if row is not None:  # same revision map existed but was superseded: revive it
                        product_id = int(row["product_id"])
                        conn.execute(
                            "UPDATE products SET is_current = 1 WHERE product_id = ?", (product_id,)
                        )
                    else:
                        cur = conn.execute(
                            "INSERT INTO products (group_id, revision_map, is_current)"
                            " VALUES (?, ?, 1)",
                            (group_id, canonical),
                        )
                        product_id = int(cur.lastrowid)
                        conn.executemany(
                            "INSERT INTO product_thresholds"
                            " (product_id, threshold, probability_blob, denom_blob)"
                            " VALUES (?, ?, ?, ?)",
                            [
                                (
                                    product_id,
                                    float(thr),
                                    encode_blob(per_threshold[float(thr)][0]),
                                    encode_blob(per_threshold[float(thr)][1]),
                                )
                                for thr in group["thresholds"]
                            ],
                        )
                    conn.execute("COMMIT")
                    return _load_product(product_id)
                except ApiError:
                    if conn.in_transaction:
                        conn.execute("ROLLBACK")
                    raise
                except Exception as exc:
                    if conn.in_transaction:
                        conn.execute("ROLLBACK")
                    if _is_busy(exc) and write_attempts < 3:
                        time.sleep(0.05 * write_attempts)
                        continue
                    if _is_busy(exc):
                        raise ApiError(503, "DATABASE_BUSY", "database locked by another writer; retry") from exc
                    raise
        if should_retry_snapshot:
            continue

    raise ApiError(409, "REVISION_CHANGED", "member revisions kept changing during publish; retry the request")


# --------------------------------------------------------------------------- #
# Storage inspection (post volume-migration integrity patrol)
# --------------------------------------------------------------------------- #
def inspect_group(group_id: str) -> Dict[str, Any]:
    """Patrol the *latest* revision of every member of one forecast time.

    Flow (mirrors the publish concurrency design):

    1. short read transaction: snapshot each member's latest revision and its
       stored blobs/summary, then release all locks;
    2. outside any transaction: verify grid byte lengths and recompute the
       SHA-256 over the actual bytes (slow on big grids, but never holding the
       write lock, so other times keep ingesting);
    3. short ``BEGIN IMMEDIATE`` write transaction: re-read the latest revisions
       and only apply verdicts that are still current. A member whose revision
       moved on during hashing is reported ``unverified``; a stale damage
       verdict can never revoke a healthy newer product. Current products whose
       revision set contains a confirmed damaged revision lose their current
       qualification in the same transaction (history rows are retained).
    """
    for attempt in range(1, 4):
        # ---- 1) snapshot (short read tx, no write lock) ----
        with get_conn() as conn:
            try:
                conn.execute("BEGIN")
                group = _get_group(conn, group_id)
                member_rows = conn.execute(
                    "SELECT member_idx, member_id FROM members"
                    " WHERE group_id = ? ORDER BY member_idx",
                    (group_id,),
                ).fetchall()
                n_cells = group["rows"] * group["cols"]
                snapshot: Dict[int, Dict[str, Any]] = {}
                for m in member_rows:
                    rev = conn.execute(
                        "SELECT revision, content_hash, rain_blob, miss_blob"
                        " FROM member_revisions"
                        " WHERE group_id = ? AND member_idx = ?"
                        " ORDER BY revision DESC LIMIT 1",
                        (group_id, m["member_idx"]),
                    ).fetchone()
                    if rev is not None:
                        snapshot[int(m["member_idx"])] = {
                            "member_id": m["member_id"],
                            "revision": int(rev["revision"]),
                            "content_hash": rev["content_hash"],
                            "rain_blob": rev["rain_blob"],
                            "miss_blob": rev["miss_blob"],
                        }
                    else:
                        snapshot[int(m["member_idx"])] = {"member_id": m["member_id"], "revision": None}
                conn.execute("COMMIT")
            except Exception:
                conn.execute("ROLLBACK")
                raise

        # ---- 2) heavy verification OUTSIDE every transaction/lock ----
        if INSPECT_DELAY_S > 0:
            time.sleep(INSPECT_DELAY_S)  # test hook: widen the interleaving window
        verdicts: Dict[int, Dict[str, Any]] = {}
        for idx, item in snapshot.items():
            if item["revision"] is None:
                continue
            verdicts[idx] = verify_revision_blobs(
                item["rain_blob"], item["miss_blob"], n_cells, item["content_hash"]
            )

        # ---- 3) re-confirm under the write lock and revoke atomically ----
        write_attempts = 0
        retry_snapshot = False
        with get_conn() as conn:
            while True:
                write_attempts += 1
                try:
                    conn.execute("BEGIN IMMEDIATE")
                    current_state: Dict[int, Dict[str, Any]] = {}
                    for m in member_rows:
                        rev = conn.execute(
                            "SELECT revision, content_hash FROM member_revisions"
                            " WHERE group_id = ? AND member_idx = ?"
                            " ORDER BY revision DESC LIMIT 1",
                            (group_id, m["member_idx"]),
                        ).fetchone()
                        idx = int(m["member_idx"])
                        if rev is None:
                            current_state[idx] = {"revision": None, "content_hash": None}
                        else:
                            current_state[idx] = {
                                "revision": int(rev["revision"]),
                                "content_hash": rev["content_hash"],
                            }

                    damaged_pairs: List[Tuple[int, int]] = []
                    member_reports: List[Dict[str, Any]] = []
                    n_damaged = n_unverified = 0
                    for m in member_rows:
                        idx = int(m["member_idx"])
                        member_id = m["member_id"]
                        snap = snapshot[idx]
                        cur = current_state[idx]
                        base = {
                            "member_idx": idx,
                            "member_id": member_id,
                            "inspected_revision": snap["revision"],
                            "latest_revision": cur["revision"],
                        }
                        if snap["revision"] is None:
                            # nothing ingested when the patrol started
                            if cur["revision"] is None:
                                base["status"] = "no_revision"
                            else:
                                # landed while we were hashing: never claim health
                                base["status"] = "unverified"
                                base["latest_verified"] = False
                                n_unverified += 1
                            member_reports.append(base)
                            continue

                        same_revision = cur["revision"] == snap["revision"]
                        same_hash = cur["content_hash"] == snap["content_hash"]
                        if not same_revision:
                            # a new revision landed while we were hashing: the
                            # old verdict must not act on the new one, and the
                            # new version itself is not verified in this patrol
                            base["status"] = "unverified"
                            base["latest_verified"] = False
                            n_unverified += 1
                            old_verdict = verdicts[idx]
                            base["superseded_inspection"] = {
                                "revision": snap["revision"],
                                "healthy": old_verdict["healthy"],
                                "checks": old_verdict["checks"],
                                "reasons": old_verdict["reasons"],
                            }
                            member_reports.append(base)
                            continue
                        if not same_hash:
                            # same revision number but bytes/summary changed
                            # during hashing: discard the race-tainted verdict
                            # and re-patrol from a fresh snapshot
                            conn.execute("ROLLBACK")
                            if attempt >= 3:
                                raise ApiError(
                                    409,
                                    "REVISION_CHANGED",
                                    "stored grid bytes changed during inspection; retry the request",
                                )
                            retry_snapshot = True
                            break

                        verdict = verdicts[idx]
                        base["status"] = "healthy" if verdict["healthy"] else "damaged"
                        base["latest_verified"] = True
                        base["checks"] = verdict["checks"]
                        base["reasons"] = verdict["reasons"]
                        base["stored_hash"] = snap["content_hash"]
                        if verdict["healthy"]:
                            member_reports.append(base)
                        else:
                            n_damaged += 1
                            damaged_pairs.append((idx, snap["revision"]))
                            base["damage"] = {
                                "revision": snap["revision"],
                                "reasons": verdict["reasons"],
                                "checks": verdict["checks"],
                            }
                            member_reports.append(base)
                    if retry_snapshot:
                        break

                    # Revoke current qualification of every current product that
                    # references a revision now confirmed damaged. Rows stay in
                    # the table (history retained); only is_current flips to 0.
                    revoked: List[Dict[str, Any]] = []
                    if damaged_pairs:
                        current_products = conn.execute(
                            "SELECT product_id, revision_map FROM products"
                            " WHERE group_id = ? AND is_current = 1",
                            (group_id,),
                        ).fetchall()
                        for prow in current_products:
                            rev_map = {
                                int(k): int(v)
                                for k, v in json.loads(prow["revision_map"]).items()
                            }
                            hits = [
                                {"member_idx": idx, "revision": rev}
                                for idx, rev in damaged_pairs
                                if rev_map.get(idx) == rev
                            ]
                            if hits:
                                conn.execute(
                                    "UPDATE products SET is_current = 0 WHERE product_id = ?",
                                    (prow["product_id"],),
                                )
                                revoked.append(
                                    {
                                        "product_id": int(prow["product_id"]),
                                        "revision_map": rev_map,
                                        "damaged_references": hits,
                                    }
                                )
                    conn.execute("COMMIT")
                    damaged_members = [
                        {
                            "member_idx": rep["member_idx"],
                            "member_id": rep["member_id"],
                            "revision": rep["damage"]["revision"],
                            "reasons": rep["damage"]["reasons"],
                            "checks": rep["damage"]["checks"],
                        }
                        for rep in member_reports
                        if rep["status"] == "damaged"
                    ]
                    return {
                        "group_id": group_id,
                        "forecast_time": group["forecast_time"],
                        "inspected_at": _now_iso(),
                        "summary": {
                            "members_total": len(member_rows),
                            "members_healthy": sum(
                                1 for r in member_reports if r["status"] == "healthy"
                            ),
                            "members_damaged": n_damaged,
                            "members_unverified": n_unverified,
                            "members_without_revision": sum(
                                1 for r in member_reports if r["status"] == "no_revision"
                            ),
                            "products_revoked": len(revoked),
                            "all_latest_healthy": n_damaged == 0
                            and n_unverified == 0
                            and all(
                                r["status"] == "healthy" for r in member_reports
                            ),
                        },
                        "members": member_reports,
                        "damaged_members": damaged_members,
                        "revoked_products": revoked,
                    }
                except ApiError:
                    if conn.in_transaction:
                        conn.execute("ROLLBACK")
                    raise
                except Exception as exc:
                    if conn.in_transaction:
                        conn.execute("ROLLBACK")
                    if _is_busy(exc) and write_attempts < 3:
                        time.sleep(0.05 * write_attempts)
                        continue
                    if _is_busy(exc):
                        raise ApiError(
                            503, "DATABASE_BUSY",
                            "database locked by another writer; retry the inspection",
                        ) from exc
                    raise
        if retry_snapshot:
            continue

    raise ApiError(503, "DATABASE_BUSY", "inspection could not obtain a consistent snapshot; retry")


def _now_iso() -> str:
    from datetime import datetime

    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%S.%f")[:-3] + "Z"


# --------------------------------------------------------------------------- #
# Queries
# --------------------------------------------------------------------------- #
def _load_product(product_id: int) -> Dict[str, Any]:
    with get_conn() as conn:
        prow = conn.execute(
            "SELECT * FROM products WHERE product_id = ?", (product_id,)
        ).fetchone()
        if prow is None:
            raise ApiError(404, "PRODUCT_NOT_FOUND", f"product {product_id} not found")
        group = _get_group(conn, prow["group_id"])
        rows, cols = group["rows"], group["cols"]
        n = rows * cols
        revision_map = {int(k): v for k, v in json.loads(prow["revision_map"]).items()}
        member_rows = conn.execute(
            "SELECT member_idx, member_id FROM members WHERE group_id = ? ORDER BY member_idx",
            (prow["group_id"],),
        ).fetchall()
        thresholds_out: List[Dict[str, Any]] = []
        for trow in conn.execute(
            "SELECT threshold, probability_blob, denom_blob"
            " FROM product_thresholds WHERE product_id = ? ORDER BY threshold",
            (product_id,),
        ):
            prob = decode_float(trow["probability_blob"], n)
            denom = decode_int(trow["denom_blob"], n)
            thresholds_out.append(
                {
                    "threshold": float(trow["threshold"]),
                    "probability": matrix_to_json(prob, rows, cols),
                    "denominator": denom.reshape(rows, cols).tolist(),
                }
            )
        member_revisions = [
            {
                "member_idx": int(m["member_idx"]),
                "member_id": m["member_id"],
                "revision": int(revision_map[int(m["member_idx"])]),
            }
            for m in member_rows
        ]
        return {
            "product_id": int(prow["product_id"]),
            "group_id": prow["group_id"],
            "forecast_time": group["forecast_time"],
            "is_current": bool(prow["is_current"]),
            "created_at": prow["created_at"],
            "revision_map": revision_map,
            "members_used": member_revisions,
            "thresholds": thresholds_out,
        }


def get_current_product(group_id: str) -> Dict[str, Any]:
    with get_conn() as conn:
        _get_group(conn, group_id)  # 404 on unknown group
        row = conn.execute(
            "SELECT product_id FROM products WHERE group_id = ? AND is_current = 1",
            (group_id,),
        ).fetchone()
        if row is None:
            raise ApiError(
                404,
                "NO_CURRENT_PRODUCT",
                "no current product for this group (publish first, or the previous product was superseded)",
            )
        product_id = int(row["product_id"])
    return _load_product(product_id)


def list_products(group_id: str) -> Dict[str, Any]:
    with get_conn() as conn:
        _get_group(conn, group_id)
        rows = conn.execute(
            "SELECT product_id, revision_map, is_current, created_at"
            " FROM products WHERE group_id = ? ORDER BY product_id",
            (group_id,),
        ).fetchall()
        return {
            "group_id": group_id,
            "products": [
                {
                    "product_id": int(r["product_id"]),
                    "revision_map": {int(k): v for k, v in json.loads(r["revision_map"]).items()},
                    "is_current": bool(r["is_current"]),
                    "created_at": r["created_at"],
                }
                for r in rows
            ],
        }


def get_product(group_id: str, product_id: int) -> Dict[str, Any]:
    product = _load_product(product_id)
    if product["group_id"] != group_id:
        raise ApiError(404, "PRODUCT_NOT_FOUND", f"product {product_id} does not belong to this group")
    return product


def get_group_status(group_id: str) -> Dict[str, Any]:
    with get_conn() as conn:
        group = _get_group(conn, group_id)
        members = conn.execute(
            "SELECT member_idx, member_id FROM members WHERE group_id = ? ORDER BY member_idx",
            (group_id,),
        ).fetchall()
        out_members = []
        complete = True
        for m in members:
            rev = conn.execute(
                "SELECT revision, content_hash, created_at FROM member_revisions"
                " WHERE group_id = ? AND member_idx = ? ORDER BY revision DESC LIMIT 1",
                (group_id, m["member_idx"]),
            ).fetchone()
            if rev is None:
                complete = False
                out_members.append(
                    {"member_idx": int(m["member_idx"]), "member_id": m["member_id"], "latest_revision": None}
                )
            else:
                out_members.append(
                    {
                        "member_idx": int(m["member_idx"]),
                        "member_id": m["member_id"],
                        "latest_revision": int(rev["revision"]),
                        "content_hash": rev["content_hash"],
                        "updated_at": rev["created_at"],
                    }
                )
        current = conn.execute(
            "SELECT product_id FROM products WHERE group_id = ? AND is_current = 1",
            (group_id,),
        ).fetchone()
        return {
            "group_id": group_id,
            "forecast_time": group["forecast_time"],
            "rows": group["rows"],
            "cols": group["cols"],
            "thresholds": group["thresholds"],
            "members": out_members,
            "all_members_present": complete,
            "current_product_id": int(current["product_id"]) if current else None,
        }


def list_groups() -> Dict[str, Any]:
    with get_conn() as conn:
        rows = conn.execute(
            "SELECT group_id, forecast_time, rows, cols, thresholds, created_at"
            " FROM groups ORDER BY created_at"
        ).fetchall()
        return {
            "groups": [
                {
                    "group_id": r["group_id"],
                    "forecast_time": r["forecast_time"],
                    "rows": r["rows"],
                    "cols": r["cols"],
                    "thresholds": json.loads(r["thresholds"]),
                    "created_at": r["created_at"],
                }
                for r in rows
            ]
        }
