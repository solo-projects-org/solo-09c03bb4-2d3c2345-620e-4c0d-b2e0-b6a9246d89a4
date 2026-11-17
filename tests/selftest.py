#!/usr/bin/env python3
"""End-to-end self test for the ensemble rainfall probability API.

By default the script spawns TWO uvicorn processes sharing one SQLite file
(dual-instance deployment), waits for health, and exercises every key flow:

  * registration / member-count / threshold / datetime validation
  * idempotent resubmit (same revision+content) and conflicting resubmit rejection
  * revision ordering, grid-shape and non-negativity checks
  * publish blocked until every member is present
  * probability / denominator / all-missing null semantics and used revisions
  * correction immediately expires the old product while history stays queryable
  * two-instance interleaved publish vs. correction -> only consistent products
  * parallel duplicate publishes collapse to one idempotent product
  * injected compute failure leaves no half-written product (atomicity)
  * slow computation never blocks another forecast time's ingestion
  * storage inspection: healthy baseline, hash/length damage detection,
    same-transaction revocation of current qualification with history retained,
    REVISION_DAMAGED publish gate, healthy higher-revision repair + re-publish,
    and interleaved inspection vs ingestion/republish (unverified never reported
    healthy; stale verdicts never revoke healthy new products)
  * Swagger UI / OpenAPI availability

Only the Python standard library is required.
"""
from __future__ import annotations

import json
import os
import signal
import subprocess
import sys
import tempfile
import threading
import time
import urllib.error
import urllib.request
from typing import Any, Dict, Optional, Tuple

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
PASS_COUNT = 0


def check(name: str, cond: bool, detail: str = "") -> None:
    global PASS_COUNT
    if not cond:
        raise AssertionError(f"FAIL: {name} {detail}")
    PASS_COUNT += 1
    print(f"  PASS  {name}")


def http(method: str, base: str, path: str, body: Optional[dict] = None) -> Tuple[int, Any]:
    data = None
    headers = {"Accept": "application/json"}
    if body is not None:
        data = json.dumps(body).encode()
        headers["Content-Type"] = "application/json"
    req = urllib.request.Request(base + path, data=data, headers=headers, method=method)
    try:
        with urllib.request.urlopen(req, timeout=30) as resp:
            return resp.status, json.loads(resp.read().decode())
    except urllib.error.HTTPError as exc:
        raw = exc.read().decode()
        try:
            return exc.code, json.loads(raw)
        except json.JSONDecodeError:
            return exc.code, {"raw": raw}


def approx(a: Any, b: Any, tol: float = 1e-9) -> bool:
    if a is None and b is None:
        return True
    if a is None or b is None:
        return False
    return abs(float(a) - float(b)) <= tol


def matrix_approx(a: list, b: list) -> bool:
    if len(a) != len(b):
        return False
    for ra, rb in zip(a, b):
        if len(ra) != len(rb):
            return False
        for x, y in zip(ra, rb):
            if not approx(x, y):
                return False
    return True


# --------------------------------------------------------------------------- #
# dual-instance process management
# --------------------------------------------------------------------------- #
def spawn_servers() -> Tuple[list, str, str, str, str]:
    tmp = tempfile.mkdtemp(prefix="rain-selftest-")
    db_path = os.path.join(tmp, "rain.db")
    fail_marker = os.path.join(tmp, "fail.marker")
    env = dict(os.environ)
    env.update(
        RAIN_DB_PATH=db_path,
        RAIN_TEST_PUBLISH_DELAY_S="0.6",
        RAIN_TEST_INSPECT_DELAY_S="0.7",
        RAIN_TEST_FAIL_MARKER=fail_marker,
        PYTHONPATH=ROOT,
    )
    procs = []
    for port in (8091, 8092):
        log = open(os.path.join(tmp, f"uvicorn-{port}.log"), "w")
        procs.append(
            subprocess.Popen(
                [sys.executable, "-m", "uvicorn", "app.main:app",
                 "--host", "127.0.0.1", "--port", str(port)],
                cwd=ROOT, env=env, stdout=log, stderr=subprocess.STDOUT,
            )
        )
    base_a, base_b = "http://127.0.0.1:8091", "http://127.0.0.1:8092"
    deadline = time.time() + 20
    while time.time() < deadline:
        try:
            s1, _ = http("GET", base_a, "/health")
            s2, _ = http("GET", base_b, "/health")
            if s1 == 200 and s2 == 200:
                return procs, base_a, base_b, fail_marker, tmp
        except Exception:
            pass
        time.sleep(0.2)
    raise RuntimeError("servers did not become healthy")


def shutdown(procs: list) -> None:
    for p in procs:
        p.send_signal(signal.SIGINT)
    for p in procs:
        try:
            p.wait(timeout=10)
        except subprocess.TimeoutExpired:
            p.kill()


def main() -> int:
    base_a = os.environ.get("RAIN_BASE_A")
    base_b = os.environ.get("RAIN_BASE_B")
    fail_marker = os.environ.get("RAIN_TEST_FAIL_MARKER", "")
    db_path = os.environ.get("RAIN_DB_PATH", "")
    spawned = False
    procs: list = []
    if not base_a or not base_b:
        procs, base_a, base_b, fail_marker, tmp = spawn_servers()
        db_path = os.path.join(tmp, "rain.db")
        spawned = True
        print(f"[setup] spawned dual instances {base_a} / {base_b}")
    hooks_enabled = bool(fail_marker)
    try:
        run(base_a, base_b, fail_marker, hooks_enabled, db_path)
    finally:
        if spawned:
            shutdown(procs)
    print(f"\nALL TESTS PASSED ({PASS_COUNT} assertions)")
    return 0


def register(base: str, members: int = 3, rows: int = 2, cols: int = 2,
             thresholds=(5.0, 10.0), names=None) -> dict:
    if names is None:
        names = [f"m{i}" for i in range(1, members + 1)]
    body = {
        "forecast_time": "2026-11-16T00:00:00Z",
        "rows": rows, "cols": cols,
        "member_names": names,
        "thresholds": list(thresholds),
    }
    status, data = http("POST", base, "/groups", body)
    assert status == 201, (status, data)
    return data


def run(base_a: str, base_b: str, fail_marker: str, hooks_enabled: bool,
        db_path: str = "") -> None:
    print("\n[1] validation on registration")
    bad = {"forecast_time": "2026-11-16T00:00:00Z", "rows": 2, "cols": 2,
           "member_names": ["only-one"], "thresholds": [5.0]}
    s, d = http("POST", base_a, "/groups", bad)
    check("1 member rejected (need 2-8)", s == 422, str(d))
    bad["member_names"] = [f"m{i}" for i in range(9)]
    s, d = http("POST", base_a, "/groups", bad)
    check("9 members rejected", s == 422)
    bad["member_names"] = ["a", "b"]
    bad["thresholds"] = [-1.0]
    s, d = http("POST", base_a, "/groups", bad)
    check("negative threshold rejected", s == 422)
    bad["thresholds"] = [5.0]
    bad["forecast_time"] = "2026-11-16T00:00:00"  # naive, no timezone
    s, d = http("POST", base_a, "/groups", bad)
    check("naive datetime rejected", s == 422)

    print("\n[2] happy-path group + member submission semantics")
    g = register(base_a, members=3, thresholds=[5.0, 10.0])
    gid = g["group_id"]
    check("registered 3 members", len(g["members"]) == 3)
    s, d = http("GET", base_b, f"/groups/{gid}")
    check("group visible from instance B", s == 200 and d["all_members_present"] is False)

    rain1 = [[0.0, 10.0], [20.0, 0.0]]
    s, d = http("PUT", base_a, f"/groups/{gid}/members/m1",
                {"rain": rain1, "revision": 1})
    check("m1 rev1 accepted", s == 200 and d["duplicate"] is False, str(d))
    s, d2 = http("PUT", base_b, f"/groups/{gid}/members/m1",
                 {"rain": rain1, "revision": 1})
    check("same revision+content resent is idempotent", s == 200 and d2["duplicate"] is True)
    check("idempotent hash stable", d["content_hash"] == d2["content_hash"])
    s, d = http("PUT", base_a, f"/groups/{gid}/members/m1",
                {"rain": [[0.0, 10.0], [20.0, 1.0]], "revision": 1})
    check("same revision different content rejected (异文拒收)",
          s == 409 and d["error"]["code"] == "REVISION_CONTENT_CONFLICT", str(d))
    s, d = http("PUT", base_a, f"/groups/{gid}/members/ghost",
                {"rain": rain1, "revision": 1})
    check("unknown member -> 404", s == 404 and d["error"]["code"] == "MEMBER_NOT_FOUND")
    s, d = http("PUT", base_a, "/groups/nope/members/m1", {"rain": rain1, "revision": 1})
    check("unknown group -> 404", s == 404 and d["error"]["code"] == "GROUP_NOT_FOUND")

    print("\n[3] revision ordering and grid validation on a second group")
    g2 = register(base_b, members=2, thresholds=[5.0])
    gid2 = g2["group_id"]
    s, d = http("PUT", base_a, f"/groups/{gid2}/members/m1",
                {"rain": [[1.0, 2.0], [3.0, 4.0]], "revision": 1})
    check("g2 m1 rev1", s == 200)
    s, d = http("PUT", base_b, f"/groups/{gid2}/members/m1",
                {"rain": [[9.0, 9.0], [9.0, 9.0]], "revision": 2})
    check("g2 m1 rev2", s == 200)
    s, d = http("PUT", base_a, f"/groups/{gid2}/members/m1",
                {"rain": [[1.0, 2.0], [3.0, 4.0]], "revision": 1})
    check("lower revision rejected", s == 409 and d["error"]["code"] == "REVISION_CONFLICT")
    s, d = http("PUT", base_a, f"/groups/{gid2}/members/m1",
                {"rain": [[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]], "revision": 3})
    check("grid shape mismatch rejected", s == 422 and d["error"]["code"] == "GRID_MISMATCH")
    s, d = http("PUT", base_a, f"/groups/{gid2}/members/m1",
                {"rain": [[-1.0, 2.0], [3.0, 4.0]], "revision": 3})
    check("negative rain rejected", s == 422 and d["error"]["code"] == "NEGATIVE_RAIN")
    s, d = http("PUT", base_a, f"/groups/{gid2}/members/m1",
                {"rain": [[1.0, 2.0], [3.0, 4.0]], "missing": [[1, 0, 0]], "revision": 3})
    check("missing grid shape mismatch rejected",
          s == 422 and d["error"]["code"] == "GRID_MISMATCH", str(d))

    print("\n[4] publishing requires every member")
    s, d = http("POST", base_a, f"/groups/{gid}/publish")
    check("incomplete group cannot publish",
          s == 409 and d["error"]["code"] == "MEMBERS_INCOMPLETE", str(d))

    print("\n[5] complete group and verify probability math")
    rain2 = [[10.0, 10.0], [0.0, 0.0]]
    rain3 = [[5.0, 0.0], [99.0, 0.0]]
    miss3 = [[0, 0], [1, 1]]  # m3 missing on second row
    s, _ = http("PUT", base_b, f"/groups/{gid}/members/m2",
                {"rain": rain2, "revision": 1})
    check("m2 rev1 via instance B", s == 200)
    s, _ = http("PUT", base_a, f"/groups/{gid}/members/m3",
                {"rain": rain3, "missing": miss3, "revision": 1})
    check("m3 rev1 with missing grid", s == 200)

    s, p1 = http("POST", base_a, f"/groups/{gid}/publish")
    check("publish succeeds when complete", s == 200, str(p1))
    check("product marked current", p1["is_current"] is True)
    check("3 member revisions recorded",
          {m["member_id"]: m["revision"] for m in p1["members_used"]}
          == {"m1": 1, "m2": 1, "m3": 1})
    by_thr = {t["threshold"]: t for t in p1["thresholds"]}
    check("denominators exclude missing members",
          by_thr[5.0]["denominator"] == [[3, 3], [2, 2]],
          str(by_thr[5.0]["denominator"]))
    check("probability threshold=5",
          matrix_approx(by_thr[5.0]["probability"],
                        [[2 / 3, 2 / 3], [0.5, 0.0]]),
          str(by_thr[5.0]["probability"]))
    check("probability threshold=10",
          matrix_approx(by_thr[10.0]["probability"],
                        [[1 / 3, 2 / 3], [0.5, 0.0]]),
          str(by_thr[10.0]["probability"]))
    s, p1b = http("POST", base_b, f"/groups/{gid}/publish")
    check("republish identical revision set is idempotent",
          s == 200 and p1b["product_id"] == p1["product_id"])
    s, cur = http("GET", base_a, f"/groups/{gid}/products/current")
    check("current product query", s == 200 and cur["product_id"] == p1["product_id"])
    s, listing = http("GET", base_b, f"/groups/{gid}/products")
    check("one product in history listing", s == 200 and len(listing["products"]) == 1)
    s, d = http("GET", base_a, f"/groups/{gid2}/products/{p1['product_id']}")
    check("product from other group -> 404", s == 404)

    print("\n[6] all-missing cell yields null probability and zero denominator")
    g3 = register(base_a, members=2, rows=1, cols=2, thresholds=[5.0])
    gid3 = g3["group_id"]
    http("PUT", base_a, f"/groups/{gid3}/members/m1",
         {"rain": [[5.0, 9.0]], "missing": [[0, 1]], "revision": 1})
    http("PUT", base_b, f"/groups/{gid3}/members/m2",
         {"rain": [[10.0, 9.0]], "missing": [[0, 1]], "revision": 1})
    s, p3 = http("POST", base_a, f"/groups/{gid3}/publish")
    t3 = p3["thresholds"][0]
    check("all-missing cell null", t3["probability"] == [[1.0, None]], str(t3))
    check("all-missing cell denominator 0", t3["denominator"] == [[2, 0]])

    print("\n[7] correction expires old product immediately, history retained")
    s, _ = http("PUT", base_b, f"/groups/{gid}/members/m1",
                {"rain": [[0.0, 10.0], [20.0, 5.0]], "revision": 2})
    check("correction rev2 accepted", s == 200)
    s, d = http("GET", base_a, f"/groups/{gid}/products/current")
    check("no stale product served as current warning basis",
          s == 404 and d["error"]["code"] == "NO_CURRENT_PRODUCT", str(d))
    s, listing = http("GET", base_b, f"/groups/{gid}/products")
    check("old product retained but superseded",
          len(listing["products"]) == 1 and listing["products"][0]["is_current"] is False)
    s, hist = http("GET", base_a, f"/groups/{gid}/products/{p1['product_id']}")
    check("historical product still queryable", s == 200 and hist["is_current"] is False)
    s, p2 = http("POST", base_b, f"/groups/{gid}/publish")
    check("republish after correction", s == 200 and p2["product_id"] != p1["product_id"])
    check("new product uses rev2 for corrected member",
          {m["member_id"]: m["revision"] for m in p2["members_used"]}
          == {"m1": 2, "m2": 1, "m3": 1})
    s, cur = http("GET", base_b, f"/groups/{gid}/products/current")
    check("current points at new product", cur["product_id"] == p2["product_id"])

    print("\n[8] dual-instance interleaving: publish vs concurrent correction")
    g4 = register(base_a, members=3, rows=64, cols=64, thresholds=[5.0, 20.0])
    gid4 = g4["group_id"]
    for i, inst in enumerate((base_a, base_b, base_a)):
        name = f"m{i + 1}"
        grid = [[float((i + 1) * (r + c) % 25) for c in range(64)] for r in range(64)]
        s, _ = http("PUT", inst, f"/groups/{gid4}/members/{name}",
                    {"rain": grid, "revision": 1})
        assert s == 200

    result: Dict[str, Any] = {}

    def worker_publish() -> None:
        result["status"], result["body"] = http("POST", base_a, f"/groups/{gid4}/publish")

    t = threading.Thread(target=worker_publish)
    t.start()
    time.sleep(0.15)  # let the publish enter its slow compute phase
    t0 = time.time()
    s, corr = http("PUT", base_b, f"/groups/{gid4}/members/m2",
                   {"rain": [[7.0] * 64 for _ in range(64)], "revision": 2})
    write_elapsed = time.time() - t0
    check("correction write not blocked by in-flight compute", s == 200, str(corr))
    if hooks_enabled:
        check("ingest stayed fast while compute was running", write_elapsed < 0.5,
              f"{write_elapsed:.2f}s")
    t.join()
    check("in-flight publish eventually returns 200", result["status"] == 200, str(result))

    s, status4 = http("GET", base_a, f"/groups/{gid4}")
    latest = {m["member_id"]: m["latest_revision"] for m in status4["members"]}
    s, cur4 = http("GET", base_b, f"/groups/{gid4}/products/current")
    check("a current product exists after interleaving", s == 200)
    cur_map = {m["member_id"]: m["revision"] for m in cur4["members_used"]}
    check("current product matches latest member revisions (no mixed revision group)",
          cur_map == latest, f"{cur_map} vs {latest}")
    s, listing4 = http("GET", base_a, f"/groups/{gid4}/products")
    for prod in listing4["products"]:
        check(f"product {prod['product_id']} has complete revision set",
              len(prod["revision_map"]) == 3)
        if prod["is_current"]:
            check("only one current product",
                  sum(1 for x in listing4["products"] if x["is_current"]) == 1)
    check("current listing entry consistent",
          [x for x in listing4["products"] if x["is_current"]][0]["product_id"]
          == cur4["product_id"])

    print("\n[9] parallel publishes of the same revision set collapse to one product")
    g5 = register(base_b, members=2, rows=8, cols=8, thresholds=[5.0])
    gid5 = g5["group_id"]
    http("PUT", base_a, f"/groups/{gid5}/members/m1",
         {"rain": [[6.0] * 8 for _ in range(8)], "revision": 1})
    http("PUT", base_b, f"/groups/{gid5}/members/m2",
         {"rain": [[0.0] * 8 for _ in range(8)], "revision": 1})
    outs: Dict[str, Any] = {}

    def pub(instance: str) -> None:
        outs[instance] = http("POST", instance, f"/groups/{gid5}/publish")

    threads = [threading.Thread(target=pub, args=(base_a,)),
               threading.Thread(target=pub, args=(base_b,))]
    for th in threads:
        th.start()
    for th in threads:
        th.join()
    ids = {body["product_id"] for st, body in outs.values() if st == 200}
    check("both parallel publishes succeed", all(st == 200 for st, _ in outs.values()))
    check("both return the same idempotent product", len(ids) == 1, str(ids))
    s, listing5 = http("GET", base_a, f"/groups/{gid5}/products")
    check("exactly one product row for the revision set", len(listing5["products"]) == 1)

    print("\n[10] compute failure leaves no half-written product")
    g6 = register(base_a, members=2, rows=4, cols=4, thresholds=[5.0])
    gid6 = g6["group_id"]
    http("PUT", base_a, f"/groups/{gid6}/members/m1",
         {"rain": [[6.0] * 4 for _ in range(4)], "revision": 1})
    http("PUT", base_b, f"/groups/{gid6}/members/m2",
         {"rain": [[4.0] * 4 for _ in range(4)], "revision": 1})
    if hooks_enabled:
        with open(fail_marker, "w") as fh:
            fh.write("fail")
        s, d = http("POST", base_a, f"/groups/{gid6}/publish")
        os.remove(fail_marker)
        check("injected failure surfaces as 500", s == 500 and d["error"]["code"] == "COMPUTE_FAILED")
        s, d = http("GET", base_b, f"/groups/{gid6}/products/current")
        check("no current product after failed compute", s == 404)
        s, listing6 = http("GET", base_a, f"/groups/{gid6}/products")
        check("no product rows at all after failed compute", listing6["products"] == [])
        s, p6 = http("POST", base_b, f"/groups/{gid6}/publish")
        check("publish succeeds after failure cleared", s == 200 and p6["is_current"])
    else:
        print("  SKIP  failure injection (RAIN_TEST_FAIL_MARKER not configured)")

    print("\n[11] slow compute on one time does not block another time's ingestion")
    g7 = register(base_a, members=2, rows=16, cols=16, thresholds=[5.0])
    gid7 = g7["group_id"]
    http("PUT", base_a, f"/groups/{gid7}/members/m1",
         {"rain": [[6.0] * 16 for _ in range(16)], "revision": 1})
    http("PUT", base_a, f"/groups/{gid7}/members/m2",
         {"rain": [[6.0] * 16 for _ in range(16)], "revision": 1})

    bg = {}

    def slow_publish() -> None:
        bg["result"] = http("POST", base_a, f"/groups/{gid7}/publish")

    th = threading.Thread(target=slow_publish)
    th.start()
    time.sleep(0.15)
    t0 = time.time()
    g8 = register(base_b, members=2, rows=3, cols=3, thresholds=[5.0])
    http("PUT", base_b, f"/groups/{g8['group_id']}/members/m1",
         {"rain": [[1.0, 2.0, 3.0], [4.0, 5.0, 6.0], [7.0, 8.0, 9.0]], "revision": 1})
    s, _ = http("PUT", base_b, f"/groups/{g8['group_id']}/members/m2",
                {"rain": [[9.0, 8.0, 7.0], [6.0, 5.0, 4.0], [3.0, 2.0, 1.0]], "revision": 1})
    ingest_elapsed = time.time() - t0
    check("other time registration+ingest succeeded during slow compute", s == 200)
    if hooks_enabled:
        check("other time ingest stayed fast", ingest_elapsed < 0.5, f"{ingest_elapsed:.2f}s")
    th.join()
    check("slow publish completed", bg["result"][0] == 200, str(bg))

    print("\n[12] Swagger / OpenAPI")
    s, openapi = http("GET", base_a, "/openapi.json")
    check("openapi.json served", s == 200 and "paths" in openapi)
    for path in ("/groups", "/groups/{group_id}/members/{member_id}",
                 "/groups/{group_id}/publish",
                 "/groups/{group_id}/inspect",
                 "/groups/{group_id}/products/current"):
        check(f"path {path} documented", path in openapi["paths"])
    req = urllib.request.Request(base_a + "/docs")
    with urllib.request.urlopen(req, timeout=10) as resp:
        check("Swagger UI served", resp.status == 200 and b"swagger" in resp.read().lower())

    # ------------------------------------------------------------------ #
    # Storage inspection (post volume-migration patrol)
    # ------------------------------------------------------------------ #
    import sqlite3

    def db_corrupt_flip(gid: str, member_idx: int, revision: int) -> None:
        """Same-length byte flip inside rain_blob -> content_hash mismatch."""
        conn = sqlite3.connect(db_path)
        try:
            row = conn.execute(
                "SELECT rain_blob FROM member_revisions"
                " WHERE group_id=? AND member_idx=? AND revision=?",
                (gid, member_idx, revision),
            ).fetchone()
            assert row is not None, "revision row missing for corruption"
            blob = bytearray(row[0])
            blob[0] ^= 0x5A
            conn.execute(
                "UPDATE member_revisions SET rain_blob=?"
                " WHERE group_id=? AND member_idx=? AND revision=?",
                (bytes(blob), gid, member_idx, revision),
            )
            conn.commit()
        finally:
            conn.close()

    def db_corrupt_truncate(gid: str, member_idx: int, revision: int,
                            column: str = "miss_blob") -> None:
        """Truncate a grid blob -> byte-length check failure."""
        assert column in ("rain_blob", "miss_blob")
        conn = sqlite3.connect(db_path)
        try:
            row = conn.execute(
                f"SELECT {column} FROM member_revisions"
                " WHERE group_id=? AND member_idx=? AND revision=?",
                (gid, member_idx, revision),
            ).fetchone()
            assert row is not None and len(row[0]) > 1
            conn.execute(
                f"UPDATE member_revisions SET {column}=?"
                " WHERE group_id=? AND member_idx=? AND revision=?",
                (row[0][:-1], gid, member_idx, revision),
            )
            conn.commit()
        finally:
            conn.close()

    print("\n[13] inspection on healthy warehouses reports everything intact")
    g9 = register(base_a, members=2, rows=4, cols=4, thresholds=[5.0])
    gid9 = g9["group_id"]
    http("PUT", base_a, f"/groups/{gid9}/members/m1",
         {"rain": [[6.0] * 4 for _ in range(4)], "revision": 1})
    http("PUT", base_b, f"/groups/{gid9}/members/m2",
         {"rain": [[0.0] * 4 for _ in range(4)], "revision": 1})
    s, insp = http("POST", base_a, f"/groups/{gid9}/inspect")
    check("inspect 200 on healthy group", s == 200, str(insp))
    check("healthy summary",
          insp["summary"]["members_damaged"] == 0
          and insp["summary"]["members_healthy"] == 2
          and insp["summary"]["all_latest_healthy"] is True
          and insp["summary"]["products_revoked"] == 0, str(insp["summary"]))
    check("every check ok",
          all(c["status"] == "ok" for m in insp["members"] for c in m["checks"]))
    check("length + hash checks present per member",
          all({c["check"] for c in m["checks"]}
              == {"rain_length", "miss_length", "content_hash"}
              for m in insp["members"]))
    s, _ = http("POST", base_a, "/groups/does-not-exist/inspect")
    check("inspect unknown group -> 404", s == 404)

    print("\n[14] damage detection revokes current qualification and blocks publishing")
    g10 = register(base_b, members=3, rows=3, cols=3, thresholds=[5.0, 10.0])
    gid10 = g10["group_id"]
    for i, name in enumerate(("m1", "m2", "m3")):
        http("PUT", base_a, f"/groups/{gid10}/members/{name}",
             {"rain": [[float(i + 1)] * 3 for _ in range(3)], "revision": 1})
    s, p10 = http("POST", base_a, f"/groups/{gid10}/publish")
    check("g10 baseline publish", s == 200 and p10["is_current"] is True)
    pid10 = p10["product_id"]

    if db_path:
        db_corrupt_flip(gid10, member_idx=0, revision=1)       # m1: hash mismatch
        db_corrupt_truncate(gid10, member_idx=2, revision=1)  # m3: length mismatch
        s, insp = http("POST", base_b, f"/groups/{gid10}/inspect")
        check("inspect 200 after corruption", s == 200)
        sm = insp["summary"]
        check("two damaged members reported",
              sm["members_damaged"] == 2 and sm["members_healthy"] == 1
              and sm["all_latest_healthy"] is False, str(sm))
        by_member = {m["member_id"]: m for m in insp["members"]}
        check("m1 damaged via content_hash",
              by_member["m1"]["status"] == "damaged"
              and any("content_hash mismatch" in r for r in by_member["m1"]["reasons"])
              and by_member["m1"]["latest_verified"] is True)
        check("m3 damaged via length",
              by_member["m3"]["status"] == "damaged"
              and any("length" in r for r in by_member["m3"]["reasons"]))
        check("m2 stays healthy", by_member["m2"]["status"] == "healthy")
        check("damaged_members payload lists m1 and m3 with revisions",
              {d["member_id"]: d["revision"] for d in insp["damaged_members"]}
              == {"m1": 1, "m3": 1})
        check("one current product revoked in the same inspection",
              len(insp["revoked_products"]) == 1
              and insp["revoked_products"][0]["product_id"] == pid10
              and {h["member_idx"] for h in insp["revoked_products"][0]["damaged_references"]}
              == {0, 2}, str(insp["revoked_products"]))
        s, d = http("GET", base_a, f"/groups/{gid10}/products/current")
        check("damaged revision no longer serves a current warning product",
              s == 404 and d["error"]["code"] == "NO_CURRENT_PRODUCT")
        s, listing = http("GET", base_b, f"/groups/{gid10}/products")
        check("revoked product retained as history",
              [(p["product_id"], p["is_current"]) for p in listing["products"]]
              == [(pid10, False)])
        s, hist = http("GET", base_a, f"/groups/{gid10}/products/{pid10}")
        check("historical product still fetchable", s == 200 and hist["is_current"] is False)
        s, d = http("POST", base_a, f"/groups/{gid10}/publish")
        check("publish refuses damaged revisions",
              s == 409 and d["error"]["code"] == "REVISION_DAMAGED", str(d))
        # re-inspection is deterministic and does not create extra product rows
        s, insp2 = http("POST", base_b, f"/groups/{gid10}/inspect")
        check("re-inspect still reports the same two damaged members",
          insp2["summary"]["members_damaged"] == 2
          and insp2["summary"]["products_revoked"] == 0)
        s, listing = http("GET", base_a, f"/groups/{gid10}/products")
        check("no new product rows created by inspection",
              len(listing["products"]) == 1)
    else:
        print("  SKIP  direct-db corruption tests (RAIN_DB_PATH unavailable)")

    print("\n[15] healthy higher revisions allow re-publish after damage")
    if db_path:
        # repair by submitting strictly higher, healthy revisions
        for name in ("m1", "m2", "m3"):
            s, _ = http("PUT", base_b, f"/groups/{gid10}/members/{name}",
                        {"rain": [[8.0] * 3 for _ in range(3)], "revision": 2})
            check(f"{name} rev2 accepted after damage", s == 200)
        s, insp = http("POST", base_a, f"/groups/{gid10}/inspect")
        check("post-repair inspection all healthy",
              s == 200 and insp["summary"]["members_damaged"] == 0
              and insp["summary"]["all_latest_healthy"] is True
              and insp["summary"]["products_revoked"] == 0, str(insp["summary"]))
        check("inspection verifies revision 2",
              all(m["inspected_revision"] == 2 and m["latest_revision"] == 2
                  for m in insp["members"]))
        s, p11 = http("POST", base_b, f"/groups/{gid10}/publish")
        check("re-publish after repair succeeds",
              s == 200 and p11["is_current"] is True and p11["product_id"] != pid10)
        check("new product uses rev2 only",
              {m["member_id"]: m["revision"] for m in p11["members_used"]}
              == {"m1": 2, "m2": 2, "m3": 2})
        s, cur = http("GET", base_a, f"/groups/{gid10}/products/current")
        check("current product restored", s == 200 and cur["product_id"] == p11["product_id"])
        s, listing = http("GET", base_b, f"/groups/{gid10}/products")
        check("full history retained (revoked old + new current)",
              sorted((p["product_id"], p["is_current"]) for p in listing["products"])
              == sorted([(pid10, False), (p11["product_id"], True)]))
    else:
        print("  SKIP  repair flow (RAIN_DB_PATH unavailable)")

    print("\n[16] interleaved inspection vs ingestion/republish across two instances")
    # 16a: new healthy revision landing during hashing is never reported healthy
    g11 = register(base_a, members=2, rows=8, cols=8, thresholds=[5.0])
    gid11 = g11["group_id"]
    http("PUT", base_a, f"/groups/{gid11}/members/m1",
         {"rain": [[1.0] * 8 for _ in range(8)], "revision": 1})
    http("PUT", base_b, f"/groups/{gid11}/members/m2",
         {"rain": [[2.0] * 8 for _ in range(8)], "revision": 1})
    http("POST", base_a, f"/groups/{gid11}/publish")
    result16: Dict[str, Any] = {}

    def slow_inspect() -> None:
        result16["inspect"] = http("POST", base_a, f"/groups/{gid11}/inspect")

    th = threading.Thread(target=slow_inspect)
    th.start()
    time.sleep(0.2)  # enter the slow hashing window (RAIN_TEST_INSPECT_DELAY_S)
    t0 = time.time()
    s, _ = http("PUT", base_b, f"/groups/{gid11}/members/m1",
                {"rain": [[9.0] * 8 for _ in range(8)], "revision": 2})
    ingest_elapsed = time.time() - t0
    check("ingestion not blocked by in-flight inspection hashing", s == 200)
    if hooks_enabled:
        check("ingestion stayed fast during inspection", ingest_elapsed < 0.5,
              f"{ingest_elapsed:.2f}s")
    th.join()
    s, insp = result16["inspect"]
    check("interleaved inspection still 200", s == 200, str(insp))
    by_member = {m["member_id"]: m for m in insp["members"]}
    check("new rev2 reported unverified, never falsely healthy",
          by_member["m1"]["status"] == "unverified"
          and by_member["m1"]["latest_verified"] is False
          and by_member["m1"]["inspected_revision"] == 1
          and by_member["m1"]["latest_revision"] == 2, str(by_member["m1"]))
    check("superseded rev1 verdict carried for traceability",
          by_member["m1"]["superseded_inspection"]["revision"] == 1
          and by_member["m1"]["superseded_inspection"]["healthy"] is True)
    check("unaffected member still verified healthy",
          by_member["m2"]["status"] == "healthy")
    check("no product revoked by a stale verdict against a healthy new revision",
          insp["summary"]["products_revoked"] == 0)
    s, cur = http("GET", base_b, f"/groups/{gid11}/products/current")
    check("existing current product untouched by the interleaved inspection",
          s == 404)  # rev2 submission already expired the rev1 product
    # a follow-up inspection verifies the new revision properly
    s, insp = http("POST", base_b, f"/groups/{gid11}/inspect")
    check("follow-up inspection verifies rev2 as healthy",
          insp["summary"]["all_latest_healthy"] is True
          and {m["member_id"]: m["status"] for m in insp["members"]}
          == {"m1": "healthy", "m2": "healthy"})
    s, p12 = http("POST", base_a, f"/groups/{gid11}/publish")
    check("republish healthy new revision set", s == 200 and p12["is_current"] is True)

    # 16b: damaged product revoked while another time ingests concurrently
    if db_path:
        g12 = register(base_b, members=2, rows=4, cols=4, thresholds=[5.0])
        gid12 = g12["group_id"]
        http("PUT", base_a, f"/groups/{gid12}/members/m1",
             {"rain": [[6.0] * 4 for _ in range(4)], "revision": 1})
        http("PUT", base_a, f"/groups/{gid12}/members/m2",
             {"rain": [[6.0] * 4 for _ in range(4)], "revision": 1})
        s, p13 = http("POST", base_a, f"/groups/{gid12}/publish")
        check("g12 baseline publish", s == 200 and p13["is_current"] is True)
        db_corrupt_flip(gid12, member_idx=1, revision=1)

        other = register(base_a, members=2, rows=2, cols=2, thresholds=[5.0])
        goid = other["group_id"]
        result16b: Dict[str, Any] = {}

        def inspect_damaged() -> None:
            result16b["inspect"] = http("POST", base_b, f"/groups/{gid12}/inspect")

        th = threading.Thread(target=inspect_damaged)
        th.start()
        time.sleep(0.2)
        t0 = time.time()
        s1, _ = http("PUT", base_a, f"/groups/{goid}/members/m1",
                     {"rain": [[1.0, 2.0], [3.0, 4.0]], "revision": 1})
        s2, _ = http("PUT", base_b, f"/groups/{goid}/members/m2",
                     {"rain": [[4.0, 3.0], [2.0, 1.0]], "revision": 1})
        other_elapsed = time.time() - t0
        check("other time ingested during damage inspection", s1 == 200 and s2 == 200)
        if hooks_enabled:
            check("other-time ingest stayed fast", other_elapsed < 0.5,
                  f"{other_elapsed:.2f}s")
        th.join()
        s, insp = result16b["inspect"]
        check("damaged member confirmed and product revoked",
              s == 200 and insp["summary"]["members_damaged"] == 1
              and insp["summary"]["products_revoked"] == 1
              and insp["revoked_products"][0]["product_id"] == p13["product_id"])
        s, d = http("GET", base_a, f"/groups/{gid12}/products/current")
        check("revoked product not current", s == 404)
        s, po = http("POST", base_b, f"/groups/{goid}/publish")
        check("other time publishes fine after concurrent inspection",
              s == 200 and po["is_current"] is True)

        # 16c: repair + republish landing during hashing must keep the healthy
        # new product current; the stale damage verdict is never applied to it
        g13 = register(base_a, members=2, rows=5, cols=5, thresholds=[5.0])
        gid13 = g13["group_id"]
        http("PUT", base_a, f"/groups/{gid13}/members/m1",
             {"rain": [[6.0] * 5 for _ in range(5)], "revision": 1})
        http("PUT", base_b, f"/groups/{gid13}/members/m2",
             {"rain": [[0.0] * 5 for _ in range(5)], "revision": 1})
        s, p14 = http("POST", base_a, f"/groups/{gid13}/publish")
        check("g13 baseline publish", s == 200 and p14["is_current"] is True)
        db_corrupt_flip(gid13, member_idx=0, revision=1)
        result16c: Dict[str, Any] = {}

        def inspect_then_repair() -> None:
            result16c["inspect"] = http("POST", base_b, f"/groups/{gid13}/inspect")

        th = threading.Thread(target=inspect_then_repair)
        th.start()
        time.sleep(0.2)  # inspection is hashing the damaged rev1 snapshot
        s, _ = http("PUT", base_a, f"/groups/{gid13}/members/m1",
                    {"rain": [[7.0] * 5 for _ in range(5)], "revision": 2})
        check("repair revision lands during inspection", s == 200)
        s, p15 = http("POST", base_b, f"/groups/{gid13}/publish")
        check("healthy new product published during inspection",
              s == 200 and p15["is_current"] is True and p15["product_id"] != p14["product_id"])
        th.join()
        s, insp = result16c["inspect"]
        check("interleaved inspection 200", s == 200, str(insp))
        check("stale damage verdict did not revoke the healthy new product",
              insp["summary"]["products_revoked"] == 0
              and {m["member_id"]: m["status"] for m in insp["members"]}["m1"]
              == "unverified", str(insp["summary"]))
        s, cur = http("GET", base_a, f"/groups/{gid13}/products/current")
        check("healthy new product remains the current warning basis",
              s == 200 and cur["product_id"] == p15["product_id"] and cur["is_current"] is True)
        s, listing = http("GET", base_b, f"/groups/{gid13}/products")
        check("old product history retained, new one current",
              sorted((p["product_id"], p["is_current"]) for p in listing["products"])
              == sorted([(p14["product_id"], False), (p15["product_id"], True)]))
    else:
        print("  SKIP  concurrent damage inspection (RAIN_DB_PATH unavailable)")


if __name__ == "__main__":
    sys.exit(main())
