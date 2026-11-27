#!/usr/bin/env python3
"""qbam_io.py -- minimal, self-contained readers for the Q-BAM frozen engine contract files.

Single source of truth for the byte layouts: qbam/docs/engine_contract.md v1.2 (§1-§5).
This module is deliberately independent of qbam/python/workloads/common.py (the engine-side
source of truth) so that the external-system adapters in qbam/external/ can be run from any
checkout and so that a layout drift shows up as a hard header assertion here.

    pool.bin        "QBAMPOOL" u32 version, u32 dim, u32 nvec, nvec*dim f32 row-major
    init.bin        "QBAMINIT" u32 version, u32 dim, u32 n, u32 R,
                    n * (u64 id, u32 shard, u32 vec_ref)
    centroids.f32   R*dim f32 row-major (no header)
    trace.bin       "QBAMTRAC" u32 version, u32 dim, u32 n_events, u32 n_windows, u32 K,
                    then 26-byte packed events:
                    u32 window, u8 type, u64 id, u64 version, u32 vec_ref, u8 flags
    gt.bin          "QBAMGT\0\0" u32 version, u32 k, u32 n_queries, u32 dim,
                    n_queries*k * (u64 id, f32 squared_l2)

All integers little-endian.  Event types: 0=insert 1=update 2=erase 3=query.
"""
from __future__ import annotations

import struct
from dataclasses import dataclass
from pathlib import Path

import numpy as np

POOL_MAGIC = b"QBAMPOOL"
INIT_MAGIC = b"QBAMINIT"
TRACE_MAGIC = b"QBAMTRAC"
GT_MAGIC = b"QBAMGT\0\0"

EV_INSERT, EV_UPDATE, EV_ERASE, EV_QUERY = 0, 1, 2, 3
TYPE_NAME = {EV_INSERT: "insert", EV_UPDATE: "update", EV_ERASE: "erase", EV_QUERY: "query"}

INIT_REC = np.dtype([("id", "<u8"), ("shard", "<u4"), ("vec_ref", "<u4")])
EVENT_REC = np.dtype(
    [("window", "<u4"), ("type", "u1"), ("id", "<u8"), ("version", "<u8"), ("vec_ref", "<u4"), ("flags", "u1")]
)
GT_REC = np.dtype([("id", "<u8"), ("dist", "<f4")])
assert INIT_REC.itemsize == 16
assert EVENT_REC.itemsize == 26
assert GT_REC.itemsize == 12


@dataclass
class Header:
    path: str
    magic: bytes
    version: int
    fields: dict
    header_size: int


def _read_header(path, magic, fmt):
    with open(path, "rb") as f:
        got = f.read(8)
        assert got == magic, f"{path}: bad magic {got!r} != {magic!r}"
        vals = struct.unpack(fmt, f.read(struct.calcsize(fmt)))
        size = f.tell()
    return vals, size


def read_pool(path) -> tuple[np.ndarray, Header]:
    vals, size = _read_header(path, POOL_MAGIC, "<III")
    version, dim, nvec = vals
    nbytes = Path(path).stat().st_size
    expect = size + nvec * dim * 4
    assert nbytes == expect, f"{path}: size {nbytes} != {expect} (nvec={nvec} dim={dim})"
    vec = np.fromfile(path, dtype="<f4", offset=size).reshape(nvec, dim)
    return vec, Header(str(path), POOL_MAGIC, version, {"dim": dim, "nvec": nvec}, size)


def read_init(path) -> tuple[np.ndarray, Header]:
    vals, size = _read_header(path, INIT_MAGIC, "<IIII")
    version, dim, n, R = vals
    nbytes = Path(path).stat().st_size
    expect = size + n * 16
    assert nbytes == expect, f"{path}: size {nbytes} != {expect} (n={n})"
    rec = np.fromfile(path, dtype=INIT_REC, offset=size, count=n)
    return rec, Header(str(path), INIT_MAGIC, version, {"dim": dim, "n": n, "R": R}, size)


def read_centroids(path, R: int, dim: int) -> np.ndarray:
    arr = np.fromfile(path, dtype="<f4")
    assert arr.size == R * dim, f"{path}: {arr.size} != R*dim = {R * dim}"
    return arr.reshape(R, dim)


def read_trace(path) -> tuple[np.ndarray, Header]:
    vals, size = _read_header(path, TRACE_MAGIC, "<IIIII")
    version, dim, n_events, n_windows, K = vals
    nbytes = Path(path).stat().st_size
    expect = size + n_events * 26
    assert nbytes == expect, f"{path}: size {nbytes} != {expect} (n_events={n_events})"
    ev = np.fromfile(path, dtype=EVENT_REC, offset=size, count=n_events)
    return ev, Header(str(path), TRACE_MAGIC, version,
                      {"dim": dim, "n_events": n_events, "n_windows": n_windows, "K": K}, size)


def read_gt(path) -> tuple[np.ndarray, np.ndarray, Header]:
    vals, size = _read_header(path, GT_MAGIC, "<IIII")
    version, k, nq, dim = vals
    nbytes = Path(path).stat().st_size
    expect = size + nq * k * 12
    assert nbytes == expect, f"{path}: size {nbytes} != {expect} (nq={nq} k={k})"
    rec = np.fromfile(path, dtype=GT_REC, offset=size, count=nq * k).reshape(nq, k)
    return rec["id"], rec["dist"], Header(str(path), GT_MAGIC, version,
                                          {"k": k, "n_queries": nq, "dim": dim}, size)


def window_events(ev: np.ndarray, n_windows: int):
    """Yield (window, crud_events, query_events) in contract order; asserts CRUD-before-query."""
    order = np.argsort(ev["window"], kind="stable")
    ev = ev[order]
    bounds = np.searchsorted(ev["window"], np.arange(n_windows + 1))
    for w in range(n_windows):
        e = ev[bounds[w]:bounds[w + 1]]
        types = e["type"]
        n_crud = int(np.sum(types != EV_QUERY))
        if n_crud:
            assert np.all(types[:n_crud] != EV_QUERY), f"window {w}: query before CRUD (contract §4.1)"
            assert np.all(types[n_crud:] == EV_QUERY), f"window {w}: CRUD after query (contract §4.1)"
        yield w, e[:n_crud], e[n_crud:]


def write_diskann_bin(path, rows: np.ndarray) -> None:
    """DiskANN .bin: int32 npts, int32 dim, then npts*dim little-endian f32 row-major."""
    rows = np.ascontiguousarray(rows, dtype="<f4")
    n, d = rows.shape
    with open(path, "wb") as f:
        f.write(struct.pack("<ii", int(n), int(d)))
        f.write(rows.tobytes())


def write_u32(path, arr) -> None:
    np.ascontiguousarray(arr, dtype="<u4").tofile(path)


def mean_recall_at_k(returned: np.ndarray, gt_ids: np.ndarray, k: int) -> float:
    """Mean recall@k: |returned_topk ∩ gt_topk| / k, averaged over queries (ids are int64)."""
    assert returned.shape[0] == gt_ids.shape[0], (returned.shape, gt_ids.shape)
    kk = min(k, gt_ids.shape[1])
    got = np.asarray(returned[:, :kk], dtype=np.int64)
    want = np.asarray(gt_ids[:, :kk], dtype=np.int64)
    hits = 0
    for i in range(got.shape[0]):
        hits += len(set(got[i].tolist()) & set(want[i].tolist()))
    return hits / (kk * got.shape[0])
