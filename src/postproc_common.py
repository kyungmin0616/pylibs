#!/usr/bin/env python3
"""
Common helpers for SCHISM post-processing scripts.
"""

from __future__ import annotations

import copy
import csv
import os
import sys
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Sequence, Tuple, Union

import numpy as np


PathLike = Union[str, Path]


def deep_update_dict(
    base: Dict[str, Any],
    override: Dict[str, Any],
    merge_list_of_dicts: bool = False,
) -> Dict[str, Any]:
    """
    Recursively deep-merge dictionaries.

    When merge_list_of_dicts=True, list entries that are dicts are merged by index.
    If override list is longer than base list, the first base dict is used as template.
    """
    out = copy.deepcopy(base)
    for key, val in override.items():
        if isinstance(val, dict) and isinstance(out.get(key), dict):
            out[key] = deep_update_dict(out[key], val, merge_list_of_dicts=merge_list_of_dicts)
            continue

        if (
            merge_list_of_dicts
            and isinstance(val, list)
            and isinstance(out.get(key), list)
            and all(isinstance(v, dict) for v in val)
            and all(isinstance(v, dict) for v in out.get(key, []))
            and len(out.get(key, [])) > 0
        ):
            merged: List[Dict[str, Any]] = []
            base_list = out.get(key, [])
            for i, item in enumerate(val):
                template = base_list[i] if i < len(base_list) else base_list[0]
                merged.append(
                    deep_update_dict(template, item, merge_list_of_dicts=merge_list_of_dicts)
                )
            out[key] = merged
            continue

        out[key] = copy.deepcopy(val)
    return out


def _env_int(names: Sequence[str], default: int = 0) -> int:
    for name in names:
        val = os.environ.get(name)
        if val is None:
            continue
        try:
            return int(val)
        except Exception:
            continue
    return default


def _looks_like_mpi_launch() -> bool:
    size = _env_int(
        [
            "OMPI_COMM_WORLD_SIZE",
            "PMI_SIZE",
            "PMIX_SIZE",
            "MPI_LOCALNRANKS",
            "SLURM_NTASKS",
        ],
        default=0,
    )
    return size > 1


def init_mpi_runtime(
    argv: Optional[List[str]] = None,
    mpi_flag: str = "--mpi",
    no_mpi_flag: str = "--no-mpi",
    enable_env: str = "ENABLE_MPI",
    consume_flags: bool = True,
) -> Tuple[Any, Any, int, int, bool]:
    """
    Detect and initialize MPI runtime.

    Returns:
      (MPI, COMM, RANK, SIZE, USE_MPI)
    """
    if argv is None:
        argv = sys.argv

    use_mpi = (
        mpi_flag in argv
        or os.environ.get(enable_env, "0") == "1"
        or _looks_like_mpi_launch()
    )
    if no_mpi_flag in argv:
        use_mpi = False

    if consume_flags:
        while mpi_flag in argv:
            argv.remove(mpi_flag)
        while no_mpi_flag in argv:
            argv.remove(no_mpi_flag)

    mpi_mod = None
    comm = None
    rank = 0
    size = 1
    if use_mpi:
        try:
            from mpi4py import MPI as _MPI  # type: ignore

            mpi_mod = _MPI
            comm = _MPI.COMM_WORLD
            rank = comm.Get_rank()
            size = comm.Get_size()
        except (ImportError, Exception) as exc:
            print(
                f"[WARN] MPI requested but initialization failed: {exc}. "
                "Falling back to serial mode.",
                flush=True,
            )
            mpi_mod = None
            comm = None
            rank = 0
            size = 1
            use_mpi = False

    return mpi_mod, comm, rank, size, bool(use_mpi)


def rank_log(
    msg: str,
    rank: int = 0,
    size: int = 1,
    rank0_only: bool = False,
    flush: bool = True,
) -> None:
    if rank0_only and int(rank) != 0:
        return
    prefix = f"[rank {rank}/{size}] " if int(size) > 1 else ""
    print(prefix + str(msg), flush=flush)


def report_work_assignment(
    tag: str,
    total_count: int,
    local_indices: Sequence[int],
    rank: int,
    size: int,
    comm: Any = None,
    mpi_enabled: bool = False,
    logger: Optional[Callable[[str], None]] = None,
) -> None:
    nloc = len(local_indices)
    if nloc > 0:
        first_idx = int(local_indices[0])
        last_idx = int(local_indices[-1])
    else:
        first_idx = -1
        last_idx = -1

    if logger is None:
        logger = lambda text: rank_log(text, rank=rank, size=size)

    logger(
        f"{tag} assignment: local={nloc}/{int(total_count)}, "
        f"index_range=[{first_idx},{last_idx}], stride={int(size)}"
    )

    if mpi_enabled and comm is not None:
        counts = comm.gather(nloc, root=0)
        if int(rank) == 0:
            summary = ", ".join([f"r{i}:{c}" for i, c in enumerate(counts)])
            logger(f"{tag} distribution by rank -> {summary}")
        comm.Barrier()


def _empty_metrics(n: int = 0) -> Dict[str, float]:
    return {
        "n": int(n),
        "bias": np.nan,
        "rmse": np.nan,
        "corr": np.nan,
        "obs_std": np.nan,
        "mod_std": np.nan,
        "nrmse_std": np.nan,
        "wss": np.nan,
        "crmsd": np.nan,
    }


def compute_skill_metrics(obs: np.ndarray, mod: np.ndarray, min_n: int = 2) -> Dict[str, float]:
    obs_arr = np.asarray(obs, dtype=float)
    mod_arr = np.asarray(mod, dtype=float)
    valid = np.isfinite(obs_arr) & np.isfinite(mod_arr)
    n_valid = int(valid.sum())
    if n_valid == 0 or n_valid < int(min_n):
        return _empty_metrics(n_valid)

    obs_use = obs_arr[valid]
    mod_use = mod_arr[valid]
    n = int(len(obs_use))
    if n == 0:
        return _empty_metrics(0)

    diff = mod_use - obs_use
    bias = float(np.mean(diff))
    rmse = float(np.sqrt(np.mean(diff**2)))
    obs_std = float(np.std(obs_use))
    mod_std = float(np.std(mod_use))
    corr = float(np.corrcoef(obs_use, mod_use)[0, 1]) if n > 1 else np.nan
    nrmse = float(rmse / obs_std) if obs_std > 0 else np.nan
    obs_mean = float(np.mean(obs_use))
    denom = np.sum((np.abs(mod_use - obs_mean) + np.abs(obs_use - obs_mean)) ** 2)
    wss = float(1.0 - np.sum((mod_use - obs_use) ** 2) / denom) if denom > 0 else np.nan
    obs_anom = obs_use - obs_mean
    mod_anom = mod_use - float(np.mean(mod_use))
    crmsd = float(np.sqrt(np.mean((mod_anom - obs_anom) ** 2)))
    return {
        "n": n,
        "bias": bias,
        "rmse": rmse,
        "corr": corr,
        "obs_std": obs_std,
        "mod_std": mod_std,
        "nrmse_std": nrmse,
        "wss": wss,
        "crmsd": crmsd,
    }


def write_csv_rows(path: PathLike, rows: List[Dict[str, Any]], fieldnames: List[str]) -> None:
    out_path = Path(path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


def read_csv_rows(path: PathLike) -> List[Dict[str, Any]]:
    in_path = Path(path)
    if not in_path.exists():
        return []
    with open(in_path, "r", newline="", encoding="utf-8") as f:
        return list(csv.DictReader(f))


def write_rank_csv_chunk(
    outdir: PathLike,
    chunk_name: str,
    rank: int,
    rows: List[Dict[str, Any]],
    fieldnames: List[str],
    prefix: str = "chunk",
) -> Tuple[Path, Path]:
    chunk_dir = Path(outdir) / str(chunk_name)
    chunk_dir.mkdir(parents=True, exist_ok=True)
    chunk_path = chunk_dir / f"{prefix}_rank_{int(rank):04d}.csv"
    write_csv_rows(chunk_path, rows, fieldnames)
    return chunk_dir, chunk_path


def collect_rank_csv_chunks(
    chunk_dir: PathLike,
    nrank: int,
    prefix: str = "chunk",
) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    cdir = Path(chunk_dir)
    for rr in range(int(nrank)):
        cpath = cdir / f"{prefix}_rank_{int(rr):04d}.csv"
        rows.extend(read_csv_rows(cpath))
    return rows


def cleanup_rank_csv_chunks(chunk_dir: PathLike, prefix: str = "chunk") -> None:
    cdir = Path(chunk_dir)
    if not cdir.is_dir():
        return
    for fn in cdir.iterdir():
        if fn.name.startswith(f"{prefix}_rank_") and fn.suffix.lower() == ".csv":
            try:
                fn.unlink()
            except Exception:
                pass
    try:
        cdir.rmdir()
    except Exception:
        pass
