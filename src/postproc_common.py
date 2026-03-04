#!/usr/bin/env python3
"""
Common helpers for SCHISM post-processing scripts.
"""

from __future__ import annotations

import copy
import csv
import os
import re
import sys
from glob import glob
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


def to_scalar(value: Any, default: Any = None) -> Any:
    if value is None:
        return default
    if isinstance(value, (list, tuple, np.ndarray)):
        if len(value) == 0:
            return default
        return value[0]
    return value


def normalize_stack_list(stacks: Any, discovered_stacks: Any) -> np.ndarray:
    if stacks is None:
        return np.asarray(
            sorted({int(v) for v in np.asarray(discovered_stacks).ravel()}),
            dtype=int,
        )
    if isinstance(stacks, (list, tuple)) and len(stacks) == 2:
        s0 = int(stacks[0])
        s1 = int(stacks[1])
        if s1 < s0:
            raise ValueError(f"Invalid stack range [{s0}, {s1}]")
        return np.arange(s0, s1 + 1, dtype=int)
    return np.asarray(
        sorted({int(v) for v in np.asarray(stacks).ravel()}),
        dtype=int,
    )


def primary_stack_file(outputs_dir: PathLike, stack: int, outfmt: int) -> Optional[str]:
    outputs = str(outputs_dir)
    st = int(stack)
    if int(outfmt) == 0:
        fn = os.path.join(outputs, f"out2d_{st}.nc")
        return fn if os.path.exists(fn) else None

    cand = sorted(glob(os.path.join(outputs, f"schout_*_{st}.nc")))
    if len(cand) > 0:
        return cand[0]
    fn = os.path.join(outputs, f"schout_{st}.nc")
    return fn if os.path.exists(fn) else None


def stack_files_for_check(
    outputs_dir: PathLike,
    stack: int,
    outfmt: int,
    check_all_files: bool = False,
) -> List[str]:
    primary = primary_stack_file(outputs_dir, stack, outfmt)
    if primary is None:
        return []
    if not bool(check_all_files):
        return [primary]
    files = sorted(glob(os.path.join(str(outputs_dir), f"*_{int(stack)}.nc")))
    if len(files) == 0:
        return [primary]
    if primary not in files:
        files.insert(0, primary)
    return files


def header_time_ok(
    nc_path: PathLike,
    readnc: Optional[Callable[[str], Any]] = None,
) -> Tuple[bool, str]:
    handle = None
    try:
        if readnc is None:
            from netCDF4 import Dataset  # type: ignore

            handle = Dataset(str(nc_path), mode="r")
        else:
            handle = readnc(str(nc_path))
        variables = getattr(handle, "variables", {})
        if "time" not in variables:
            return False, "missing time variable"
        tvar = variables["time"]
        if hasattr(tvar, "shape") and len(tvar.shape) > 0:
            nt = int(tvar.shape[0])
        else:
            nt = int(len(np.asarray(tvar)))
        if nt <= 0:
            return False, "empty time variable"
        _ = float(np.asarray(tvar[0]).ravel()[0])
        return True, "ok"
    except Exception as exc:
        return False, str(exc)
    finally:
        if handle is not None:
            try:
                handle.close()
            except Exception:
                pass


def size_ok(
    path: PathLike,
    ref_size: float,
    ratio_min: float = 0.70,
    abs_min_bytes: Optional[int] = None,
) -> Tuple[bool, str]:
    try:
        size = int(os.path.getsize(str(path)))
    except Exception as exc:
        return False, f"size check failed: {exc}"
    if abs_min_bytes is not None and size < int(abs_min_bytes):
        return False, f"size={size} < abs_min={int(abs_min_bytes)}"
    thr = float(ratio_min) * float(ref_size)
    if size < thr:
        return False, f"size={size} < ratio_min*median={int(thr)}"
    return True, "ok"


def screen_stacks(
    outputs_dir: PathLike,
    stacks: Any,
    outfmt: int,
    mode: Optional[str] = "light",
    check_all_files: bool = False,
    ratio_min: float = 0.70,
    abs_min_bytes: Optional[int] = None,
    readnc: Optional[Callable[[str], Any]] = None,
    logger: Optional[Callable[[str], None]] = None,
    log_limit: int = 20,
) -> Tuple[np.ndarray, Dict[int, str]]:
    stack_vals = [int(v) for v in np.asarray(stacks).ravel()]
    if len(stack_vals) == 0:
        return np.asarray([], dtype=int), {}

    mode_norm = "none" if mode is None else str(mode).lower()
    if mode_norm == "none":
        return np.asarray(stack_vals, dtype=int), {}

    primary: Dict[int, str] = {}
    for st in stack_vals:
        pth = primary_stack_file(outputs_dir, st, outfmt)
        if pth is not None:
            primary[st] = pth

    ref_size: Optional[int] = None
    sizes = [os.path.getsize(pth) for pth in primary.values() if os.path.exists(pth)]
    if len(sizes) > 0:
        ref_size = int(np.median(np.asarray(sizes, dtype=float)))

    valid: List[int] = []
    skipped: Dict[int, str] = {}
    for st in stack_vals:
        files = stack_files_for_check(outputs_dir, st, outfmt, check_all_files=check_all_files)
        if len(files) == 0:
            skipped[st] = "missing primary stack file"
            continue

        need_light = mode_norm in {"light", "light+size"}
        need_size = mode_norm in {"size", "light+size"}
        ok = True
        reason = ""

        if need_light:
            for fn in files:
                c_ok, c_reason = header_time_ok(fn, readnc=readnc)
                if not c_ok:
                    ok = False
                    reason = f"{os.path.basename(fn)}: {c_reason}"
                    break

        if ok and need_size and ref_size is not None:
            for fn in files:
                s_ok, s_reason = size_ok(
                    fn,
                    ref_size=ref_size,
                    ratio_min=ratio_min,
                    abs_min_bytes=abs_min_bytes,
                )
                if not s_ok:
                    ok = False
                    reason = f"{os.path.basename(fn)}: {s_reason}"
                    break

        if ok:
            valid.append(st)
        else:
            skipped[st] = reason

    if logger is not None:
        logger(
            f"Stack screen ({mode_norm}): requested={len(stack_vals)}, "
            f"valid={len(valid)}, skipped={len(skipped)}"
        )
        if len(skipped) > 0 and int(log_limit) > 0:
            for i, st in enumerate(sorted(skipped.keys())):
                if i >= int(log_limit):
                    logger(f"  ... {len(skipped) - int(log_limit)} more skipped stacks")
                    break
                logger(f"  skip stack {st}: {skipped[st]}")

    return np.asarray(valid, dtype=int), skipped


def get_model_start_datenum(
    run_dir: PathLike,
    apply_utc_start: bool = False,
    read_schism_param_func: Optional[Callable[[str, int], Dict[str, Any]]] = None,
    datenum_func: Optional[Callable[[int, int, int], float]] = None,
    param_name: str = "param.nml",
) -> Tuple[Optional[float], str]:
    pfile = os.path.join(str(run_dir), param_name)
    if not os.path.exists(pfile):
        return None, f"{param_name} not found in {run_dir}"

    if read_schism_param_func is None or datenum_func is None:
        try:
            from pylib import read_schism_param as _read_schism_param, datenum as _datenum  # type: ignore

            if read_schism_param_func is None:
                read_schism_param_func = _read_schism_param
            if datenum_func is None:
                datenum_func = _datenum
        except Exception as exc:
            return None, f"missing read_schism_param/datenum functions: {exc}"

    try:
        params = read_schism_param_func(pfile, 1)
    except Exception as exc:
        return None, f"failed to parse {param_name}: {exc}"

    req = ("start_year", "start_month", "start_day", "start_hour")
    for key in req:
        if key not in params:
            return None, f"missing {key} in {param_name}"

    try:
        sy = int(to_scalar(params.get("start_year")))
        sm = int(to_scalar(params.get("start_month")))
        sd = int(to_scalar(params.get("start_day")))
        sh = float(to_scalar(params.get("start_hour"), 0.0))
        us = float(to_scalar(params.get("utc_start"), 0.0))
    except Exception as exc:
        return None, f"invalid start fields in {param_name}: {exc}"

    d0 = float(datenum_func(sy, sm, sd))
    d0 += sh / 24.0
    if bool(apply_utc_start):
        d0 -= us / 24.0
    return d0, f"{sy:04d}-{sm:02d}-{sd:02d} {sh:05.2f}h (utc_start={us})"


def read_stack_times_abs(
    nc_path: PathLike,
    start_datenum: Optional[float] = None,
    readnc: Optional[Callable[[str], Any]] = None,
    time_to_days: Optional[Callable[[Any], np.ndarray]] = None,
) -> np.ndarray:
    handle = None
    try:
        if readnc is None:
            from netCDF4 import Dataset  # type: ignore

            handle = Dataset(str(nc_path), mode="r")
        else:
            handle = readnc(str(nc_path))

        tvar = getattr(handle, "variables", {}).get("time")
        if tvar is None:
            return np.asarray([], dtype=float)
        if time_to_days is not None:
            tvals = np.asarray(time_to_days(tvar), dtype=float).ravel()
        else:
            tvals = np.asarray(tvar[:], dtype=float).ravel()
    except Exception:
        return np.asarray([], dtype=float)
    finally:
        if handle is not None:
            try:
                handle.close()
            except Exception:
                pass

    if start_datenum is not None:
        tvals = tvals + float(start_datenum)
    return tvals


def _first_value(item: Dict[str, Any], keys: Sequence[str]) -> Any:
    for key in keys:
        if key in item and item[key] is not None:
            return item[key]
    return None


def normalize_run_specs(
    cfg: Dict[str, Any],
    runs_key: str = "RUNS",
    run_keys: Sequence[str] = ("RUN", "run", "run_dir"),
    name_keys: Sequence[str] = ("NAME", "name", "RUN_NAME", "run_name"),
    output_keys: Sequence[str] = ("SNAME", "sname", "out_npz"),
    output_template_key: str = "SNAME_TEMPLATE",
    default_output_template: str = "./npz/{run_name}",
    include_keys: Optional[Sequence[str]] = None,
) -> List[Dict[str, Any]]:
    include = list(include_keys) if include_keys is not None else []
    runs = cfg.get(runs_key)
    specs: List[Dict[str, Any]] = []

    if runs is None:
        run = _first_value(cfg, run_keys)
        if run is None:
            raise ValueError(f"Either {runs_key} or one of {list(run_keys)} must be configured.")

        name = _first_value(cfg, name_keys)
        if name is None:
            name = os.path.basename(os.path.abspath(str(run)))
        output = _first_value(cfg, output_keys)
        if output is None:
            tmpl = str(cfg.get(output_template_key, default_output_template))
            output = tmpl.format(run_name=name, run=run)

        spec: Dict[str, Any] = {"NAME": str(name), "RUN": str(run), "SNAME": str(output)}
        for key in include:
            spec[key] = cfg.get(key)
        specs.append(spec)
        return specs

    for i, item in enumerate(runs):
        if isinstance(item, str):
            item = {run_keys[0]: item}
        if not isinstance(item, dict):
            raise ValueError(f"{runs_key}[{i}] must be dict or string")

        run = _first_value(item, run_keys)
        if run is None:
            raise ValueError(f"{runs_key}[{i}] missing run key; expected one of {list(run_keys)}")

        name = _first_value(item, name_keys)
        if name is None:
            name = os.path.basename(os.path.abspath(str(run)))

        output = _first_value(item, output_keys)
        if output is None:
            tmpl = str(cfg.get(output_template_key, default_output_template))
            output = tmpl.format(run_name=name, run=run)

        spec = {"NAME": str(name), "RUN": str(run), "SNAME": str(output)}
        for key in include:
            spec[key] = item.get(key, cfg.get(key))
        specs.append(spec)

    return specs


def stack_num_from_name(path: PathLike) -> Optional[int]:
    name = os.path.basename(str(path))
    match = re.search(r"_(\d+)\.nc$", name)
    return int(match.group(1)) if match else None
