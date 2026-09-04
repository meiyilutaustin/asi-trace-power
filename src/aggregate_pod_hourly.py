#!/usr/bin/env python
"""Aggregate the 351 GB pod_hourly table into a small hourly composition table.

Group key: (day, hour, cluster_id, gpu_spec_public, job_type_public,
priority_class, state_public).  For each group we store *sufficient statistics*
for any piecewise-linear util->power curve with knots at u in {0,.25,.5,.75}:

    S_b = sum_pods w * max(avg_gpu_sm_util/100 - b, 0)      (w = used_gpu_hours)

so downstream Monte Carlo over curve shapes never has to touch the big table
again.  Also aggregates server_hourly to per-(day,hour,cluster,gpu_spec)
capacity for the idle-floor term.

Restartable: one intermediate parquet per day in $AGG_DIR/days/, skipped if
present.  Final outputs: $AGG_DIR/pod_hourly_agg.parquet,
$AGG_DIR/server_hourly_agg.parquet.
"""
import glob
import os
import re
import sys
from multiprocessing import Pool

import numpy as np
import pandas as pd
import pyarrow.parquet as pq

DATA = os.environ.get("DATA_DIR", os.path.join(os.environ.get("ASI_ROOT", "."), "data"))
AGG = os.environ.get("AGG_DIR", os.path.join(os.environ.get("ASI_ROOT", "."), "agg"))
NPROC = int(os.environ.get("NPROC", "16"))
POD_DIR = os.path.join(DATA, "asi_opensource_pod_hourly")
KNOTS = [0.0, 0.25, 0.5, 0.75]
GROUP = ["day", "hour", "cluster_id", "gpu_spec_public",
         "job_type_public", "priority_class", "state_public"]
POD_COLS = ["cluster_id", "gpu_spec_public", "job_type_public",
            "priority_class", "state_public", "gpu_request",
            "used_gpu_hours", "avg_gpu_sm_util", "cpu_request_cores",
            "avg_cpu_request_util"]


def log(msg):
    print(msg, flush=True)


UTIL_DIV = float(os.environ.get("UTIL_DIV", "0"))  # 0 = auto-detect in main


def detect_util_scale():
    """Look at a few mid-trace files once; percent scale -> 100, else 1."""
    files = sorted(glob.glob(os.path.join(POD_DIR, "day=*", "hour=12",
                                          "*.parquet")))
    mx = 0.0
    for f in files[len(files) // 2:len(files) // 2 + 3]:
        u = pq.read_table(
            f, columns=["avg_gpu_sm_util"])[0].to_pandas().to_numpy()
        if len(u):
            mx = max(mx, float(np.nanmax(u)))
    return 100.0 if mx > 1.5 else 1.0


def agg_one_hour(path, day, hour):
    df = pq.read_table(path, columns=POD_COLS).to_pandas()
    df["day"], df["hour"] = day, hour
    for c in ["gpu_request", "used_gpu_hours", "cpu_request_cores"]:
        df[c] = df[c].fillna(0.0)
    u = (df["avg_gpu_sm_util"].astype(float) / UTIL_DIV).clip(0, 1)
    w = df["used_gpu_hours"].astype(float).clip(lower=0)
    null_u = u.isna()
    df["_w"] = w
    df["_w_null_u"] = np.where(null_u, w, 0.0)
    uf = u.fillna(0.0)
    for b in KNOTS:
        df[f"_S{int(b*100)}"] = np.where(null_u, 0.0, w * np.maximum(uf - b, 0.0))
    # host power sufficient stats
    cw = df["cpu_request_cores"].astype(float).clip(lower=0)
    cu = df["avg_cpu_request_util"].astype(float).clip(0, 1)
    df["_cpu_req_cores"] = cw
    df["_cpu_used_cores"] = (cw * cu.fillna(0.0))
    df["_n"] = 1
    out = (df.groupby(GROUP, observed=True, dropna=False)
             .agg(n_pods=("_n", "sum"),
                  gpu_request=("gpu_request", "sum"),
                  gpu_hours=("_w", "sum"),
                  gpu_hours_null_util=("_w_null_u", "sum"),
                  S0=("_S0", "sum"), S25=("_S25", "sum"),
                  S50=("_S50", "sum"), S75=("_S75", "sum"),
                  cpu_req_cores=("_cpu_req_cores", "sum"),
                  cpu_used_cores=("_cpu_used_cores", "sum"))
             .reset_index())
    return out


def agg_one_day(day_dir):
    day = re.search(r"day=(\d+)", day_dir).group(1)
    out_path = os.path.join(AGG, "days", f"pod_day={int(day):04d}.parquet")
    if os.path.exists(out_path):
        return f"day {day}: cached"
    parts = []
    for hour_dir in sorted(glob.glob(os.path.join(day_dir, "hour=*"))):
        hour = re.search(r"hour=(\d+)", hour_dir).group(1)
        for f in sorted(glob.glob(os.path.join(hour_dir, "*.parquet"))):
            parts.append(agg_one_hour(f, int(day), int(hour)))
    if not parts:
        return f"day {day}: EMPTY"
    res = (pd.concat(parts, ignore_index=True)
             .groupby(GROUP, observed=True, dropna=False).sum().reset_index())
    tmp = out_path + ".tmp"
    res.to_parquet(tmp, index=False)
    os.replace(tmp, out_path)
    return f"day {day}: {len(res)} rows"


def aggregate_pod():
    os.makedirs(os.path.join(AGG, "days"), exist_ok=True)
    day_dirs = sorted(glob.glob(os.path.join(POD_DIR, "day=*")))
    log(f"pod_hourly: {len(day_dirs)} day partitions, {NPROC} workers")
    with Pool(NPROC) as pool:
        for msg in pool.imap_unordered(agg_one_day, day_dirs):
            log(msg)
    files = sorted(glob.glob(os.path.join(AGG, "days", "pod_day=*.parquet")))
    full = pd.concat([pd.read_parquet(f) for f in files], ignore_index=True)
    full.to_parquet(os.path.join(AGG, "pod_hourly_agg.parquet"), index=False)
    log(f"pod_hourly_agg: {len(full)} rows -> {AGG}/pod_hourly_agg.parquet")


def aggregate_server():
    import pyarrow.dataset as ds
    d = ds.dataset(os.path.join(DATA, "asi_opensource_server_hourly"),
                   partitioning="hive")
    df = d.to_table(columns=["server_id", "cluster_id", "gpu_spec_public",
                             "gpu_count", "cpu_capacity_cores", "day",
                             "hour"]).to_pandas()
    out = (df.groupby(["day", "hour", "cluster_id", "gpu_spec_public"],
                      observed=True, dropna=False)
             .agg(n_servers=("server_id", "count"),
                  gpu_count=("gpu_count", "sum"),
                  cpu_capacity_cores=("cpu_capacity_cores", "sum"))
             .reset_index())
    out["day"] = out["day"].astype(int)
    out["hour"] = out["hour"].astype(int)
    out.to_parquet(os.path.join(AGG, "server_hourly_agg.parquet"), index=False)
    log(f"server_hourly_agg: {len(out)} rows")


if __name__ == "__main__":
    os.makedirs(AGG, exist_ok=True)
    if UTIL_DIV == 0:
        UTIL_DIV = detect_util_scale()
        os.environ["UTIL_DIV"] = str(UTIL_DIV)  # inherited by Pool workers
    log(f"UTIL_DIV = {UTIL_DIV}")
    aggregate_server()
    aggregate_pod()
    log("DONE aggregate_pod_hourly")
