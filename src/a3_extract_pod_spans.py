#!/usr/bin/env python
"""A3a: extract per-pod execution spans from the big pod_hourly table.

For every GPU pod (gpu_request > 0): first/last observed hour and total
gpu_hours. Needed because job_execution_summary has schedule_delay but no
timestamps — actual placement in time lives only in pod_hourly.

Two stages, both restartable:
  1. per-day: concat 24 hourly files -> groupby pod_id -> parquet with a
     hash bucket column (pods cross days, so no day-local merge is final)
  2. per-bucket global merge -> $AGG_DIR/pod_spans.parquet
"""
import glob
import os
import re
import zlib
from multiprocessing import Pool

import numpy as np
import pandas as pd
import pyarrow.compute as pc
import pyarrow.parquet as pq

DATA = os.environ.get("DATA_DIR", os.path.join(os.environ.get("ASI_ROOT", "."), "data"))
AGG = os.environ.get("AGG_DIR", os.path.join(os.environ.get("ASI_ROOT", "."), "agg"))
NPROC = int(os.environ.get("NPROC", "16"))
NBUCKET = 16
POD_DIR = os.path.join(DATA, "asi_opensource_pod_hourly")
SPAN_DAYS = os.path.join(AGG, "span_days")
COLS = ["pod_id", "gpu_request", "used_gpu_hours"]


def log(msg):
    print(msg, flush=True)


def day_spans(day_dir):
    day = int(re.search(r"day=(\d+)", day_dir).group(1))
    done_flag = os.path.join(SPAN_DAYS, f"day={day:04d}.done")
    if os.path.exists(done_flag):
        return f"day {day}: cached"
    parts = []
    for hour_dir in sorted(glob.glob(os.path.join(day_dir, "hour=*"))):
        hour = int(re.search(r"hour=(\d+)", hour_dir).group(1))
        for f in sorted(glob.glob(os.path.join(hour_dir, "*.parquet"))):
            t = pq.read_table(f, columns=COLS)
            # filter in arrow BEFORE pandas: keeps string materialization to
            # the ~7% of rows that are GPU pods (OOM fix, an earlier job)
            t = t.filter(pc.greater(pc.fill_null(t["gpu_request"], 0.0), 0.0))
            df = t.select(["pod_id", "used_gpu_hours"]).to_pandas()
            df["t"] = day * 24 + hour
            parts.append(df)
    if not parts:
        return f"day {day}: EMPTY"
    d = pd.concat(parts, ignore_index=True)
    g = (d.groupby("pod_id", observed=True)
         .agg(t_min=("t", "min"), t_max=("t", "max"),
              gh=("used_gpu_hours", "sum")).reset_index())
    g["bucket"] = g["pod_id"].map(
        lambda s: zlib.crc32(s.encode()) % NBUCKET).astype("int8")
    for b, sub in g.groupby("bucket"):
        bdir = os.path.join(SPAN_DAYS, f"bucket={b:02d}")
        os.makedirs(bdir, exist_ok=True)
        sub.drop(columns="bucket").to_parquet(
            os.path.join(bdir, f"day={day:04d}.parquet"), index=False)
    open(done_flag, "w").close()
    return f"day {day}: {len(g)} pods"


def merge_bucket(b):
    out = os.path.join(AGG, f"pod_spans_b{b:02d}.parquet")
    if os.path.exists(out):
        return f"bucket {b}: cached"
    parts = [pd.read_parquet(f)
             for f in sorted(glob.glob(os.path.join(
                 SPAN_DAYS, f"bucket={b:02d}", "day=*.parquet")))]
    d = pd.concat(parts, ignore_index=True)
    g = (d.groupby("pod_id", observed=True)
         .agg(t_min=("t_min", "min"), t_max=("t_max", "max"),
              gh=("gh", "sum")).reset_index())
    tmp = out + ".tmp"
    g.to_parquet(tmp, index=False)
    os.replace(tmp, out)
    return f"bucket {b}: {len(g)} pods"


if __name__ == "__main__":
    os.makedirs(SPAN_DAYS, exist_ok=True)
    day_dirs = sorted(glob.glob(os.path.join(POD_DIR, "day=*")))
    log(f"stage 1: {len(day_dirs)} days, {NPROC} workers")
    with Pool(NPROC) as pool:
        for msg in pool.imap_unordered(day_spans, day_dirs):
            log(msg)
    log("stage 2: merging buckets")
    with Pool(min(NPROC, 4)) as pool:   # bucket merge is memory-heavy
        for msg in pool.imap_unordered(merge_bucket, range(NBUCKET)):
            log(msg)
    files = [f for f in sorted(glob.glob(
        os.path.join(AGG, "pod_spans_b*.parquet")))
        if not f.endswith(".tmp")]
    n = sum(pq.read_metadata(f).num_rows for f in files)
    log(f"pod_spans: {n} pods across {len(files)} bucket files")
    log("DONE a3_extract_pod_spans")
