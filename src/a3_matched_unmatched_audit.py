#!/usr/bin/env python
"""Audit A3 execution-summary coverage for days 110--184.

Match status is established exactly from per-pod daily span aggregates and
execution-summary pod IDs. Unmatched spans do not carry priority/workload
labels, so the script reports exact all-population composition and sharp
worst-case bounds for matched composition instead of inventing labels.

Only existing aggregates are read; the raw 351 GB pod-hour table is not read.
"""

from __future__ import annotations

import gc
import hashlib
import json
import os
import sys
import time
import zlib
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd
import pyarrow.parquet as pq

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import provenance as provenance_lib  # noqa: E402


DATA = Path(os.environ.get("DATA_DIR", os.path.join(os.environ.get("ASI_ROOT", "."), "data")))
AGG = Path(os.environ.get("AGG_DIR", os.path.join(os.environ.get("ASI_ROOT", "."), "agg")))
OUT = Path(
    os.environ.get(
        "OUT_DIR", os.path.join(os.environ.get("ASI_ROOT", "."), "a3_matched_unmatched_audit")
    )
)
DAY_START = int(os.environ.get("DAY_START", "110"))
DAY_END = int(os.environ.get("DAY_END", "184"))
NBUCKET = 16
MISSING = "__MISSING__"
RECONCILE_TOL_GH = 1e-4


def log(message: str) -> None:
    print(message, flush=True)


def write_json(path: Path, value: dict) -> None:
    with path.open("w") as handle:
        json.dump(value, handle, indent=2)
        handle.write("\n")


def normalize_label(series: pd.Series) -> pd.Series:
    return series.astype("string").fillna(MISSING)


def records(frame: pd.DataFrame) -> list[dict]:
    return json.loads(frame.to_json(orient="records", double_precision=15))


def composition_bounds(
    all_gpu_hours: pd.Series,
    unmatched_gpu_hours: float,
    label_names: list[str],
) -> pd.DataFrame:
    """Return exact all shares and sharp per-cell matched-share mass bounds."""
    all_total = float(all_gpu_hours.sum())
    matched_total = all_total - unmatched_gpu_hours
    lower_gpu_hours = all_gpu_hours - np.minimum(all_gpu_hours, unmatched_gpu_hours)
    frame = pd.DataFrame(
        {
            "all_gpu_hours": all_gpu_hours,
            "all_composition_share_pct": all_gpu_hours / all_total * 100.0,
            "matched_composition_share_lower_pct": (
                lower_gpu_hours / matched_total * 100.0
            ),
            "matched_composition_share_upper_pct": (
                all_gpu_hours / matched_total * 100.0
            ),
        }
    )
    frame["matched_bound_width_pp"] = (
        frame["matched_composition_share_upper_pct"]
        - frame["matched_composition_share_lower_pct"]
    )
    frame = frame.reset_index()
    frame.columns = label_names + list(frame.columns[len(label_names) :])
    return frame.sort_values(label_names, kind="stable").reset_index(drop=True)


def main() -> None:
    started = time.monotonic()
    if DAY_START > DAY_END:
        raise ValueError("DAY_START must be <= DAY_END")
    OUT.mkdir(parents=True, exist_ok=True)

    job_path = DATA / "asi_opensource_job_execution_summary"
    aggregate_path = AGG / "pod_hourly_agg.parquet"
    span_paths_by_bucket: dict[int, list[Path]] = {}
    for bucket in range(NBUCKET):
        paths = [
            AGG / "span_days" / f"bucket={bucket:02d}" / f"day={day:04d}.parquet"
            for day in range(DAY_START, DAY_END + 1)
        ]
        missing = [str(path) for path in paths if not path.exists()]
        if missing:
            raise FileNotFoundError(
                f"missing {len(missing)} span-day files; first={missing[0]}"
            )
        span_paths_by_bucket[bucket] = paths

    all_span_paths = [path for paths in span_paths_by_bucket.values() for path in paths]
    span_input_bytes = int(sum(path.stat().st_size for path in all_span_paths))
    span_input_rows = int(sum(pq.read_metadata(path).num_rows for path in all_span_paths))

    provenance = provenance_lib.provenance()
    source_commit = os.environ.get("SOURCE_GIT_COMMIT")
    if source_commit:
        provenance["git_commit"] = source_commit
    provenance.update(
        {
            "generated_at_utc": datetime.now(timezone.utc).isoformat(),
            "script": str(Path(__file__).resolve()),
            "script_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
            "parameters": {
                "day_start_inclusive": DAY_START,
                "day_end_inclusive": DAY_END,
            },
            "inputs": {
                "span_days": {
                    "path_pattern": str(
                        AGG / "span_days" / "bucket=*" / "day=*.parquet"
                    ),
                    "selected_files": len(all_span_paths),
                    "selected_bytes": span_input_bytes,
                    "selected_rows": span_input_rows,
                    "columns": ["pod_id", "gh"],
                },
                "job_execution_summary": {
                    "path": str(job_path),
                    "columns": ["pod_id"],
                },
                "pod_hourly_agg": {
                    "path": str(aggregate_path),
                    "bytes": aggregate_path.stat().st_size,
                    "rows": pq.read_metadata(aggregate_path).num_rows,
                    "columns": [
                        "day",
                        "priority_class",
                        "job_type_public",
                        "gpu_request",
                        "gpu_hours",
                    ],
                },
            },
        }
    )
    write_json(OUT / "provenance.json", provenance)

    all_rows = pd.read_parquet(
        aggregate_path,
        columns=[
            "day",
            "priority_class",
            "job_type_public",
            "gpu_request",
            "gpu_hours",
        ],
    )
    all_rows = all_rows[
        all_rows["day"].between(DAY_START, DAY_END, inclusive="both")
        & (all_rows["gpu_request"] > 0)
    ].copy()
    for label in ("priority_class", "job_type_public"):
        all_rows[label] = normalize_label(all_rows[label])
    all_priority = all_rows.groupby(
        "priority_class", observed=True, dropna=False
    )["gpu_hours"].sum()
    all_workload = all_rows.groupby(
        "job_type_public", observed=True, dropna=False
    )["gpu_hours"].sum()
    all_cross = all_rows.groupby(
        ["priority_class", "job_type_public"], observed=True, dropna=False
    )["gpu_hours"].sum()
    aggregate_total_gpu_hours = float(all_rows["gpu_hours"].sum())
    aggregate_rows_selected = int(len(all_rows))
    observed_days = sorted(int(day) for day in all_rows["day"].unique())
    del all_rows
    gc.collect()

    jobs = pd.read_parquet(job_path, columns=["pod_id"])
    job_rows = int(len(jobs))
    jobs = jobs.drop_duplicates("pod_id", keep="first")
    unique_job_pods = int(len(jobs))
    jobs["bucket"] = jobs["pod_id"].map(
        lambda value: zlib.crc32(value.encode()) % NBUCKET
    ).astype("int8")

    span_total_gpu_hours = 0.0
    matched_gpu_hours = 0.0
    unmatched_gpu_hours = 0.0
    matched_pod_day_rows = 0
    unmatched_pod_day_rows = 0
    matched_unique_pods = 0
    unmatched_unique_pods = 0
    negative_input_rows = 0
    null_input_rows = 0
    for bucket, paths in span_paths_by_bucket.items():
        spans = pd.concat(
            [pd.read_parquet(path, columns=["pod_id", "gh"]) for path in paths],
            ignore_index=True,
        )
        negative_input_rows += int((spans["gh"] < 0).sum())
        null_input_rows += int(spans["gh"].isna().sum())
        bucket_total = float(spans["gh"].sum())
        span_total_gpu_hours += bucket_total
        joined = spans.merge(
            jobs.loc[jobs["bucket"] == bucket, ["pod_id"]],
            on="pod_id",
            how="left",
            indicator=True,
            validate="many_to_one",
        )
        is_matched = joined["_merge"].eq("both")
        matched = joined.loc[is_matched]
        unmatched = joined.loc[~is_matched]
        matched_gpu_hours += float(matched["gh"].sum())
        unmatched_gpu_hours += float(unmatched["gh"].sum())
        matched_pod_day_rows += int(len(matched))
        unmatched_pod_day_rows += int(len(unmatched))
        matched_unique_pods += int(matched["pod_id"].nunique())
        unmatched_unique_pods += int(unmatched["pod_id"].nunique())
        log(
            f"bucket {bucket:02d}: {len(joined):,} rows, "
            f"{matched['gh'].sum() / bucket_total * 100:.6f}% matched GPU-h"
        )
        del spans, joined, matched, unmatched, is_matched
        gc.collect()

    priority_bounds = composition_bounds(
        all_priority, unmatched_gpu_hours, ["priority_class"]
    )
    workload_bounds = composition_bounds(
        all_workload, unmatched_gpu_hours, ["job_type_public"]
    )
    cross_bounds = composition_bounds(
        all_cross, unmatched_gpu_hours, ["priority_class", "job_type_public"]
    )
    status_closure = span_total_gpu_hours - matched_gpu_hours - unmatched_gpu_hours
    aggregate_closure = span_total_gpu_hours - aggregate_total_gpu_hours
    checks = {
        "span_total_minus_aggregate_total_gpu_hours": aggregate_closure,
        "span_total_minus_matched_minus_unmatched_gpu_hours": status_closure,
        "negative_span_gpu_hour_rows": negative_input_rows,
        "null_span_gpu_hour_rows": null_input_rows,
        "passes": {
            "span_and_aggregate_totals_reconcile": bool(
                abs(aggregate_closure) <= RECONCILE_TOL_GH
            ),
            "matched_plus_unmatched_closes": bool(
                abs(status_closure) <= RECONCILE_TOL_GH
            ),
            "no_negative_or_null_span_gpu_hours": bool(
                negative_input_rows == 0 and null_input_rows == 0
            ),
        },
    }
    matched_total = span_total_gpu_hours - unmatched_gpu_hours
    summary = {
        "analysis": "A3 post-jump execution-summary coverage and composition bound",
        "window": {
            "day_start_inclusive": DAY_START,
            "day_end_inclusive": DAY_END,
            "n_days": DAY_END - DAY_START + 1,
            "observed_days": observed_days,
        },
        "method": (
            "Match status is exact from span-day pod IDs versus execution-summary pod IDs. "
            "Unmatched span rows contain no priority/workload labels, so unmatched composition "
            "is not identified. Exact all-population label totals plus the unmatched mass give "
            "sharp worst-case bounds for every matched composition share."
        ),
        "input_counts": {
            "span_day_files": len(all_span_paths),
            "span_day_bytes": span_input_bytes,
            "span_pod_day_rows": span_input_rows,
            "aggregate_rows_selected": aggregate_rows_selected,
            "job_execution_rows": job_rows,
            "job_execution_unique_pods": unique_job_pods,
        },
        "overall": {
            "all_gpu_hours": span_total_gpu_hours,
            "matched_gpu_hours": matched_gpu_hours,
            "unmatched_gpu_hours": unmatched_gpu_hours,
            "matched_gpu_hour_share_pct": matched_gpu_hours
            / span_total_gpu_hours
            * 100.0,
            "unmatched_gpu_hour_share_pct": unmatched_gpu_hours
            / span_total_gpu_hours
            * 100.0,
            "matched_pod_day_rows": matched_pod_day_rows,
            "unmatched_pod_day_rows": unmatched_pod_day_rows,
            "matched_unique_pods": matched_unique_pods,
            "unmatched_unique_pods": unmatched_unique_pods,
        },
        "composition_identification": {
            "all_population_composition": "exact from pod_hourly_agg",
            "matched_composition": "bounded; unmatched labels unavailable",
            "unmatched_composition": "not identified from existing aggregates",
            "max_matched_share_bound_width_pp": unmatched_gpu_hours
            / matched_total
            * 100.0,
            "interpretation": (
                "Even adversarial assignment of all unmatched mass to one label can change any "
                "matched composition share by no more than the reported bound width."
            ),
        },
        "composition_bounds_by_priority": records(priority_bounds),
        "composition_bounds_by_workload": records(workload_bounds),
        "composition_bounds_by_priority_and_workload": records(cross_bounds),
        "quality_checks": checks,
        "interpretation_guardrail": (
            "This audit tests coverage bias in days 110--184. It does not make pre-day-109 "
            "coverage representative, nor convert observed schedule delay into demonstrated "
            "flexibility or tolerance."
        ),
        "elapsed_seconds": time.monotonic() - started,
    }
    write_json(OUT / "summary.json", summary)
    priority_bounds.to_csv(OUT / "composition_bounds_priority.csv", index=False)
    workload_bounds.to_csv(OUT / "composition_bounds_workload.csv", index=False)
    cross_bounds.to_csv(OUT / "composition_bounds_priority_by_workload.csv", index=False)
    provenance["elapsed_seconds"] = summary["elapsed_seconds"]
    provenance["outputs"] = [
        "summary.json",
        "composition_bounds_priority.csv",
        "composition_bounds_workload.csv",
        "composition_bounds_priority_by_workload.csv",
        "provenance.json",
    ]
    write_json(OUT / "provenance.json", provenance)

    failed = [name for name, passed in checks["passes"].items() if not passed]
    if failed:
        raise RuntimeError(f"quality checks failed: {failed}")
    log(json.dumps(summary["overall"], indent=2))
    log(f"DONE in {summary['elapsed_seconds']:.1f} s -> {OUT}")


if __name__ == "__main__":
    main()
