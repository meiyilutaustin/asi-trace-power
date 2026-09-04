"""Drop into training/solver scripts; call write_provenance() at start.

Records the hardware/software a result came from, so every number can be
traced to the machine that produced it (COMPUTE_RESOURCES.md §7 Rule 3).
"""
import json, os, platform, subprocess


def _run(cmd):
    try:
        return subprocess.run(cmd, capture_output=True, text=True).stdout.strip()
    except Exception:
        return None


def provenance():
    p = {
        "cluster": os.environ.get("SLURM_CLUSTER_NAME", "unknown"),
        "node": platform.node(),
        "slurm_job": os.environ.get("SLURM_JOB_ID"),
        "allocation": os.environ.get("SLURM_JOB_ACCOUNT"),
        "nnodes": os.environ.get("SLURM_NNODES"),
        "ntasks": os.environ.get("SLURM_NTASKS"),
        "python": platform.python_version(),
        "git_commit": _run(["git", "rev-parse", "HEAD"]),
        "driver": _run(["nvidia-smi", "--query-gpu=driver_version",
                        "--format=csv,noheader"]),
    }
    try:
        import torch
        p.update({
            "torch": torch.__version__,
            "cuda_build": torch.version.cuda,
            "gpus": [torch.cuda.get_device_name(i)
                     for i in range(torch.cuda.device_count())],
            "capability": [torch.cuda.get_device_capability(i)
                           for i in range(torch.cuda.device_count())],
            "tf32_matmul": torch.backends.cuda.matmul.allow_tf32,
        })
    except ImportError:
        pass
    return p


def write_provenance(path="provenance.json"):
    with open(path, "w") as f:
        json.dump(provenance(), f, indent=2)


if __name__ == "__main__":
    write_provenance()
