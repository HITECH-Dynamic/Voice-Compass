"""
Run Kaggle submission inference over a range of test shards.

Features:
- uses one worker per listed GPU;
- processes shards sequentially on each GPU;
- skips completed shard CSVs;
- reuses partial CSVs when rerun;
- writes a separate log and failure report per shard.
"""

from __future__ import annotations

import argparse
import os
import subprocess
import sys
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()

    parser.add_argument("--test-root", required=True)
    parser.add_argument("--adapter-dir", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--log-dir", required=True)

    parser.add_argument("--start-shard", type=int, required=True)
    parser.add_argument("--end-shard", type=int, required=True)

    parser.add_argument(
        "--gpus",
        default="0,1",
        help="Comma-separated GPU indices, for example 0,1.",
    )
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--save-every", type=int, default=100)
    parser.add_argument("--num-beams", type=int, default=1)

    return parser.parse_args()


def run_shard(
    shard_index: int,
    gpu_index: str,
    args: argparse.Namespace,
    generator: Path,
    output_dir: Path,
    log_dir: Path,
) -> tuple[int, int]:
    final_csv = output_dir / f"shard_{shard_index:03d}.csv"
    partial_csv = output_dir / f"shard_{shard_index:03d}_partial.csv"
    failure_csv = output_dir / f"shard_{shard_index:03d}_failures.csv"
    log_path = log_dir / f"shard_{shard_index:03d}.log"

    if final_csv.exists():
        print(f"SKIP shard {shard_index:03d}: final CSV already exists.")
        return shard_index, 0

    command = [
        sys.executable,
        "-u",
        str(generator),
        "--test-root",
        args.test_root,
        "--adapter-dir",
        args.adapter_dir,
        "--shard-index",
        str(shard_index),
        "--output-csv",
        str(final_csv),
        "--progress-csv",
        str(partial_csv),
        "--failure-csv",
        str(failure_csv),
        "--batch-size",
        str(args.batch_size),
        "--save-every",
        str(args.save_every),
        "--num-beams",
        str(args.num_beams),
    ]

    env = os.environ.copy()
    env["CUDA_VISIBLE_DEVICES"] = gpu_index

    print(
        f"START shard {shard_index:03d} on GPU {gpu_index} "
        f"(log: {log_path.name})"
    )

    with log_path.open("a", encoding="utf-8") as log_file:
        result = subprocess.run(
            command,
            cwd=generator.parent.parent,
            env=env,
            stdout=log_file,
            stderr=subprocess.STDOUT,
            check=False,
        )

    if result.returncode == 0 and final_csv.exists():
        print(f"DONE  shard {shard_index:03d} on GPU {gpu_index}")
        return shard_index, 0

    print(
        f"FAIL  shard {shard_index:03d} on GPU {gpu_index}; "
        f"return code {result.returncode}. See {log_path}"
    )
    return shard_index, result.returncode or 1


def main() -> None:
    args = parse_args()

    if args.end_shard < args.start_shard:
        raise ValueError("--end-shard must be >= --start-shard")

    repo_root = Path(__file__).resolve().parent.parent
    generator = repo_root / "scripts/generate_kaggle_submission.py"

    if not generator.exists():
        raise FileNotFoundError(generator)

    output_dir = Path(args.output_dir)
    log_dir = Path(args.log_dir)

    output_dir.mkdir(parents=True, exist_ok=True)
    log_dir.mkdir(parents=True, exist_ok=True)

    gpu_indices = [x.strip() for x in args.gpus.split(",") if x.strip()]
    if not gpu_indices:
        raise ValueError("At least one GPU index is required.")

    shard_indices = list(range(args.start_shard, args.end_shard + 1))

    # Split shards into one sequential queue per GPU. This prevents two
    # simultaneous models from competing for the same GPU.
    queues = {
        gpu: shard_indices[position::len(gpu_indices)]
        for position, gpu in enumerate(gpu_indices)
    }

    failures: list[int] = []

    def run_queue(gpu: str, queue: list[int]) -> list[int]:
        queue_failures = []

        for shard_index in queue:
            _, return_code = run_shard(
                shard_index,
                gpu,
                args,
                generator,
                output_dir,
                log_dir,
            )

            if return_code != 0:
                queue_failures.append(shard_index)
                break

        return queue_failures

    with ThreadPoolExecutor(max_workers=len(gpu_indices)) as executor:
        futures = {
            executor.submit(run_queue, gpu, queue): gpu
            for gpu, queue in queues.items()
        }

        for future in as_completed(futures):
            failures.extend(future.result())

    completed = len(list(output_dir.glob("shard_[0-9][0-9][0-9].csv")))

    print()
    print(f"Completed shard CSVs in output directory: {completed}")

    if failures:
        print("Failed shards:", failures)
        raise SystemExit(1)

    print(
        f"Requested range {args.start_shard:03d}–"
        f"{args.end_shard:03d} completed successfully."
    )


if __name__ == "__main__":
    main()
