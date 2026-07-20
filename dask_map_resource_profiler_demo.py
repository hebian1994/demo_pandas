"""
DEMO：算完返回 → map 写 part CSV → Queue 通知 → 单 writer 边收边并入最终 CSV

流程:
  1. read_csv → to_delayed → compute → Futures
  2. client.map(业务) → 返回 DataFrame（分区内不写盘）
  3. 先 submit(合并消费者)，再 map(写 part + Queue.put)
     → 哪个 part 先写完就先并入最终 CSV（到达序，不保证分区号序）
  4. 不改 Dask 源码；最终 CSV 仍是单 writer 串行 append

Windows 下必须 if __name__ == "__main__"。
"""

from __future__ import annotations

import os
import shutil
import time
from pathlib import Path

for _k in (
    "OMP_NUM_THREADS",
    "MKL_NUM_THREADS",
    "OPENBLAS_NUM_THREADS",
    "NUMEXPR_NUM_THREADS",
    "VECLIB_MAXIMUM_THREADS",
):
    os.environ.setdefault(_k, "1")

import dask.dataframe as dd
import numpy as np
import pandas as pd
from dask.diagnostics import ResourceProfiler
from dask.distributed import Client, Future, LocalCluster, Queue, wait

OUTPUT_DIR = Path(__file__).parent / "output"
INPUT_CSV = OUTPUT_DIR / "map_pipeline_input.csv"
PART_DIR = OUTPUT_DIR / "map_pipeline_parts"
OUTPUT_CSV = OUTPUT_DIR / "map_pipeline_result.csv"
PROFILE_HTML = OUTPUT_DIR / "dask-resource-profiler.html"
MERGE_QUEUE_NAME = "map-pipeline-merge-q"

PROD_CORES = 32
PROD_RAM_GB = 256
PROD_N_WORKERS = 10
PROD_THREADS_PER_WORKER = 1
PROD_MEMORY_LIMIT = 0
PROD_NPARTITIONS = 128

N_WORKERS = 4
THREADS_PER_WORKER = 1
MEMORY_LIMIT = 0
TARGET_NPARTITIONS = 48
N_ROWS = 80_000


def make_sample_csv(n_rows: int = N_ROWS) -> Path:
    OUTPUT_DIR.mkdir(exist_ok=True)
    rng = np.random.default_rng(42)
    pd.DataFrame(
        {
            "order_id": np.arange(n_rows),
            "sku": rng.integers(1000, 1100, size=n_rows),
            "qty": rng.integers(1, 20, size=n_rows),
            "price": rng.random(n_rows) * 100.0,
        }
    ).to_csv(INPUT_CSV, index=False)
    return INPUT_CSV


def process_partition(pdf: pd.DataFrame) -> pd.DataFrame:
    """分区业务：只返回 DataFrame，不在这里写 CSV。"""
    out = pdf.copy()
    out["amount"] = out["qty"] * out["price"]
    acc = 0.0
    for i in range(400_000):
        acc += (i % 97) * 0.001
    out["score"] = out["amount"] + (acc % 1.0)
    time.sleep(0.05)
    return out


def custom_to_csv(pdf: pd.DataFrame, path: str, queue_name: str) -> tuple[str, int]:
    """
    写一个 part CSV，写完后把 (path, nrows) 放入 Queue 通知合并端。
    client.map 调用；queue_name 广播给每个任务。
    """
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    pdf.to_csv(path, index=False)
    nrows = len(pdf)
    Queue(queue_name).put((path, nrows))
    return path, nrows


def merge_parts_from_queue(
    queue_name: str,
    dest_path: str,
    n_parts: int,
) -> tuple[str, int]:
    """
    单消费者：从 Queue 取「已写完的 part」，按到达顺序并入最终 CSV。
    先到的第一个 part 带 header，后续跳过 header 后二进制追加。
    """
    q = Queue(queue_name)
    Path(dest_path).parent.mkdir(parents=True, exist_ok=True)
    total_rows = 0
    with open(dest_path, "wb") as out_f:
        for i in range(n_parts):
            part, nrows = q.get()
            total_rows += nrows
            with open(part, "rb") as in_f:
                if i == 0:
                    shutil.copyfileobj(in_f, out_f)
                else:
                    in_f.readline()
                    shutil.copyfileobj(in_f, out_f)
    return dest_path, total_rows


def print_cluster_vs_prod(client: Client, nparts: int) -> None:
    n_workers = len(client.scheduler_info()["workers"])
    slots = n_workers * THREADS_PER_WORKER
    print(f"Dashboard: {client.dashboard_link}")
    print(f"本机 DEMO: workers={n_workers}, threads/worker={THREADS_PER_WORKER}, "
          f"slots={slots}, memory_limit={MEMORY_LIMIT}, partitions={nparts}")
    print(f"生产对照: workers={PROD_N_WORKERS}, threads/worker={PROD_THREADS_PER_WORKER}, "
          f"slots={PROD_N_WORKERS * PROD_THREADS_PER_WORKER}/{PROD_CORES}核, "
          f"memory_limit={PROD_MEMORY_LIMIT}, partitions={PROD_NPARTITIONS}, "
          f"RAM={PROD_RAM_GB}GB")


def print_profiler_summary(rprof: ResourceProfiler, n_workers: int, threads: int) -> None:
    print(f"\n采样点数: {len(rprof.results)}")
    if not rprof.results:
        return
    peak_mem_mb = max(r.mem for r in rprof.results)
    peak_cpu = max(r.cpu for r in rprof.results)
    slots = max(1, n_workers * threads)
    print(f"峰值 mem: {peak_mem_mb:.1f} MB ({peak_mem_mb / 1024:.2f} GB)")
    print(f"峰值 cpu(加总): {peak_cpu:.1f}%  ≈ {peak_cpu / 100:.2f} 核当量")
    print(f"公式 max(cpu)/n_workers/threads = {peak_cpu / n_workers / threads:.2f}")
    print(f"更合理 peak_cpu/100/slots = {peak_cpu / 100 / slots:.2f}")


def demo() -> None:
    csv_path = make_sample_csv()
    PART_DIR.mkdir(parents=True, exist_ok=True)

    cluster = LocalCluster(
        n_workers=N_WORKERS,
        threads_per_worker=THREADS_PER_WORKER,
        memory_limit=MEMORY_LIMIT,
        dashboard_address=":8787",
    )
    client = Client(cluster)

    file_size = csv_path.stat().st_size
    blocksize = max(file_size // TARGET_NPARTITIONS, 64 * 1024)
    ddf = dd.read_csv(
        csv_path,
        blocksize=blocksize,
        engine="pyarrow",
        dtype_backend="pyarrow",
    )
    nparts = ddf.npartitions
    print_cluster_vs_prod(client, nparts)
    print(f"输入 CSV: {csv_path} ({file_size / 1e6:.2f} MB)")
    print(f"part 目录: {PART_DIR}")
    print(f"目的地 CSV: {OUTPUT_CSV}")
    print(f"合并队列: {MERGE_QUEUE_NAME} (到达序并入，不保证分区号序)")

    dest_path = str(OUTPUT_CSV)
    t0 = time.perf_counter()
    with ResourceProfiler(dt=0.25) as rprof:
        part_futures: list[Future] = client.compute(ddf.to_delayed())
        print(f"读入 Futures: {len(part_futures)}")

        t1 = time.perf_counter()
        result_futures = client.map(process_partition, part_futures)

        # 先挂上消费者，再写 part；写完即 put，与合并重叠
        merge_fut = client.submit(
            merge_parts_from_queue,
            MERGE_QUEUE_NAME,
            dest_path,
            nparts,
        )
        part_paths = [str(PART_DIR / f"part-{i:04d}.csv") for i in range(nparts)]
        write_futures = client.map(
            custom_to_csv,
            result_futures,
            part_paths,
            queue_name=MERGE_QUEUE_NAME,
        )

        try:
            wait(write_futures)
            for fut in write_futures:
                if fut.status == "error":
                    fut.result()
            final_path, total_rows = merge_fut.result()
        except Exception:
            merge_fut.cancel()
            raise

        t2 = time.perf_counter()
        print(f"写出+队列合并完成: {t2 - t1:.2f}s → {final_path}, rows={total_rows}")
        time.sleep(0.5)

    elapsed = time.perf_counter() - t0
    sample = pd.read_csv(final_path, nrows=3)
    print(f"总墙钟: {elapsed:.1f}s")
    print(f"结果预览:\n{sample}")

    n_workers = len(client.scheduler_info()["workers"])
    print_profiler_summary(rprof, n_workers, THREADS_PER_WORKER)

    try:
        rprof.visualize(filename=str(PROFILE_HTML), show=False)
        print(f"\n资源曲线: {PROFILE_HTML}")
    except Exception as exc:  # noqa: BLE001
        print(f"\nvisualize 跳过: {exc}")

    client.close()
    cluster.close()


if __name__ == "__main__":
    demo()
