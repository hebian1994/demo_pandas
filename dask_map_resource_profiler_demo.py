"""
DEMO：client.map 算分区 → as_completed 谁先算完谁立刻追加进【一个】CSV
（不写 part 文件、不改 Dask 源码）

流程:
  1. read_csv → to_delayed → compute → Futures
  2. client.map(业务) → 每个 Future 持有一个 DataFrame
  3. as_completed：分区一完成就 fut.result()，追加写入最终 CSV
     → 到达序；无中间 part；最终文件仍是单 writer（本 DEMO 在 client 端写）

说明:
  - 比「先写 N 个 part 再合并」少一轮磁盘
  - 数据会经 client 拉回再写；分区很大时可用 Actor 在某一 worker 上写（见文件末尾注释）

Windows 下必须 if __name__ == "__main__"。
"""

from __future__ import annotations

import os
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
from dask.distributed import Client, Future, LocalCluster, as_completed

OUTPUT_DIR = Path(__file__).parent / "output"
INPUT_CSV = OUTPUT_DIR / "map_pipeline_input.csv"
OUTPUT_CSV = OUTPUT_DIR / "map_pipeline_result.csv"
PROFILE_HTML = OUTPUT_DIR / "dask-resource-profiler.html"

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
    """分区业务：只返回 DataFrame，不写盘。"""
    out = pdf.copy()
    out["amount"] = out["qty"] * out["price"]
    acc = 0.0
    for i in range(400_000):
        acc += (i % 97) * 0.001
    out["score"] = out["amount"] + (acc % 1.0)
    time.sleep(0.05)
    return out


def append_completed_to_csv(
    result_futures: list[Future],
    dest_path: str,
) -> tuple[str, int]:
    """
    谁先算完谁先追加：不落 part 文件。
    as_completed 本身就是「完成通知」；每次只拉回一个分区 DF。
    """
    Path(dest_path).parent.mkdir(parents=True, exist_ok=True)
    total_rows = 0
    first = True
    with open(dest_path, "w", newline="", encoding="utf-8") as f:
        for fut in as_completed(result_futures):
            pdf = fut.result()  # 该分区已完成；失败会在这里抛出
            pdf.to_csv(f, index=False, header=first)
            first = False
            total_rows += len(pdf)
            # pdf 出作用域后可被回收；不必等全部算完再写
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
    print(f"目的地 CSV: {OUTPUT_CSV}")
    print("写出方式: as_completed → 直接追加（无 part 文件，到达序）")

    dest_path = str(OUTPUT_CSV)
    t0 = time.perf_counter()
    with ResourceProfiler(dt=0.25) as rprof:
        part_futures: list[Future] = client.compute(ddf.to_delayed())
        print(f"读入 Futures: {len(part_futures)}")

        t1 = time.perf_counter()
        result_futures = client.map(process_partition, part_futures)

        final_path, total_rows = append_completed_to_csv(result_futures, dest_path)
        t2 = time.perf_counter()
        print(f"计算+追加完成: {t2 - t1:.2f}s → {final_path}, rows={total_rows}")
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

# ---------------------------------------------------------------------------
# 可选：分区很大、不想经 client 拉数据时，用 Actor 在集群内单点追加
#
# class CsvAppendActor:
#     def __init__(self, dest: str):
#         from pathlib import Path
#         Path(dest).parent.mkdir(parents=True, exist_ok=True)
#         self.dest = dest
#         self.first = True
#         self.total = 0
#         self._f = open(dest, "w", newline="", encoding="utf-8")
#
#     def append(self, pdf: pd.DataFrame) -> int:
#         pdf.to_csv(self._f, index=False, header=self.first)
#         self.first = False
#         self.total += len(pdf)
#         self._f.flush()
#         return self.total
#
#     def finish(self) -> tuple[str, int]:
#         self._f.close()
#         return self.dest, self.total
#
# def process_and_append(pdf, actor):
#     out = process_partition(pdf)
#     actor.append(out).result()  # Actor 方法调用是串行的，天然单 writer
#     return len(out)
#
# writer = client.submit(CsvAppendActor, dest_path, actor=True).result()
# client.map(process_and_append, part_futures, writer=writer)
# final_path, total_rows = writer.finish().result()
# ---------------------------------------------------------------------------
