"""
DEMO 03：内存压力 —— Bytes Stored 色带与 spill-to-disk。

对开 URL:
  /status                         — Bytes Stored / Bytes per Worker
  /individual-workers-memory      — 单图：每 worker 内存

期望现象:
  - 内存条从绿 → 黄（接近 spill）→ 橙/红（pause）变化
  - 压力大时 Task Stream 可能出现橙色磁盘 I/O 条
  - Worker 内存接近 memory_limit

调优启示:
  1. 分区/chunk 太大 → 单任务峰值内存高，易 spill
  2. 该 persist 的中间结果再算；不该全塞进内存的要分批或减小并发
  3. LocalCluster 的 memory_limit 要按机器真实内存设，别盲目加大 n_workers
"""

from __future__ import annotations

import sys
import time
from pathlib import Path

import dask.array as da

sys.path.insert(0, str(Path(__file__).resolve().parent))

from _common import local_client, print_cluster_info, print_watch_urls, wait_for_browser


def demo() -> None:
    # 故意把每 worker 内存限制压得很低，方便看到色带变化
    with local_client(
        n_workers=2,
        threads_per_worker=1,
        memory_limit="200MB",
        processes=True,
    ) as client:
        print_cluster_info(client)
        print_watch_urls(
            client,
            ["/status", "/individual-workers-memory"],
            title="03 内存 spill",
        )

        print("创建多个大数组并 persist，观察内存色带爬升……")
        arrays = []
        for i in range(6):
            # ~80MB float64 量级（视压缩/碎片会有差异）
            a = da.ones((2_500, 2_500), chunks=(1_250, 1_250), dtype="float64")
            a = (a * (i + 1)).persist()
            arrays.append(a)
            print(f"  persisted array #{i + 1}")
            time.sleep(0.8)

        print("对 persist 结果做一次求和，可能触发 spill / 重算……")
        total = sum(a.sum() for a in arrays)
        t0 = time.perf_counter()
        val = total.compute()
        print(f"total={val}, elapsed={time.perf_counter() - t0:.1f}s")

        print("\n保持 persist 对象存活，方便你继续看内存页；看完回车释放。")
        wait_for_browser()
        del arrays


if __name__ == "__main__":
    demo()
