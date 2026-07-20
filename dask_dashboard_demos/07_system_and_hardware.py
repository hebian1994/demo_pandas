"""
DEMO 07：System / Hardware —— 机器级资源 vs Dask 任务视图。

对开 URL:
  /system    — 集群级 CPU / 内存 / 带宽 / FD
  /hardware  — 内存 / 磁盘 / 网络带宽探测

期望现象:
  - 混合「写盘 + CPU」负载时，System 曲线抬升
  - Hardware 页给出本机带宽量级（调 chunk 大小的参考）
  - 对照 /status：任务在跑 ≠ 一定是 CPU 瓶颈（也可能是磁盘）

调优启示:
  1. Task Stream 忙但 System CPU 低 → 可能在等 I/O / 锁 / GIL
  2. System 磁盘打满 → 减小并发写、换更快盘、或减少 spill
"""

from __future__ import annotations

import sys
import tempfile
import time
from pathlib import Path

import dask.array as da
import numpy as np
from dask import delayed

sys.path.insert(0, str(Path(__file__).resolve().parent))

from _common import local_client, print_cluster_info, print_watch_urls, wait_for_browser


@delayed
def write_chunk(path: str, n: int, seed: int) -> int:
    rng = np.random.default_rng(seed)
    arr = rng.random(n, dtype=np.float64)
    out = Path(path) / f"chunk_{seed}.npy"
    np.save(out, arr)
    # 再读回来，制造磁盘带宽
    _ = np.load(out)
    return int(arr.size)


def demo() -> None:
    with local_client(n_workers=2, threads_per_worker=2, memory_limit="1GB") as client:
        print_cluster_info(client)
        print_watch_urls(client, ["/system", "/hardware"], title="07 System / Hardware")

        with tempfile.TemporaryDirectory(prefix="dask_dash_io_") as tmp:
            print(f"临时目录: {tmp}")
            print("\n[1] 并发写读 .npy —— 盯 /system 磁盘与带宽")
            jobs = [write_chunk(tmp, 8_000_000, i) for i in range(12)]
            t0 = time.perf_counter()
            sizes = client.compute(jobs, sync=True)
            print(f"  wrote {sum(sizes)} floats, elapsed={time.perf_counter() - t0:.1f}s")

            print("\n[2] CPU 密集数组 —— 对照 System CPU 曲线")
            x = da.random.random((6_000, 6_000), chunks=(1_000, 1_000))
            t0 = time.perf_counter()
            _ = np.sin(x).sum().compute()
            print(f"  elapsed={time.perf_counter() - t0:.1f}s")

        print("\n打开 /hardware 看带宽探测；结合刚才两段负载理解 System 曲线。")
        wait_for_browser()


if __name__ == "__main__":
    demo()
