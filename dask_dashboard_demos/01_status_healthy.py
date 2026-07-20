"""
DEMO 01：健康并行 —— Status 上「好」长什么样。

对开 URL:
  /status  — Task Stream + Progress + Processing/CPU

期望现象:
  - Task Stream：多行（每线程一行）色块密实、白色空闲缝少
  - Progress：各 task-prefix 进度条稳步推进
  - Processing：各 worker 任务数大致均衡

调优启示:
  1. 分区数 ≈ worker 线程数的数倍时，并行通常更满
  2. 先记住「健康」基线，后面对照 idle / transfer / spill
"""

from __future__ import annotations

import sys
import time
from pathlib import Path

import dask.array as da
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))

from _common import local_client, print_cluster_info, print_watch_urls, wait_for_browser


def cpu_chunk(block: np.ndarray) -> np.ndarray:
    """轻度 CPU 工作，让 Task Stream 有足够长的色块可见。"""
    x = block.astype(np.float64)
    for _ in range(40):
        x = np.sin(x) + np.cos(x * 0.5)
    return x


def demo() -> None:
    with local_client(n_workers=2, threads_per_worker=2, memory_limit="1GB") as client:
        print_cluster_info(client)
        print_watch_urls(client, ["/status"], title="01 健康 Status")

        # 多 chunk → 多任务，填满 4 个线程
        x = da.random.random((8_000, 8_000), chunks=(1_000, 1_000))
        y = x.map_blocks(cpu_chunk, dtype=float)
        print("开始 compute（约十几秒）——请盯着 /status 的 Task Stream / Progress")
        t0 = time.perf_counter()
        result = y.mean().compute()
        print(f"mean={result:.6f}, elapsed={time.perf_counter() - t0:.1f}s")

        print("\n计算已结束；Task Stream 历史仍在，可慢慢对照 Progress 颜色。")
        wait_for_browser()


if __name__ == "__main__":
    demo()
