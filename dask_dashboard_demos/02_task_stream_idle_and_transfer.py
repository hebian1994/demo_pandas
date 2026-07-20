"""
DEMO 02：Task Stream 上的坏信号 —— 白色空闲 + 红色 transfer。

对开 URL:
  /status  — Task Stream（主）
  /tasks   — 更长回溯的任务块视图

期望现象:
  A) 过少任务：4 个线程里只有 1 条在干活，大量白色空闲缝
  B) 数据放在 worker0、任务指定到 worker1：出现红色 transfer 条

调优启示:
  1. 白缝多 → 提高并行度（更多分区 / 合理 n_workers）
  2. 长红条 → 减小 chunk、减少不必要的 shuffle、注意数据局部性

说明:
  [A] 用 delayed+sleep 演示空闲，避免超大单 chunk 把 worker 内存打爆
"""

from __future__ import annotations

import sys
import time
from pathlib import Path

import numpy as np
from dask import delayed

sys.path.insert(0, str(Path(__file__).resolve().parent))

from _common import local_client, print_cluster_info, print_watch_urls, wait_for_browser


@delayed
def lone_slow_task(seconds: float) -> float:
    """单个慢任务：其它线程只能闲着 → Task Stream 白缝。"""
    time.sleep(seconds)
    return seconds


def demo() -> None:
    with local_client(n_workers=2, threads_per_worker=2, memory_limit="1GB") as client:
        print_cluster_info(client)
        print_watch_urls(client, ["/status", "/tasks"], title="02 idle + transfer")

        workers = list(client.scheduler_info()["workers"])
        if len(workers) < 2:
            raise SystemExit("需要至少 2 个 worker 才能演示跨机传输")
        w0, w1 = workers[0], workers[1]

        # --- A) 任务数 << 线程数 → 白缝 ---
        print("\n[A] 只提交 1 个慢任务，集群有 4 线程 → 盯 Task Stream 白色空闲")
        t0 = time.perf_counter()
        _ = client.compute([lone_slow_task(4.0)], sync=True)
        print(f"  elapsed={time.perf_counter() - t0:.1f}s")
        time.sleep(1)

        # --- B) 强制跨 worker 搬数据 → 红条 ---
        print("\n[B] scatter 到 worker0，再在 worker1 上计算 —— 盯红色 transfer")
        # 约 64MB/块 × 多轮，足够在 Task Stream 上看到红条，又远低于 1GB limit
        t0 = time.perf_counter()
        total = 0.0
        for i in range(8):
            arr = np.random.random(8_000_000)  # ~64MB float64
            fut = client.scatter(arr, workers=[w0], direct=True)
            # 指定到另一台 worker 执行 → 必须先红条传数据
            total += client.submit(lambda x: float(x.sum()), fut, workers=[w1]).result()
            print(f"  round {i + 1}/8 done", flush=True)
        print(f"  checksum={total:.3f}, elapsed={time.perf_counter() - t0:.1f}s")

        print("\n对照 [A] 的白缝与 [B] 的红条，再回车结束。")
        wait_for_browser()


if __name__ == "__main__":
    demo()
