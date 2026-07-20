"""
DEMO 08：Fine Performance Metrics —— 时间花在 execute / transfer / idle？

对开 URL:
  More... → Fine Performance Metrics
  （或）/individual-aggregate-time-per-action
  （或）/individual-compute-time-per-key

期望现象:
  - 「过多过小任务」：调度/开销占比高，idle 或细碎 execute 增多
  - 「合理分区」：execute 占主导，空闲与无意义传输更少
  - 左侧按 function、中间按 activity 可对照

调优启示:
  1. 任务太碎 → 增大 chunk / 减少 map 层数
  2. transfer 占比高 → 查 shuffle、广播、数据局部性
  3. idle 高且 Task Stream 白缝多 → 并行度或依赖链有问题
"""

from __future__ import annotations

import sys
import time
from pathlib import Path

import dask.array as da

sys.path.insert(0, str(Path(__file__).resolve().parent))

from _common import local_client, print_cluster_info, print_watch_urls, wait_for_browser


def demo() -> None:
    with local_client(n_workers=2, threads_per_worker=2, memory_limit="1GB") as client:
        print_cluster_info(client)
        print_watch_urls(
            client,
            [
                "/individual-aggregate-time-per-action",
                "/individual-compute-time-per-key",
                "/status",
            ],
            title="08 Fine Performance Metrics",
        )
        print("也请在 Dashboard 导航 More... 里打开「Fine Performance Metrics」。")

        shape = (4_000, 4_000)

        print("\n[坏] 极小 chunk → 海量小任务（看 metrics 里开销/碎片）")
        x = da.random.random(shape, chunks=(100, 100))
        t0 = time.perf_counter()
        _ = (x + 1).sum().compute()
        print(f"  elapsed={time.perf_counter() - t0:.1f}s, npartitions≈{x.npartitions}")
        time.sleep(1)

        print("\n[好] 合理 chunk → 同样计算（对照 Fine Metrics）")
        x = da.random.random(shape, chunks=(1_000, 1_000))
        t0 = time.perf_counter()
        _ = (x + 1).sum().compute()
        print(f"  elapsed={time.perf_counter() - t0:.1f}s, npartitions≈{x.npartitions}")

        print("\n两段跑完后刷新 Fine Metrics / aggregate-time-per-action 对比。")
        wait_for_browser()


if __name__ == "__main__":
    demo()
