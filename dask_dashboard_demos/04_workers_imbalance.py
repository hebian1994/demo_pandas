"""
DEMO 04：Worker 负载倾斜 —— Processing / Occupancy 不均。

对开 URL:
  /workers  — 各 worker 状态
  /status   — Processing / Occupancy / CPU 标签

期望现象:
  - 少数「慢任务」拖住某个 worker，Occupancy 条特别长
  - 其它 worker 较早变空闲（Processing 接近白）
  - Workers 页上 CPU/任务数明显不对称

调优启示:
  1. 分区大小不均、或个别 key 很重 → 倾斜（skew）
  2. 先看 Occupancy/Processing 定位倾斜，再考虑重分区 / 拆热点
"""

from __future__ import annotations

import sys
import time
from pathlib import Path

from dask import delayed

sys.path.insert(0, str(Path(__file__).resolve().parent))

from _common import local_client, print_cluster_info, print_watch_urls, wait_for_browser


@delayed
def fast_task(i: int) -> int:
    time.sleep(0.15)
    return i


@delayed
def slow_task(i: int) -> int:
    """故意极慢，制造单 worker 长占用。"""
    time.sleep(6.0)
    return i


def demo() -> None:
    with local_client(n_workers=2, threads_per_worker=2, memory_limit="512MB") as client:
        print_cluster_info(client)
        print_watch_urls(client, ["/workers", "/status"], title="04 worker 倾斜")

        print("提交：大量快任务 + 少量极慢任务 —— 盯 Occupancy / Processing")
        tasks = [fast_task(i) for i in range(40)]
        tasks += [slow_task(i) for i in range(2)]
        t0 = time.perf_counter()
        results = client.compute(tasks, sync=True)
        print(f"got {len(results)} results, elapsed={time.perf_counter() - t0:.1f}s")

        print("\n回看 Workers / Status 历史：慢任务对应的那一段 Occupancy 会突出。")
        wait_for_browser()


if __name__ == "__main__":
    demo()
