"""
DEMO 06：Graph / Groups —— 看清任务依赖形状。

对开 URL:
  /graph   — 当前图的依赖树
  /groups  — 按 task group 看依赖、内存、进度

期望现象:
  - Graph：小而清晰的节点/边
  - Groups：frompandas → assign → groupby-agg 等分组节点

说明:
  上游 /groups 在依赖 group 已释放时会 KeyError；本仓库已在
  .venv 的 distributed 里打了防护补丁。若你重装 distributed，需重打补丁
  或只看 /graph。

调优启示:
  1. 图过深/过宽 → 调度开销大，考虑融合阶段或增大分区
  2. Groups/Graph 适合判断依赖形状与哪一类 op 占进度
"""

from __future__ import annotations

import sys
from pathlib import Path

import dask.dataframe as dd
import numpy as np
import pandas as pd
from dask.distributed import wait

sys.path.insert(0, str(Path(__file__).resolve().parent))

from _common import local_client, print_cluster_info, print_watch_urls, wait_for_browser


def demo() -> None:
    with local_client(n_workers=2, threads_per_worker=2, memory_limit="512MB") as client:
        print_cluster_info(client)
        print_watch_urls(client, ["/graph", "/groups"], title="06 Graph / Groups")

        pdf = pd.DataFrame(
            {
                "cat": np.random.choice(list("ABCD"), size=8_000),
                "x": np.random.randn(8_000),
                "y": np.random.randn(8_000),
            }
        )

        # 分层 persist，方便在 Graph/Groups 上对照各阶段
        print("阶段1: from_pandas → persist")
        raw = dd.from_pandas(pdf, npartitions=4).persist()
        wait(raw)

        print("阶段2: assign(z=x*y) → persist")
        mid = raw.assign(z=raw.x * raw.y).persist()
        wait(mid)

        print("阶段3: groupby.agg → persist")
        out = mid.groupby("cat").agg({"x": "mean", "y": "sum", "z": "max"}).persist()
        wait(out)

        print("\n结果预览:")
        print(out.compute())

        print("\n现在打开 /graph 与 /groups；看完回车释放。")
        wait_for_browser()
        del raw, mid, out


if __name__ == "__main__":
    demo()
