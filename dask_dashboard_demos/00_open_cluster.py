"""
DEMO 00：只起 LocalCluster，熟悉 Dashboard 导航。

对开 URL:
  /status   — 总览入口
  /workers  — worker 列表
  /info     — worker 详情入口
  （脚本还会打印完整导航栏 URL 列表）

说明:
  不要打开 /sitemap.json —— 当前 distributed 会 500
  （routing.py KeyError: 'path'，属上游 bug）

期望现象:
  - Status 上几乎没有任务色块（空闲集群）
  - Workers 页能看到 n_workers / nthreads / 内存限制
  - 导航打印的各导航页均可 200 打开

调优启示:
  1. 永远先确认 dashboard_link，端口被占用时地址会变
  2. 先认路（导航栏），再带着负载去读信号
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

from _common import (
    local_client,
    print_cluster_info,
    print_nav_endpoints,
    print_watch_urls,
    wait_for_browser,
)


def demo() -> None:
    with local_client(n_workers=2, threads_per_worker=2, memory_limit="512MB") as client:
        print_cluster_info(client)
        print_watch_urls(
            client,
            ["/status", "/workers", "/info"],
            title="00 开集群 / 认路",
        )
        print_nav_endpoints(client)
        print("现在集群是空闲的。点开导航栏各页看一眼布局，再回车进入下一课。")
        wait_for_browser()


if __name__ == "__main__":
    demo()
