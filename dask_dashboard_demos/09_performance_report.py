"""
DEMO 09：performance_report —— 导出静态 HTML 做调参前后对比。

对开:
  生成的 HTML 报告（默认 output/dask-performance-report.html）
  也可同时打开实时 /status 对照

期望现象:
  - 报告内含 Task Stream、Profile、带宽等快照
  - 不依赖集群仍在跑，适合存档 / 发 PR / 对比两次实验

调优启示:
  1. 改 n_workers / chunks / 算法前后各导出一份，用浏览器并排看
  2. 实时 Dashboard 适合「当下调试」，report 适合「可复现的证据」
"""

from __future__ import annotations

import sys
import time
from pathlib import Path

import dask.array as da
from distributed import performance_report

sys.path.insert(0, str(Path(__file__).resolve().parent))

from _common import local_client, print_cluster_info, print_watch_urls, wait_for_browser

OUTPUT_DIR = Path(__file__).resolve().parent.parent / "output"
REPORT_PATH = OUTPUT_DIR / "dask-performance-report.html"


def demo() -> None:
    OUTPUT_DIR.mkdir(exist_ok=True)

    with local_client(n_workers=2, threads_per_worker=2, memory_limit="1GB") as client:
        print_cluster_info(client)
        print_watch_urls(client, ["/status"], title="09 performance_report")

        print(f"将写入报告: {REPORT_PATH}")
        with performance_report(filename=str(REPORT_PATH)):
            x = da.random.random((5_000, 5_000), chunks=(1_000, 1_000))
            y = (da.sin(x) + da.cos(x)).sum()
            t0 = time.perf_counter()
            val = y.compute()
            print(f"result={val:.6f}, elapsed={time.perf_counter() - t0:.1f}s")

        print(f"\n报告已生成: {REPORT_PATH}")
        print("用浏览器打开该 HTML（无需集群仍在运行）。")
        print("可选：再跑一遍改 chunks 的版本，换 filename 做前后对比。")
        wait_for_browser("看完报告或 /status 后按回车结束...")


if __name__ == "__main__":
    demo()
