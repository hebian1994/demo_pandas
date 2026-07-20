"""
Dashboard DEMO 共用：起 LocalCluster、打印要对开的 URL、等待你在浏览器里观察。
"""

from __future__ import annotations

import atexit
import time
from contextlib import contextmanager
from typing import Iterator, Sequence
from urllib.parse import urlsplit

from dask.distributed import Client, LocalCluster

DEFAULT_DASHBOARD = ":8787"

# 注意：当前 distributed 的 /sitemap.json 会 500（routing KeyError: path），不要用它列 endpoint。
NAV_ENDPOINTS: Sequence[tuple[str, str]] = (
    ("/status", "总览：内存、Processing/CPU/Occupancy、Task Stream、Progress"),
    ("/workers", "各 worker 线程、内存、任务"),
    ("/tasks", "更长回溯的任务块视图"),
    ("/system", "集群级 CPU/内存/带宽/FD"),
    ("/profile", "统计采样火焰图"),
    ("/graph", "当前图依赖树"),
    ("/groups", "按 task group 看依赖/内存/进度"),
    ("/info", "worker 详情入口"),
    ("/hardware", "内存/磁盘/网络带宽探测"),
)


def dashboard_origin(client: Client) -> str:
    """client.dashboard_link 通常已带 /status，这里只取 scheme://host:port。"""
    parts = urlsplit(client.dashboard_link)
    return f"{parts.scheme}://{parts.netloc}"


def dashboard_url(client: Client, path: str = "/status") -> str:
    """把相对路径接到 dashboard 源上。"""
    if not path.startswith("/"):
        path = "/" + path
    return dashboard_origin(client) + path


def print_watch_urls(client: Client, paths: Sequence[str], title: str = "") -> None:
    if title:
        print(f"\n=== {title} ===")
    print(f"Dashboard 入口: {client.dashboard_link}")
    print("本课请打开:")
    for path in paths:
        print(f"  {dashboard_url(client, path)}")
    print()


def print_nav_endpoints(client: Client) -> None:
    """打印导航栏主页完整 URL（替代会 500 的 /sitemap.json）。"""
    print("导航栏主页（请用浏览器点这些，勿开 /sitemap.json — 当前 dask 会 500）:")
    for path, desc in NAV_ENDPOINTS:
        print(f"  {dashboard_url(client, path):<42} {desc}")
    print()


def wait_for_browser(prompt: str = "在浏览器里看完后按回车结束...") -> None:
    try:
        input(prompt)
    except EOFError:
        # 非交互环境：给几秒缓冲再退出
        print("(非交互: sleep 8s)")
        time.sleep(8)
    except KeyboardInterrupt:
        print("\n(已中断，正在关闭集群…)")


def _safe_close(obj, name: str) -> None:
    try:
        obj.close()
    except (KeyboardInterrupt, Exception) as e:
        print(f"关闭 {name} 时忽略: {type(e).__name__}: {e}")


@contextmanager
def local_client(
    *,
    n_workers: int = 2,
    threads_per_worker: int = 2,
    memory_limit: str | float | None = "1GB",
    dashboard_address: str = DEFAULT_DASHBOARD,
    processes: bool = True,
    **cluster_kwargs,
) -> Iterator[Client]:
    """
    启动 LocalCluster + Client；退出时关闭。
    若 8787 被占用，会提示改 dashboard_address。
    """
    try:
        cluster = LocalCluster(
            n_workers=n_workers,
            threads_per_worker=threads_per_worker,
            memory_limit=memory_limit,
            dashboard_address=dashboard_address,
            processes=processes,
            **cluster_kwargs,
        )
    except OSError as e:
        raise SystemExit(
            f"无法绑定 dashboard_address={dashboard_address!r}: {e}\n"
            f"请关掉占用端口的进程，或改用例如 dashboard_address=':8788'"
        ) from e

    client = Client(cluster)
    atexit.register(lambda: _safe_close(client, "client"))

    try:
        yield client
    finally:
        _safe_close(client, "client")
        _safe_close(cluster, "cluster")


def print_cluster_info(client: Client) -> None:
    info = client.scheduler_info()
    workers = info.get("workers", {})
    print(f"workers={len(workers)}, threads≈{sum(w.get('nthreads', 0) for w in workers.values())}")
    print(f"scheduler={info.get('address')}")
