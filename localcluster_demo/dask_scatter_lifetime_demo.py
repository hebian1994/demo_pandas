"""
DEMO：scatter 的数据在 Client 断开后会不会还留在集群？

结论（默认）:
  会删。scatter / persist 一样，都靠 Client 对 Future 的引用计数；
  Client.close() 后引用归零，worker 上的数据被释放。

例外:
  client.publish_dataset(...) 显式挂名后，别的 Client 还能 get_dataset；
  需自己 unpublish（或集群关掉）才会清掉。

前置:
  终端 1:  python dask_local_cluster_server.py
  终端 2:  python dask_scatter_lifetime_demo.py

Windows 下必须 if __name__ == "__main__"。
"""

from __future__ import annotations

import time
from pathlib import Path

import numpy as np
from dask.distributed import Client

ADDRESS_FILE = Path(__file__).resolve().parent / ".dask_scheduler_address"
DEFAULT_SCHEDULER = "tcp://127.0.0.1:8786"
DATASET_NAME = "scatter_demo_payload"


def connect() -> Client:
    if ADDRESS_FILE.exists():
        addr = ADDRESS_FILE.read_text(encoding="utf-8").strip()
    else:
        addr = DEFAULT_SCHEDULER
    try:
        return Client(addr)
    except OSError as e:
        raise SystemExit(
            f"无法连接 {addr}: {e}\n"
            "请先运行:  python dask_local_cluster_server.py"
        ) from e


def worker_held_keys(client: Client) -> dict[str, list]:
    """各 worker 当前持有的 key（谁有什么）。"""
    return {addr: list(keys) for addr, keys in client.has_what().items()}


def count_keys(held: dict[str, list]) -> int:
    return sum(len(v) for v in held.values())


def print_held(client: Client, title: str) -> int:
    held = worker_held_keys(client)
    n = count_keys(held)
    print(f"\n--- {title} ---")
    print(f"  worker 上 key 总数 = {n}")
    for addr, keys in held.items():
        short = addr.split("//")[-1]
        preview = keys[:3]
        more = f" …(+{len(keys) - 3})" if len(keys) > 3 else ""
        print(f"  [{short}] {len(keys)} keys  例={preview}{more}")
    return n


def make_payload() -> list[np.ndarray]:
    # 约 64MB：4 块 × 16MB，方便在 Dashboard 内存条上看到变化
    rng = np.random.default_rng(0)
    return [rng.random(2_000_000, dtype=np.float64) for _ in range(4)]  # 4×16MB


def demo_scatter_dies_with_client() -> None:
    print("\n=== A. 普通 scatter：Client 断开后数据消失 ===")
    client = connect()
    print(f"  Client#1  dashboard={client.dashboard_link}")
    print_held(client, "scatter 前")

    chunks = make_payload()
    nbytes = sum(a.nbytes for a in chunks)
    print(f"  本地 payload ≈ {nbytes / (1024 * 1024):.1f} MB，准备 scatter…")

    futures = client.scatter(chunks)  # list[Future]，数据进 worker
    # 确认能算
    first = client.submit(lambda a: float(a.sum()), futures[0]).result()
    print(f"  scatter 后试算 sum(chunk0) = {first:.4e}")
    n_after = print_held(client, "scatter 后（Client#1 仍连着）")
    assert n_after > 0, "scatter 后 worker 应持有 key"

    print("  → 关闭 Client#1 …")
    client.close()
    time.sleep(1.0)  # 给调度器一点时间做 release

    client2 = connect()
    print(f"  Client#2 已连接  dashboard={client2.dashboard_link}")
    n_later = print_held(client2, "Client#1 断开后（Client#2 视角）")
    if n_later == 0:
        print("  结论: scatter 的数据已随 Client#1 释放（与 persist 相同）。")
    else:
        print(f"  注意: 仍看到 {n_later} 个 key（可能是集群其它任务残留）。")
    client2.close()


def demo_publish_survives() -> None:
    print("\n=== B. scatter + publish_dataset：断开后仍可取 ===")
    client = connect()
    # 清理可能残留的同名 dataset
    try:
        client.unpublish_dataset(DATASET_NAME)
    except KeyError:
        pass

    chunks = make_payload()
    futures = client.scatter(chunks)
    client.publish_dataset(**{DATASET_NAME: futures})
    print(f"  已 publish_dataset({DATASET_NAME!r})，keys={len(futures)}")
    print_held(client, "publish 后")

    print("  → 关闭 Client#1（不 unpublish）…")
    client.close()
    time.sleep(1.0)

    client2 = connect()
    names = client2.list_datasets()
    print(f"  Client#2 list_datasets = {names}")
    assert DATASET_NAME in names, "publish 的名字应还在"

    got = client2.get_dataset(DATASET_NAME)
    # got 是 list[Future]；对每个分区求 sum 再汇总
    part_sums = client2.gather(
        [client2.submit(lambda a: float(a.sum()), f) for f in got]
    )
    total = float(sum(part_sums))
    print(f"  get_dataset 后 sum(all chunks) = {total:.4e}")
    print_held(client2, "get_dataset 后（数据仍在）")
    print("  结论: publish 后，新 Client 仍能拿到 scatter 出去的数据。")

    client2.unpublish_dataset(DATASET_NAME)
    # 释放本 Client 对 Future 的引用后再关，数据才会清
    del got
    client2.close()
    time.sleep(1.0)

    client3 = connect()
    names3 = client3.list_datasets()
    n_end = print_held(client3, "unpublish + Client#2 断开后")
    print(f"  list_datasets = {names3}")
    print(f"  结论: unpublish 且无 Client 再持有引用后，数据被释放 (keys≈{n_end})。")
    client3.close()


def main() -> None:
    print("scatter 生命周期 DEMO（集群需已由 server 脚本启动）")
    demo_scatter_dies_with_client()
    demo_publish_survives()
    print("\n全部 DEMO 完成。")


if __name__ == "__main__":
    main()
