"""
DEMO：Client 连接 LocalCluster，造 ~500MB DataFrame 后按分区 map 计算

前置:
  终端 1:  python dask_local_cluster_server.py
  终端 2:  python dask_client_dataframe_demo.py

流程:
  1. 在集群上造约 500MB 的 dask DataFrame
  2. client.persist 把分区钉在 worker 内存
  3. to_delayed() → client.compute → 得到各分区 Future
  4. client.map(业务函数, futures) 在集群上按分区计算

Windows 下必须 if __name__ == "__main__"。
"""

from __future__ import annotations

import time
from pathlib import Path

import dask.dataframe as dd
import numpy as np
import pandas as pd
from dask.distributed import Client, Future, wait

ADDRESS_FILE = Path(__file__).resolve().parent / ".dask_scheduler_address"
DEFAULT_SCHEDULER = "tcp://127.0.0.1:8786"

TARGET_MB = 800
# 5 列 float64 → 每行 40 字节；再加一列 category(int8) 可忽略
N_FLOAT_COLS = 5
BYTES_PER_ROW = 8 * N_FLOAT_COLS
N_PARTITIONS = 16


def connect() -> Client:
    if ADDRESS_FILE.exists():
        addr = ADDRESS_FILE.read_text(encoding="utf-8").strip()
        print(f"  从文件读取 scheduler: {addr}")
    else:
        addr = DEFAULT_SCHEDULER
        print(f"  未找到 {ADDRESS_FILE.name}，使用默认: {addr}")

    try:
        return Client(addr)
    except OSError as e:
        raise SystemExit(
            f"无法连接 {addr}: {e}\n"
            "请先运行:  python dask_local_cluster_server.py"
        ) from e


def make_partition(part_id: int, n_rows: int, seed: int) -> pd.DataFrame:
    """在 worker 上造一个分区，避免 client 本机先扛 500MB。"""
    rng = np.random.default_rng(seed + part_id)
    data = {
        "category": rng.integers(0, 8, size=n_rows, dtype=np.int8),
        "qty": rng.integers(1, 50, size=n_rows, dtype=np.int32),
    }
    for i in range(N_FLOAT_COLS):
        data[f"v{i}"] = rng.normal(loc=100.0, scale=20.0, size=n_rows)
    return pd.DataFrame(data)


def process_partition(pdf: pd.DataFrame) -> dict:
    """
    分区业务：入参是 Future 解包后的 pandas DataFrame。
    返回轻量摘要，避免把整分区再拉回 client。
    """
    amount = pdf["qty"].to_numpy(dtype=np.float64)
    for i in range(N_FLOAT_COLS):
        amount = amount * pdf[f"v{i}"].to_numpy(dtype=np.float64)
    # 故意做一点 CPU 工作，方便在 Dashboard Task Stream 里看到
    acc = 0.0
    for i in range(200_000):
        acc += (i % 97) * 0.001
    return {
        "rows": len(pdf),
        "amount_sum": float(amount.sum() + (acc % 1.0)),
        "category_nunique": int(pdf["category"].nunique()),
        "nbytes": int(pdf.memory_usage(deep=True).sum()),
    }


def build_dataframe(client: Client) -> dd.DataFrame:
    n_rows = (TARGET_MB * 1024 * 1024) // BYTES_PER_ROW
    rows_per_part = n_rows // N_PARTITIONS
    print(f"\n=== B. 造约 {TARGET_MB}MB DataFrame ===")
    print(f"  目标行数 ≈ {n_rows:,}  ({N_FLOAT_COLS}×float64 ≈ {BYTES_PER_ROW} B/行)")
    print(f"  partitions = {N_PARTITIONS}  rows/part ≈ {rows_per_part:,}")

    # 在各 worker 上并行造分区 → from_delayed 拼成 dask DF
    build_futs: list[Future] = client.map(
        make_partition,
        list(range(N_PARTITIONS)),
        [rows_per_part] * N_PARTITIONS,
        [42] * N_PARTITIONS,
    )
    meta = make_partition(0, 0, 0)  # 空表只作 schema
    ddf = dd.from_delayed(build_futs, meta=meta)
    return ddf


def demo_persist_delayed_map(client: Client) -> None:
    ddf = build_dataframe(client)

    print("\n=== C. persist（钉在集群内存）===")
    t0 = time.perf_counter()
    ddf_p = client.persist(ddf)
    wait(ddf_p)
    est_bytes = int(ddf_p.memory_usage(deep=True).sum().compute())
    est_mb = est_bytes / (1024 * 1024)
    print(f"  persist 完成  耗时 {time.perf_counter() - t0:.2f}s")
    print(f"  memory_usage ≈ {est_mb:.1f} MB  (partitions={ddf_p.npartitions})")

    print("\n=== D. to_delayed → compute → Futures ===")
    delayed_parts = ddf_p.to_delayed()
    part_futures: list[Future] = client.compute(delayed_parts)
    # 等分区 Future 就绪（数据已在 worker；这里主要是拿到可 map 的句柄）
    wait(part_futures)
    print(f"  分区 Future 数 = {len(part_futures)}")
    print(f"  示例 key     = {part_futures[0].key}")

    print("\n=== E. client.map 提交到集群 ===")
    t1 = time.perf_counter()
    result_futures = client.map(process_partition, part_futures)
    summaries = client.gather(result_futures)
    elapsed = time.perf_counter() - t1

    total_rows = sum(s["rows"] for s in summaries)
    total_nbytes = sum(s["nbytes"] for s in summaries)
    amount_sum = sum(s["amount_sum"] for s in summaries)
    print(f"  map+gather 耗时 {elapsed:.2f}s")
    print(f"  总行数     = {total_rows:,}")
    print(f"  总 nbytes  = {total_nbytes / (1024 * 1024):.1f} MB")
    print(f"  amount_sum = {amount_sum:.4e}")
    print(f"  前 3 个分区摘要: {summaries[:3]}")
    print(f"\n  Dashboard: {client.dashboard_link}")


def main() -> None:
    print("=== A. 连接 LocalCluster ===")
    client = connect()
    print(f"  dashboard  = {client.dashboard_link}")
    info = client.scheduler_info()
    workers = info.get("workers", {})
    print(f"  workers    = {len(workers)}")
    print(f"  scheduler  = {info.get('address')}")

    try:
        demo_persist_delayed_map(client)
        print("\n全部 DEMO 完成。Client 已断开，集群仍在 server 脚本里运行。")
    finally:
        client.close()


if __name__ == "__main__":
    main()
