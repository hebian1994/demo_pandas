"""
DEMO：每个目的地各自 as_completed，多线程同时监听

对比「一个 as_completed 混听全部」：
  - 混听：A/B 的完成事件进同一队列，写 A 时写 B 要等（client 单线程处理）
  - 分听：每个目的地一个线程 + 一个 as_completed
      → 同一 CSV 内仍「来一个写一个」（单文件安全）
      → 不同 CSV 可并行追加（互不阻塞）

Windows 下必须 if __name__ == "__main__"。
"""

from __future__ import annotations

import time
from concurrent.futures import ThreadPoolExecutor, wait
from pathlib import Path

import numpy as np
import pandas as pd
from dask.distributed import Client, Future, LocalCluster, as_completed

OUTPUT_DIR = Path(__file__).parent / "output" / "multi_dest_parallel_demo"
DEST_A = OUTPUT_DIR / "A.csv"
DEST_B = OUTPUT_DIR / "B.csv"


def process_partition(pdf: pd.DataFrame, tag: str, sleep_s: float) -> pd.DataFrame:
    time.sleep(sleep_s)
    out = pdf.copy()
    out["amount"] = out["qty"] * out["price"]
    out["dest_tag"] = tag
    return out


def drain_one_destination(
    dest_path: str,
    futures: list[Future],
    t0: float,
) -> tuple[str, int]:
    """
    只监听「写到 dest_path」的那一组 Future。
    同一文件内串行追加；与其它目的地的 drain 并行运行。
    """
    Path(dest_path).parent.mkdir(parents=True, exist_ok=True)
    total = 0
    first = True
    name = Path(dest_path).name
    with open(dest_path, "w", newline="", encoding="utf-8") as f:
        for fut in as_completed(futures):
            pdf = fut.result()
            t = time.perf_counter() - t0
            mode = "新建+header" if first else "追加"
            pdf.to_csv(f, index=False, header=first)
            # 故意慢一点写盘，方便看出 A/B 写操作时间重叠
            time.sleep(0.12)
            f.flush()
            first = False
            total += len(pdf)
            print(f"  t=+{t:.3f}s  [{name}] key={fut.key!r} rows={len(pdf)} {mode}")
    return dest_path, total


def main() -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    for p in (DEST_A, DEST_B):
        if p.exists():
            p.unlink()

    rng = np.random.default_rng(0)
    parts_a = [
        pd.DataFrame(
            {
                "order_id": np.arange(i * 100, i * 100 + 100),
                "qty": rng.integers(1, 5, 100),
                "price": rng.random(100) * 10,
            }
        )
        for i in range(3)
    ]
    parts_b = [
        pd.DataFrame(
            {
                "order_id": np.arange(1000 + i * 50, 1000 + i * 50 + 50),
                "qty": rng.integers(1, 5, 50),
                "price": rng.random(50) * 10,
            }
        )
        for i in range(4)
    ]

    # 让 A/B 都有任务几乎同时完成 → 分听时两边应交错打印
    sleeps_a = [0.05, 0.20, 0.35]
    sleeps_b = [0.06, 0.18, 0.28, 0.40]

    cluster = LocalCluster(n_workers=4, threads_per_worker=1, dashboard_address=":8790")
    client = Client(cluster)
    print(f"Dashboard: {client.dashboard_link}")
    print("每个目的地一个线程 + 各自 as_completed，同时监听\n")

    futures_a = [
        client.submit(process_partition, parts_a[i], "A", sleeps_a[i], key=f"part-A-{i}")
        for i in range(len(parts_a))
    ]
    futures_b = [
        client.submit(process_partition, parts_b[i], "B", sleeps_b[i], key=f"part-B-{i}")
        for i in range(len(parts_b))
    ]

    groups = {
        str(DEST_A): futures_a,
        str(DEST_B): futures_b,
    }

    t0 = time.perf_counter()
    # 同时启动多个 drain；同一 dest 只用一个线程写，避免同文件竞争
    with ThreadPoolExecutor(max_workers=len(groups)) as pool:
        jobs = [
            pool.submit(drain_one_destination, dest, futs, t0)
            for dest, futs in groups.items()
        ]
        wait(jobs)
        results = [j.result() for j in jobs]

    elapsed = time.perf_counter() - t0
    print(f"\n总耗时: {elapsed:.2f}s")
    for dest, n in results:
        print(f"  {Path(dest).name}: {n} rows")

    df_a = pd.read_csv(DEST_A)
    df_b = pd.read_csv(DEST_B)
    assert len(df_a) == 300 and set(df_a["dest_tag"]) == {"A"}
    assert len(df_b) == 200 and set(df_b["dest_tag"]) == {"B"}
    print("校验通过。看上面日志：A/B 的 t=+ 时间应交错，说明两边在并行写。")

    client.close()
    cluster.close()


if __name__ == "__main__":
    main()
