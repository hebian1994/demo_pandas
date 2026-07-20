"""
DEMO：多目的地 CSV + as_completed 路由

- 若干分区算完后写到 A.csv，另一些写到 B.csv
- 用一个 as_completed 监听全部 Future
- 用 fut.key → dest 映射判断写到哪个文件
- 每个目的地单独跟踪是否已写过 header（新建 vs 追加）

Windows 下必须 if __name__ == "__main__"。
"""

from __future__ import annotations

import time
from collections import defaultdict
from pathlib import Path

import numpy as np
import pandas as pd
from dask.distributed import Client, LocalCluster, as_completed

OUTPUT_DIR = Path(__file__).parent / "output" / "multi_dest_demo"
DEST_A = OUTPUT_DIR / "A.csv"
DEST_B = OUTPUT_DIR / "B.csv"


def process_partition(pdf: pd.DataFrame, tag: str, sleep_s: float) -> pd.DataFrame:
    """分区业务：带 tag，并故意 sleep 打乱完成顺序。"""
    time.sleep(sleep_s)
    out = pdf.copy()
    out["amount"] = out["qty"] * out["price"]
    out["dest_tag"] = tag
    return out


def append_by_destination(
    dest_by_key: dict[str, str],
    futures,
) -> dict[str, int]:
    """
    混听所有 Future；按 fut.key 路由到对应 CSV。
    返回各目的地写入行数。
    """
    header_written: dict[str, bool] = defaultdict(lambda: False)
    files: dict[str, object] = {}
    totals: dict[str, int] = defaultdict(int)

    try:
        for fut in as_completed(futures):
            dest = dest_by_key[fut.key]
            pdf = fut.result()

            if dest not in files:
                Path(dest).parent.mkdir(parents=True, exist_ok=True)
                files[dest] = open(dest, "w", newline="", encoding="utf-8")

            first = not header_written[dest]
            pdf.to_csv(files[dest], index=False, header=first)
            header_written[dest] = True
            totals[dest] += len(pdf)

            print(
                f"  完成 key={fut.key!r} → {Path(dest).name}  "
                f"rows={len(pdf)}  mode={'新建+header' if first else '追加'}"
            )
    finally:
        for f in files.values():
            f.close()

    return dict(totals)


def main() -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    for p in (DEST_A, DEST_B):
        if p.exists():
            p.unlink()

    # 构造两组输入分区（行数不同，便于核对）
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

    # A 组：让后面的分区先完成；B 组同理，且与 A 交叉完成
    sleeps_a = [0.35, 0.20, 0.05]   # 期望完成序: A2, A1, A0
    sleeps_b = [0.30, 0.15, 0.25, 0.08]  # 期望大致: B3, B1, B2, B0

    cluster = LocalCluster(n_workers=4, threads_per_worker=1, dashboard_address=":8789")
    client = Client(cluster)
    print(f"Dashboard: {client.dashboard_link}")
    print(f"A → {DEST_A}  ({len(parts_a)} partitions)")
    print(f"B → {DEST_B}  ({len(parts_b)} partitions)")
    print()

    # --- 提交，并建立 fut.key → 目的地 映射 ---
    dest_by_key: dict[str, str] = {}

    futures_a = [
        client.submit(
            process_partition,
            parts_a[i],
            "A",
            sleeps_a[i],
            key=f"part-A-{i}",
        )
        for i in range(len(parts_a))
    ]
    futures_b = [
        client.submit(
            process_partition,
            parts_b[i],
            "B",
            sleeps_b[i],
            key=f"part-B-{i}",
        )
        for i in range(len(parts_b))
    ]

    for f in futures_a:
        dest_by_key[f.key] = str(DEST_A)
    for f in futures_b:
        dest_by_key[f.key] = str(DEST_B)

    all_futures = futures_a + futures_b
    print("映射表 fut.key → dest:")
    for k, dest in dest_by_key.items():
        print(f"  {k} → {Path(dest).name}")
    print()
    print("as_completed 混听全部，按完成先后路由写入:")

    t0 = time.perf_counter()
    totals = append_by_destination(dest_by_key, all_futures)
    elapsed = time.perf_counter() - t0

    print()
    print(f"耗时: {elapsed:.2f}s")
    print("各目的地行数:", totals)

    # 校验
    df_a = pd.read_csv(DEST_A)
    df_b = pd.read_csv(DEST_B)
    assert set(df_a["dest_tag"].unique()) == {"A"}
    assert set(df_b["dest_tag"].unique()) == {"B"}
    assert len(df_a) == sum(len(p) for p in parts_a) == totals[str(DEST_A)]
    assert len(df_b) == sum(len(p) for p in parts_b) == totals[str(DEST_B)]
    # header 只应出现一次：列名行 + 数据行
    with open(DEST_A, encoding="utf-8") as f:
        a_lines = f.readlines()
    with open(DEST_B, encoding="utf-8") as f:
        b_lines = f.readlines()
    assert a_lines[0].startswith("order_id")
    assert b_lines[0].startswith("order_id")
    assert sum(1 for line in a_lines if line.startswith("order_id")) == 1
    assert sum(1 for line in b_lines if line.startswith("order_id")) == 1

    print(f"\n校验通过: A.csv rows={len(df_a)}, B.csv rows={len(df_b)}")
    print(f"A 预览:\n{df_a.head(2)}")
    print(f"B 预览:\n{df_b.head(2)}")

    client.close()
    cluster.close()


if __name__ == "__main__":
    main()
