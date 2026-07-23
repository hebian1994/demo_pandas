"""
Client A（发布方）：造 DataFrame → persist 到集群 → publish_dataset

发布后即使本 Client 断开，其它 Client 仍可用 get_dataset 取同一份数据。
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import dask.dataframe as dd
from dask.distributed import wait

from _common import DATASET_NAME, connect_client


def main() -> None:
    print("=== Publisher：造数并 publish 到集群 ===")
    client = connect_client("publisher")

    # 若上次残留同名数据集，先摘掉
    if DATASET_NAME in client.list_datasets():
        print(f"  发现旧数据集 {DATASET_NAME!r}，unpublish…")
        client.unpublish_dataset(DATASET_NAME)

    rng = np.random.default_rng(42)
    n = 200_000
    pdf = pd.DataFrame(
        {
            "region": rng.choice(["east", "west", "north", "south"], size=n),
            "sku": rng.integers(1000, 1020, size=n),
            "qty": rng.integers(1, 20, size=n),
            "price": rng.random(n) * 100.0,
        }
    )
    ddf = dd.from_pandas(pdf, npartitions=8)
    print(f"  本地造表 rows={n:,} partitions={ddf.npartitions}")

    print("  persist → 钉在 worker 内存…")
    ddf_p = client.persist(ddf)
    wait(ddf_p)
    nbytes = int(ddf_p.memory_usage(deep=True).sum().compute())
    print(f"  persist 完成  ≈ {nbytes / 1e6:.1f} MB")

    # 关键名挂到 scheduler；其它 Client 用同名 get_dataset
    client.publish_dataset(**{DATASET_NAME: ddf_p})
    print(f"  publish_dataset({DATASET_NAME!r}) 成功")
    print(f"  list_datasets = {client.list_datasets()}")

    # 故意关掉发布方：证明数据不依赖这个 Client 进程
    client.close()
    print("  Publisher 已断开；数据集仍留在集群上，供其它 Client 使用。")


if __name__ == "__main__":
    main()
