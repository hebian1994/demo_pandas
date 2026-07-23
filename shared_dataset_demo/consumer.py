"""
Client B/C（消费方）：get_dataset 取共享数据，各自做不同计算

可多次启动（不同 --name），同时连同一集群、读同一份 published 数据。
"""

from __future__ import annotations

import argparse

from _common import DATASET_NAME, connect_client


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--name", default="consumer", help="本 Client 标签，便于日志区分")
    parser.add_argument(
        "--mode",
        choices=["by_region", "by_sku", "totals"],
        default=None,
        help="计算模式；默认按 name 自动选",
    )
    args = parser.parse_args()
    name = args.name
    mode = args.mode or (
        "by_region" if name.lower().endswith("a") or "alice" in name.lower()
        else "by_sku" if "bob" in name.lower()
        else "totals"
    )

    print(f"=== Consumer [{name}] mode={mode} ===")
    client = connect_client(name)

    names = client.list_datasets()
    print(f"  list_datasets = {names}")
    if DATASET_NAME not in names:
        raise SystemExit(
            f"集群上没有 {DATASET_NAME!r}。请先运行:  python publisher.py"
        )

    ddf = client.get_dataset(DATASET_NAME)
    print(f"  get_dataset({DATASET_NAME!r}) → {type(ddf).__name__}  npartitions={ddf.npartitions}")

    # 衍生列：各 Consumer 可独立加计算，共享同一底层分区
    ddf = ddf.assign(amount=ddf.qty * ddf.price)

    if mode == "by_region":
        out = ddf.groupby("region").amount.sum().compute().sort_index()
        print(f"\n  [{name}] 按 region 汇总 amount:\n{out}")
    elif mode == "by_sku":
        out = (
            ddf.groupby("sku")
            .agg({"qty": "sum", "amount": "sum"})
            .compute()
            .sort_index()
            .head(8)
        )
        print(f"\n  [{name}] 按 sku 汇总 (前 8):\n{out}")
    else:
        rows = int(ddf.shape[0].compute())
        amount = float(ddf.amount.sum().compute())
        print(f"\n  [{name}] totals: rows={rows:,}  amount_sum={amount:,.2f}")

    # 注意：这里不要 unpublish；其它 Consumer 可能还在用
    client.close()
    print(f"  [{name}] 断开（未 unpublish，共享数据仍在）")


if __name__ == "__main__":
    main()
