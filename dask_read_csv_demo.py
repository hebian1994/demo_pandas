"""
Dask read_csv 确保列是 PyArrow dtype 的 DEMO。

要点：
1. 关键参数是 dtype_backend=\"pyarrow\"
2. 建议同时 engine=\"pyarrow\"（解析更快，少一次 NumPy 拷贝）
3. 仅 engine=\"pyarrow\" 不够，数值列仍可能是 float64/int64
4. 显式 dtype 请写 float64[pyarrow] / double，不要写 float[pyarrow]（那是 float32）
"""

from __future__ import annotations

from pathlib import Path

import dask.dataframe as dd
import pandas as pd

OUTPUT_DIR = Path(__file__).parent / "output"
CSV_PATH = OUTPUT_DIR / "dask_pyarrow_demo.csv"


def make_sample_csv() -> Path:
    OUTPUT_DIR.mkdir(exist_ok=True)
    pd.DataFrame(
        {
            "product": ["apple", "banana", "cherry"],
            "price": [91268829.02, 1.5, 3.891],
            "qty": [10, 20, 80],
            "flag": [True, False, True],
        }
    ).to_csv(CSV_PATH, index=False)
    return CSV_PATH


def show(title: str, ddf: dd.DataFrame) -> None:
    print(f"\n=== {title} ===")
    print("分区 meta dtypes:")
    print(ddf.dtypes)
    pdf = ddf.compute()
    print("compute() 后 pandas dtypes:")
    print(pdf.dtypes)
    print(pdf)


def demo() -> None:
    csv = make_sample_csv()
    print(f"样例 CSV: {csv}")

    # 1) 默认：字符串可能已是 string[pyarrow]，但数值仍是 NumPy
    show("默认 read_csv", dd.read_csv(csv))

    # 2) 推荐：engine + dtype_backend 都指定 pyarrow
    ddf = dd.read_csv(csv, engine="pyarrow", dtype_backend="pyarrow")
    show("engine='pyarrow' + dtype_backend='pyarrow'（推荐）", ddf)

    # 3) 只有 dtype_backend 也对；只有 engine 则数值仍是 NumPy
    show("仅 dtype_backend='pyarrow'", dd.read_csv(csv, dtype_backend="pyarrow"))
    show("仅 engine='pyarrow'（不够）", dd.read_csv(csv, engine="pyarrow"))

    # 4) 显式指定每列 Arrow 类型（最稳，适合生产）
    ddf_typed = dd.read_csv(
        csv,
        engine="pyarrow",
        dtype_backend="pyarrow",
        dtype={
            "product": "string[pyarrow]",
            "price": "float64[pyarrow]",  # = double[pyarrow]；别写 float[pyarrow]
            "qty": "int64[pyarrow]",
            "flag": "bool[pyarrow]",
        },
    )
    show("显式 dtype 字典（生产推荐）", ddf_typed)

    # 校验：64 位浮点精度还在
    prices = ddf_typed["price"].compute().tolist()
    print("\nprice 值:", prices)
    assert prices[0] == 91268829.02, "不应被 float32 舍入成 91268832"


if __name__ == "__main__":
    demo()
