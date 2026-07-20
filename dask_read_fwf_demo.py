"""
Dask read_fwf DEMO：
1. 用 widths + names 读定宽文件
2. 先全部按 str 读入（engine/dtype_backend=pyarrow）
3. 再转成指定的目标类型

短行注意：
- dtype=str 时，缺字段会变成字面量 '<NA>'，转类型前要清成真正空值
- 半截字段（如 price 只读到 '3.89'）可能被 to_numeric 成功解析成错误数，不会报错
"""

from __future__ import annotations

from pathlib import Path

import dask.dataframe as dd
import pandas as pd

OUTPUT_DIR = Path(__file__).parent / "output"
FWF_PATH = OUTPUT_DIR / "short_last_line_widths.fwf"

# widths: name(5) + price(10) + qty(8) = 23
WIDTHS = [5, 10, 8]
NAMES = ["name", "price", "qty"]

# Step1 全按 str 读入后，Step2 再转成这些类型
TARGET_DTYPES = {
    "name": "string[pyarrow]",
    "price": "float64[pyarrow]",  # = double；不要写 float[pyarrow]
    "qty": "int64[pyarrow]",
}


def make_fwf_with_short_last_line() -> Path:
    OUTPUT_DIR.mkdir(exist_ok=True)
    lines = [
        # name5 | price10      | qty8
        "apple" + "91268829.0" + "00000010",  # 23
        "banan" + "0000001.50" + "00000020",  # 23
        "cherr" + "3.89",  # 仅 9：price 被截断，qty 缺失
    ]
    FWF_PATH.write_text("\n".join(lines), encoding="utf-8", newline="\n")
    return FWF_PATH


def clean_na_strings(pdf: pd.DataFrame) -> pd.DataFrame:
    """dtype=str 读入后，短行缺字段常变成字面量 '<NA>'。"""
    out = pdf.copy()
    for col in out.columns:
        s = out[col].astype("string").str.strip()
        s = s.replace({"": pd.NA, "<NA>": pd.NA, "None": pd.NA, "nan": pd.NA})
        out[col] = s
    return out


def _is_numeric_dtype_name(dtype: str) -> bool:
    name = str(dtype).lower()
    return any(k in name for k in ("int", "uint", "float", "double", "decimal"))


def cast_to_target_types(
    pdf: pd.DataFrame,
    target_dtypes: dict[str, str],
) -> pd.DataFrame:
    """清 NA 字面量后，转成传入的目标类型；非法数字 errors='raise'。"""
    out = clean_na_strings(pdf)
    for col, dtype in target_dtypes.items():
        if _is_numeric_dtype_name(dtype):
            out[col] = pd.to_numeric(out[col], errors="raise")
    return out.astype(target_dtypes)


def demo() -> None:
    path = make_fwf_with_short_last_line()
    print(f"文件: {path}")
    print(f"bytes: {path.read_bytes()!r}")
    print("各行长度:", [len(x) for x in path.read_text(encoding="utf-8").splitlines()])
    print(f"widths={WIDTHS}, names={NAMES}")
    print()

    # ---------- Step1: widths + names，先全部当 str ----------
    print("=== Step1: read_fwf(widths, names, dtype=str) ===")
    ddf = dd.read_fwf(
        path,
        widths=WIDTHS,
        names=NAMES,
        header=None,
        dtype=str,
        engine="pyarrow",
        dtype_backend="pyarrow",
    )
    print("读入 meta dtypes:")
    print(ddf.dtypes)
    raw = ddf.compute()
    print(raw)
    last = raw.iloc[-1]
    print("最后一行:", last.to_dict())
    print(f"qty 字面量 '<NA>'? {last['qty'] == '<NA>'}")
    print()

    # ---------- Step2: 再转成指定类型 ----------
    print("=== Step2: map_partitions 转成 TARGET_DTYPES ===")
    print("目标:", TARGET_DTYPES)
    typed = ddf.map_partitions(
        cast_to_target_types,
        target_dtypes=TARGET_DTYPES,  # 作为参数传给函数
        meta=TARGET_DTYPES,  # 同时告诉 Dask 输出 schema
    )
    result = typed.compute()
    print(result)
    print(result.dtypes)
    print()
    print("注意最后一行:")
    print(f"  price={result.iloc[-1]['price']!r}  # 来自截断字段 '3.89'，未必是真值")
    print(f"  qty={result.iloc[-1]['qty']!r}      # '<NA>' 清成真正空值后可转 int")
    print()

    # ---------- 对照：完整行正常转换 ----------
    print("=== 对照: 只有完整行时 ===")
    ok_path = OUTPUT_DIR / "full_width_only.fwf"
    ok_path.write_text(
        "\n".join(
            [
                "apple" + "91268829.0" + "00000010",
                "banan" + "0000001.50" + "00000020",
            ]
        ),
        encoding="utf-8",
        newline="\n",
    )
    ddf_ok = dd.read_fwf(
        ok_path,
        widths=WIDTHS,
        names=NAMES,
        header=None,
        dtype=str,
        engine="pyarrow",
        dtype_backend="pyarrow",
    )
    ok = ddf_ok.map_partitions(
        cast_to_target_types,
        target_dtypes=TARGET_DTYPES,
        meta=TARGET_DTYPES,
    ).compute()
    print(ok)
    print(ok.dtypes)
    print("price[0] =", ok.iloc[0]["price"])


if __name__ == "__main__":
    demo()
