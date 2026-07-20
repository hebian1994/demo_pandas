"""
从 engine='c' + converters=strip 迁到 engine='pyarrow' 的 DEMO。

旧写法（c 引擎）：
    converters={i: (lambda x: x.strip()) for i in columns}

新写法：
    pyarrow 不支持 converters / skipinitialspace
    → 先 read_table(..., engine='pyarrow', dtype_backend='pyarrow')
    → 再对各列 .str.strip()
"""

from __future__ import annotations

from pathlib import Path

import dask.dataframe as dd
import pandas as pd

OUTPUT_DIR = Path(__file__).parent / "output"
TSV_PATH = OUTPUT_DIR / "tab_tilde_quote_strip.tsv"

NAMES = ["name", "price", "note"]


def make_tsv() -> Path:
    OUTPUT_DIR.mkdir(exist_ok=True)
    # 字段两侧故意留空格；第 2 行 note 内含真实 tab（在 ~ ~ 里）
    lines = [
        "  apple  \t  91268829.02  \t~  plain note  ~",
        " banana \t 1.50 \t~has\tembedded\ttab~",
        "cherry\t  3.891  \t~quoted with spaces~",
    ]
    TSV_PATH.write_text("\n".join(lines), encoding="utf-8", newline="\n")
    return TSV_PATH


def strip_all_string_cols(pdf: pd.DataFrame) -> pd.DataFrame:
    """替代 converters={col: lambda x: x.strip()}。"""
    out = pdf.copy()
    for col in out.columns:
        # 数值列若已被推断成 double，先转 string 再 strip 再还原——
        # 更稳妥的做法是读入时全按 str，再 strip，再转目标类型。
        if pd.api.types.is_string_dtype(out[col]) or out[col].dtype == object:
            out[col] = out[col].astype("string[pyarrow]").str.strip()
    return out


def demo_old_style_c_engine(path: Path) -> None:
    print("=== 旧: engine='c' + converters strip ===")
    converters = {name: (lambda x: x.strip()) for name in NAMES}
    # pandas 对照（dask 的 c 引擎同样支持 converters）
    df = pd.read_table(
        path,
        sep="\t",
        quotechar="~",
        names=NAMES,
        header=None,
        engine="c",
        converters=converters,
    )
    print(df.map(repr))
    print()


def demo_new_style_pyarrow(path: Path) -> None:
    print("=== 新: engine='pyarrow'（不支持 converters）→ 读后再 strip ===")

    # 推荐：先全部当 str 读，strip 后再转类型（避免数值先推断再 strip 的麻烦）
    ddf = dd.read_table(
        path,
        sep="\t",
        quotechar="~",
        names=NAMES,
        header=None,
        dtype=str,  # 等价于「先当字符串进来」
        engine="pyarrow",
        dtype_backend="pyarrow",
    )
    print("读入后（未 strip）:")
    raw = ddf.compute()
    print(raw.map(repr))
    print()

    # 替代 converters：对所有列 strip
    ddf_stripped = ddf.map_partitions(
        lambda pdf: pdf.assign(**{c: pdf[c].astype("string[pyarrow]").str.strip() for c in pdf.columns}),
        meta=ddf._meta,
    )
    print("strip 之后:")
    stripped = ddf_stripped.compute()
    print(stripped.map(repr))
    print()

    # 再转成目标类型（需要的话）
    target = {
        "name": "string[pyarrow]",
        "price": "float64[pyarrow]",
        "note": "string[pyarrow]",
    }

    def cast(pdf: pd.DataFrame) -> pd.DataFrame:
        out = pdf.copy()
        out["price"] = pd.to_numeric(out["price"], errors="raise")
        return out.astype(target)

    typed = ddf_stripped.map_partitions(cast, meta=target).compute()
    print("转类型之后:")
    print(typed)
    print(typed.dtypes)
    print()
    print("第 2 行 note 仍含内部 tab:", repr(typed.iloc[1]["note"]))


def demo() -> None:
    path = make_tsv()
    print(f"文件: {path}")
    print(f"bytes: {path.read_bytes()!r}")
    print()
    print("结论: pyarrow 引擎传入 converters 会直接报错:")
    print("  ValueError: The 'converters' option is not supported with the 'pyarrow' engine")
    print()

    demo_old_style_c_engine(path)
    demo_new_style_pyarrow(path)


if __name__ == "__main__":
    demo()
