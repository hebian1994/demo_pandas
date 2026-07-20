"""
监控 DF 每一列 dtype：非 PyArrow 则转成对应 Arrow 类型。
可在多层转换之间插入 ensure_pyarrow(df) 做检查点；
也可用 @with_pyarrow 装饰业务函数，自动转换输入/输出中的 DF/Series（含容器）。
"""

from __future__ import annotations

from functools import wraps
from typing import Any, Callable, TypeVar

import numpy as np
import pandas as pd
import pyarrow as pa

F = TypeVar("F", bound=Callable[..., Any])


def is_pyarrow_backed(dtype) -> bool:
    """该 dtype 是否已经是 PyArrow 后端。"""
    if isinstance(dtype, pd.ArrowDtype):
        return True
    # pandas StringDtype(storage="pyarrow") 也算 Arrow 后端
    storage = getattr(dtype, "storage", None)
    return storage == "pyarrow"


def numpy_or_pandas_to_arrow(dtype) -> pa.DataType:
    """把常见 pandas/NumPy dtype 映射到 PyArrow 类型。"""
    # 已经是 ArrowDtype
    if isinstance(dtype, pd.ArrowDtype):
        return dtype.pyarrow_dtype

    # pandas StringDtype
    if isinstance(dtype, pd.StringDtype):
        return pa.string()

    # pandas nullable: Int64 / Float64 / boolean ...
    name = getattr(dtype, "name", str(dtype))
    nullable_map = {
        "Int8": pa.int8(),
        "Int16": pa.int16(),
        "Int32": pa.int32(),
        "Int64": pa.int64(),
        "UInt8": pa.uint8(),
        "UInt16": pa.uint16(),
        "UInt32": pa.uint32(),
        "UInt64": pa.uint64(),
        "Float32": pa.float32(),
        "Float64": pa.float64(),  # = double，不要用 float(=float32)
        "boolean": pa.bool_(),
        "bool": pa.bool_(),
        "string": pa.string(),
    }
    if name in nullable_map:
        return nullable_map[name]

    # NumPy dtype
    np_dtype = np.dtype(dtype)
    kind_unit = (np_dtype.kind, np_dtype.name)

    if np_dtype.kind == "f":
        # float64 -> double；float32 -> float；不要写错成 pa.float()
        return pa.from_numpy_dtype(np_dtype)
    if np_dtype.kind in "iu":
        return pa.from_numpy_dtype(np_dtype)
    if np_dtype.kind == "b":
        return pa.bool_()
    if np_dtype.kind == "M":  # datetime64[...]
        unit = np.datetime_data(np_dtype)[0]
        return pa.timestamp(unit)
    if np_dtype.kind == "m":  # timedelta64[...]
        unit = np.datetime_data(np_dtype)[0]
        return pa.duration(unit)
    if np_dtype.kind == "O":
        # object 默认当字符串；若是混合类型需业务侧自行处理
        return pa.string()

    raise TypeError(f"无法映射到 PyArrow: {dtype!r} ({kind_unit})")


def _prepare_series_for_arrow(
    series: pd.Series,
    dtype,
    *,
    target_hint: pd.ArrowDtype,
) -> tuple[pd.Series, pa.DataType, str]:
    """
    将一列准备成可 astype 到 Arrow 的形态。
    转换失败一律抛错（errors='raise'），不吞掉任何坏值。
    """
    if dtype != object:
        return series, target_hint.pyarrow_dtype, f"convert -> {target_hint}"

    inferred = pd.api.types.infer_dtype(series, skipna=True)

    if inferred in {"floating", "mixed-integer-float"}:
        series = pd.to_numeric(series, errors="raise")
        arrow_type = pa.float64()
    elif inferred in {"integer", "mixed-integer"}:
        series = pd.to_numeric(series, errors="raise")
        # 有缺省时用可空 Int64，再进 Arrow；非法值已在 to_numeric 阶段报错
        series = series.astype("Int64")
        arrow_type = pa.int64()
    elif inferred == "boolean":
        series = series.astype("boolean")
        arrow_type = pa.bool_()
    elif inferred in {"datetime", "date"}:
        series = pd.to_datetime(series, errors="raise")
        arrow_type = pa.timestamp("ns")
    elif inferred in {"string", "unicode", "bytes", "empty"}:
        series = series.astype("string")
        arrow_type = pa.string()
    else:
        # mixed / complex / categorical-like 等：不猜测、不静默转 string
        raise TypeError(
            f"object 列推断类型为 {inferred!r}，无法安全映射到 PyArrow，请先在上游清洗"
        )

    target = pd.ArrowDtype(arrow_type)
    return series, arrow_type, f"object({inferred}) -> {target}"


def _convert_one_series(
    series: pd.Series,
    *,
    col_label: str = "",
    prefix: str = "",
) -> tuple[pd.Series, dict[str, Any]]:
    """转换单列，返回 (新 series, 报告行)。"""
    dtype = series.dtype
    already = is_pyarrow_backed(dtype)
    name = col_label or (str(series.name) if series.name is not None else "<series>")

    if already:
        target = dtype if isinstance(dtype, pd.ArrowDtype) else f"(StringDtype storage={dtype.storage})"
        action = "keep"
        out = series
        if isinstance(dtype, pd.StringDtype):
            try:
                out = series.astype(pd.ArrowDtype(pa.string()))
            except Exception as e:
                raise TypeError(
                    f"{prefix}列 {name!r} 无法转为 string[pyarrow] ArrowDtype: {e}"
                ) from e
            target = out.dtype
            action = "normalize string[pyarrow] -> string[pyarrow] ArrowDtype"
        row = {
            "column": name,
            "before": str(dtype),
            "pyarrow?": already,
            "after": str(out.dtype),
            "action": action,
        }
        return out, row

    arrow_type = numpy_or_pandas_to_arrow(dtype)
    target = pd.ArrowDtype(arrow_type)
    action = f"convert -> {target}"
    try:
        prepared, arrow_type, action = _prepare_series_for_arrow(
            series, dtype, target_hint=pd.ArrowDtype(arrow_type)
        )
        target = pd.ArrowDtype(arrow_type)
        out = prepared.astype(target)
    except Exception as e:
        raise TypeError(
            f"{prefix}列 {name!r} 无法转为 PyArrow (before={dtype}): {e}"
        ) from e

    row = {
        "column": name,
        "before": str(dtype),
        "pyarrow?": already,
        "after": str(out.dtype),
        "action": action,
    }
    return out, row


def ensure_pyarrow_series(
    series: pd.Series,
    *,
    label: str = "",
    quiet: bool = True,
) -> pd.Series:
    """将 Series 转为 PyArrow 后端。"""
    prefix = f"[{label}] " if label else ""
    out, row = _convert_one_series(series, prefix=prefix)
    if not quiet:
        print(f"\n{prefix}=== Series dtype 检查 ===")
        print(pd.DataFrame([row]).to_string(index=False))
    return out


def ensure_pyarrow(
    df: pd.DataFrame,
    *,
    label: str = "",
    convert: bool = True,
    quiet: bool = False,
) -> pd.DataFrame:
    """
    检查 DF 每一列是否为 PyArrow 类型；不是则转成对应 Arrow 类型。

    Parameters
    ----------
    label : 检查点名称，方便在多层转换里定位
    convert : True=转换并返回新 DF；False=只打印报告不改动
    quiet : True=不打印报告（给装饰器/热路径用）
    """
    prefix = f"[{label}] " if label else ""
    if not quiet:
        print(f"\n{prefix}=== dtype 检查 ===")

    if not convert:
        rows = []
        for col in df.columns:
            dtype = df[col].dtype
            already = is_pyarrow_backed(dtype)
            if already:
                target = dtype
                would = "ok"
            else:
                target = pd.ArrowDtype(numpy_or_pandas_to_arrow(dtype))
                would = f"would -> {target}"
            rows.append(
                {
                    "column": col,
                    "before": str(dtype),
                    "pyarrow?": already,
                    "after": "-",
                    "action": would,
                }
            )
        if not quiet:
            print(pd.DataFrame(rows).to_string(index=False))
        return df

    out = df.copy()
    rows = []
    for col in df.columns:
        converted, row = _convert_one_series(df[col], col_label=str(col), prefix=prefix)
        out[col] = converted
        rows.append(row)

    if not quiet:
        print(pd.DataFrame(rows).to_string(index=False))
    return out


def convert_pyarrow_tree(obj: Any, *, label: str = "") -> Any:
    """
    递归转换输入/输出树：
    - DataFrame / Series → PyArrow
    - tuple / list / dict → 逐元素处理，结构不变
    - 其它类型原样返回
    """
    if isinstance(obj, pd.DataFrame):
        return ensure_pyarrow(obj, label=label, quiet=True)
    if isinstance(obj, pd.Series):
        return ensure_pyarrow_series(obj, label=label, quiet=True)
    if isinstance(obj, tuple):
        return tuple(convert_pyarrow_tree(x, label=label) for x in obj)
    if isinstance(obj, list):
        return [convert_pyarrow_tree(x, label=label) for x in obj]
    if isinstance(obj, dict):
        return {k: convert_pyarrow_tree(v, label=label) for k, v in obj.items()}
    return obj


def with_pyarrow(fn: F) -> F:
    """
    切面：调用前转换所有参数，调用后转换返回值。
    支持 DF / Series，以及它们出现在 tuple / list / dict 里的组合。
    """

    @wraps(fn)
    def wrapper(*args, **kwargs):
        args = tuple(convert_pyarrow_tree(a) for a in args)
        kwargs = {k: convert_pyarrow_tree(v) for k, v in kwargs.items()}
        return convert_pyarrow_tree(fn(*args, **kwargs))

    return wrapper  # type: ignore[return-value]


def demo() -> None:
    # 模拟「进线」后 dtype 杂乱的 DF
    df = pd.DataFrame(
        {
            "price": [91268829.02, 1.5],  # float64
            "qty": [10, 20],  # int64
            "flag": [True, False],  # bool
            "name": ["a", "b"],  # object
            "score": pd.Series([1.1, None], dtype="Float64"),  # nullable
            "cnt": pd.Series([1, None], dtype="Int64"),
            "ts": pd.to_datetime(["2024-01-01", "2024-01-02"]),
            "already": pd.Series([1.0, 2.0], dtype="double[pyarrow]"),  # 已是 Arrow
        }
    )

    df = ensure_pyarrow(df, label="step0_raw")

    # 模拟中间某层又把列搞回非 Arrow（例如 map + f-string、合并等）
    df["price"] = df["price"].astype("float64")  # 退化回 numpy
    df["name"] = df["name"].map(lambda x: f"{x}_x")  # 可能变 object/string
    df["ratio"] = df["qty"].astype("float64") / 100.0  # 新列 float64

    df = ensure_pyarrow(df, label="step1_after_transform")

    print("\n最终 dtypes:")
    print(df.dtypes)
    print("\n数据预览:")
    print(df)

    # 坏数据必须报错，不能 coerce 吃掉
    print("\n=== 坏数据应直接报错 ===")
    cases = {
        "mixed_integer": pd.DataFrame({"amount": [1, "x"]}),
        "mixed": pd.DataFrame({"amount": [1.2, "oops"]}),
    }
    for name, bad in cases.items():
        try:
            ensure_pyarrow(bad, label=name)
        except TypeError as e:
            print(f"[{name}] 按预期抛错: {e}")

    # 装饰器：输入/输出都可能是多种形态
    print("\n=== @with_pyarrow 多类型输入输出 ===")

    @with_pyarrow
    def process_partition(
        pdf: pd.DataFrame,
        extra: pd.Series,
        meta: dict,
    ) -> tuple[pd.DataFrame, pd.Series, dict]:
        out = pdf.copy()
        out["tag"] = "KKK"  # 易变成 object
        s = extra.map(lambda x: f"{x}_x")
        return out, s, {"rows": len(out), **meta}

    raw = pd.DataFrame({"x": [1, 2], "name": ["a", "b"]})
    extra = pd.Series(["p", "q"], name="extra")
    out_df, out_s, out_meta = process_partition(raw, extra, {"src": "demo"})

    print("out_df.dtypes:\n", out_df.dtypes, sep="")
    print("out_s.dtype:", out_s.dtype)
    print("out_meta:", out_meta)
    assert is_pyarrow_backed(out_df["tag"].dtype)
    assert is_pyarrow_backed(out_s.dtype)


if __name__ == "__main__":
    demo()
