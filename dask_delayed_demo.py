"""
DEMO：dask.delayed 入门

核心概念:
  1. @delayed / delayed(fn) 把普通函数变成「延迟任务」——调用时不执行，只记录依赖图
  2. 返回值是 Delayed 对象，不是真实结果；对它再调用 delayed 函数会继续搭图
  3. .compute() / dask.compute(...) 才真正执行；无依赖的任务可并行
  4. 适合：把已有 Python 函数拼成并行流水线（不必改成 Futures / client.submit）

本文件演示:
  A. 最简用法：延迟 → 计算
  B. 依赖图：扇出并行 + 扇入汇总
  C. 与顺序执行对比耗时
  D. Delayed 对象检查（key / dask 图）
  E. 可选：接 LocalCluster，用同一套 delayed 图在分布式上跑

Windows 下必须 if __name__ == "__main__"。
"""

from __future__ import annotations

import time

import dask
from dask import delayed


# ---------------------------------------------------------------------------
# 普通 Python 函数（故意 sleep，方便看出并行效果）
# ---------------------------------------------------------------------------


def load_chunk(name: str, n: int, sleep_s: float) -> list[int]:
    """模拟读一块数据。"""
    time.sleep(sleep_s)
    print(f"  [load] {name}  n={n}")
    return list(range(n))


def transform(values: list[int], factor: int, sleep_s: float) -> list[int]:
    """模拟变换。"""
    time.sleep(sleep_s)
    out = [v * factor for v in values]
    print(f"  [transform] factor={factor}  len={len(out)}")
    return out


def reduce_sum(parts: list[list[int]], sleep_s: float) -> int:
    """模拟汇总。"""
    time.sleep(sleep_s)
    total = sum(sum(p) for p in parts)
    print(f"  [reduce] total={total}")
    return total


# ---------------------------------------------------------------------------
# A. 最简：装饰器 / 包装函数
# ---------------------------------------------------------------------------


@delayed
def add(x: int, y: int) -> int:
    print(f"  [add] {x} + {y}")
    return x + y


def demo_basic() -> None:
    print("\n=== A. 最简 delayed ===")
    # 调用时不执行，只得到 Delayed
    task = add(10, 20)
    print(f"  type(task) = {type(task).__name__}")
    print(f"  task.key   = {task.key}")
    print("  → 此时 add 尚未运行")

    result = task.compute()
    print(f"  compute() → {result}")

    # 等价写法：delayed(fn)(...)
    mul = delayed(lambda a, b: a * b)
    print(f"  delayed(lambda)(3, 4).compute() → {mul(3, 4).compute()}")


# ---------------------------------------------------------------------------
# B. 依赖图：多路并行 + 汇总
# ---------------------------------------------------------------------------


def build_pipeline():
    """
    构建懒图（不执行）:

        load_a ──► transform(*, 2) ──┐
                                     ├─► reduce_sum
        load_b ──► transform(*, 3) ──┘
    """
    a = delayed(load_chunk)("A", 5, 0.15)
    b = delayed(load_chunk)("B", 4, 0.15)
    ta = delayed(transform)(a, 2, 0.10)
    tb = delayed(transform)(b, 3, 0.10)
    return delayed(reduce_sum)([ta, tb], 0.05)


def demo_graph() -> None:
    print("\n=== B. 依赖图（扇出 + 扇入）===")
    total = build_pipeline()
    print(f"  根节点 key = {total.key}")
    print(f"  图中任务数 ≈ {len(total.dask)}")

    # 可选：需要 graphviz 时取消注释
    # total.visualize(filename="delayed_pipeline", format="png")

    result = total.compute()
    # load_a: 0..4 → *2 → sum=20; load_b: 0..3 → *3 → sum=18; total=38
    assert result == 38
    print(f"  结果 = {result}  (期望 38)")


# ---------------------------------------------------------------------------
# C. 并行 vs 顺序
# ---------------------------------------------------------------------------


def run_sequential() -> int:
    a = load_chunk("A", 5, 0.15)
    b = load_chunk("B", 4, 0.15)
    ta = transform(a, 2, 0.10)
    tb = transform(b, 3, 0.10)
    return reduce_sum([ta, tb], 0.05)


def demo_speedup() -> None:
    print("\n=== C. 顺序 vs delayed 并行 ===")
    print("  顺序执行:")
    t0 = time.perf_counter()
    seq = run_sequential()
    t_seq = time.perf_counter() - t0
    print(f"  顺序耗时: {t_seq:.2f}s  result={seq}")

    print("  delayed + compute (默认线程调度器):")
    t0 = time.perf_counter()
    par = build_pipeline().compute()
    t_par = time.perf_counter() - t0
    print(f"  并行耗时: {t_par:.2f}s  result={par}")
    print(f"  加速比 ≈ {t_seq / t_par:.2f}x  (load/transform 两路可重叠)")


# ---------------------------------------------------------------------------
# D. 一次 compute 多个 Delayed；persist 思路对照
# ---------------------------------------------------------------------------


def demo_multi_compute() -> None:
    print("\n=== D. dask.compute 一次算多个 ===")
    x = delayed(load_chunk)("X", 3, 0.08)
    y = delayed(load_chunk)("Y", 3, 0.08)
    # 两个独立任务；compute 会一起调度，共享同一调度器
    rx, ry = dask.compute(x, y)
    print(f"  X={rx}  Y={ry}")

    # 纯值也可以包进 delayed，作为图的常量叶子
    const = delayed(100)
    print(f"  delayed(100).compute() → {const.compute()}")


# ---------------------------------------------------------------------------
# E. 同一套 Delayed 图放到 LocalCluster（分布式）
# ---------------------------------------------------------------------------


def demo_with_client() -> None:
    print("\n=== E. Delayed + LocalCluster ===")
    from dask.distributed import Client, LocalCluster

    cluster = LocalCluster(
        n_workers=2,
        threads_per_worker=1,
        dashboard_address=":8795",
        silence_logs=40,
    )
    client = Client(cluster)
    print(f"  Dashboard: {client.dashboard_link}")

    # Client 作为当前上下文后，.compute() 会走分布式调度器
    pipeline = build_pipeline()
    t0 = time.perf_counter()
    result = pipeline.compute()
    elapsed = time.perf_counter() - t0
    print(f"  分布式 compute → {result}  耗时 {elapsed:.2f}s")

    # 也可先变成 Future，再异步等待
    fut = client.compute(build_pipeline())
    print(f"  client.compute → Future key={fut.key}")
    print(f"  fut.result() → {fut.result()}")

    client.close()
    cluster.close()


def main() -> None:
    # 本地线程调度器：适合本机 demo；任务会打印到同一终端
    with dask.config.set(scheduler="threads"):
        demo_basic()
        demo_graph()
        demo_speedup()
        demo_multi_compute()

    demo_with_client()
    print("\n全部 DEMO 完成。")
    print(
        "记忆口诀: delayed 搭图 → compute/客户端执行；"
        "无依赖边的节点可并行，有依赖的按边先后。"
    )


if __name__ == "__main__":
    main()
