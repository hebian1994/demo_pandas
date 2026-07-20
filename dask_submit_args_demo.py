"""
DEMO：client.submit 入参类型对比

对比三种入参:
  1. 普通字符串 str
  2. 普通列表 / 数组 list
  3. Future（以及 list[Future]）

要点:
  - 普通 Python 对象：序列化后直接传给 worker，函数里拿到的就是原值
  - Future：submit 会等它完成，把 Future 的【结果】传给函数（自动解包依赖）
  - list[Future]：通常也会先 gather 成结果列表，再传给函数
  - 若只想传「路径字符串」而不依赖某个 Future，用普通 str，不要误传 Future

Windows: 放在 if __name__ == "__main__" 里。
"""

from __future__ import annotations

import time

from dask.distributed import Client, Future, LocalCluster, get_worker


def show_arg(tag: str, value) -> dict:
    """在 worker 上打印入参真实类型与内容，返回摘要供 client 侧核对。"""
    worker = get_worker().address
    typ = type(value).__name__
    # 故意不打印超大对象；list 只展示前几项
    if isinstance(value, list):
        preview = value[:5]
        extra = f"len={len(value)}"
    else:
        preview = value
        extra = ""
    print(f"[worker {worker}] {tag}: type={typ}, value={preview!r} {extra}")
    return {
        "tag": tag,
        "worker": worker,
        "type": typ,
        "preview": repr(preview),
        "is_future": isinstance(value, Future),
    }


def add_one(x: int) -> int:
    time.sleep(0.3)
    return x + 1


def join_paths(paths: list[str], dest: str) -> str:
    """模拟合并：入参应是路径字符串列表，不是 Future。"""
    return f"{dest} <= {paths}"


def consume_maybe_future_list(items, dest: str) -> dict:
    """看 list 里究竟是 str 还是已经解包后的结果。"""
    kinds = [type(x).__name__ for x in items]
    return {"dest": dest, "n": len(items), "elem_types": kinds, "items": items}


def demo() -> None:
    cluster = LocalCluster(n_workers=2, threads_per_worker=1, processes=True)
    client = Client(cluster)
    print(f"Dashboard: {client.dashboard_link}\n")

    # ---------- 1) 普通字符串 ----------
    print("=== 1) 入参 = 普通 str ===")
    fut_str = client.submit(show_arg, "plain_str", "hello.csv")
    print("client 侧返回:", fut_str.result())
    print("→ worker 收到的就是 str，没有依赖等待\n")

    # ---------- 2) 普通列表 ----------
    print("=== 2) 入参 = 普通 list[str] ===")
    paths = ["part-0.csv", "part-1.csv", "part-2.csv"]
    fut_list = client.submit(show_arg, "plain_list", paths)
    print("client 侧返回:", fut_list.result())
    print("→ worker 收到的就是 list[str]，整体被序列化送过去\n")

    fut_join = client.submit(join_paths, paths, "out.csv")
    print("join_paths 结果:", fut_join.result())
    print()

    # ---------- 3) 单个 Future ----------
    print("=== 3) 入参 = 单个 Future ===")
    fut_num = client.submit(add_one, 10)  # 结果将是 11
    print(f"client 侧 fut_num 本身是 Future: {fut_num!r}, type={type(fut_num).__name__}")

    # 把 Future 当作 submit 的参数 → worker 里拿到的是 11（结果），不是 Future 对象
    fut_from_fut = client.submit(show_arg, "from_future", fut_num)
    print("client 侧返回:", fut_from_fut.result())
    print("→ submit 会先等 fut_num 完成，再把结果 11 传给 show_arg")
    print("→ 函数参数类型是 int，不是 Future\n")

    # ---------- 4) list[Future] ----------
    print("=== 4) 入参 = list[Future] ===")
    write_futs = [
        client.submit(lambda i: (f"part-{i}.csv", i * 10), i)
        for i in range(3)
    ]
    # 不 gather：直接把 Future 列表传给下一个 submit（生产里 merge 常用这种）
    fut_merge = client.submit(consume_maybe_future_list, write_futs, "final.csv")
    print("client 侧返回:", fut_merge.result())
    print("→ list 里的 Future 会被解析成各自的结果 [(path, nrows), ...]")
    print("→ 函数里看到的是普通 tuple/str，不是 Future\n")

    # ---------- 5) 对比：先 gather 再 submit vs 直接传 Future ----------
    print("=== 5) gather 后再 submit vs 直接传 Future ===")
    gathered = client.gather(write_futs)  # 回到 client 进程的普通 Python 对象
    print("gather 后在 client 上:", gathered, type(gathered[0]))

    fut_a = client.submit(consume_maybe_future_list, gathered, "via_gather.csv")
    fut_b = client.submit(consume_maybe_future_list, write_futs, "via_future.csv")
    print("via gather :", fut_a.result())
    print("via futures:", fut_b.result())
    print("→ 最终函数入参内容可以一样；差别是：")
    print("  gather: 结果先拉回 client，再序列化发给 worker（多一次往返，大对象很亏）")
    print("  Future: 依赖在集群内传递/等待，client 不碰中间大数据\n")

    # ---------- 6) 常见误区 ----------
    print("=== 6) 误区：想传路径字符串，却传了 Future ===")
    path_fut = client.submit(lambda: "only_path.csv")
    # 正确：要路径字符串就 .result() 或 gather；或让下游函数接收解包后的 str
    ok = client.submit(show_arg, "expect_str", path_fut)
    print(ok.result())
    print("→ 因为传的是 Future，worker 实际收到的是结果 'only_path.csv'（仍然是 str）")
    print("→ 所以「Future[str]」对函数来说往往就是 str；真正要小心的是语义：")
    print("   你是在表达依赖关系，还是只想塞一个常量字符串。\n")

    print("=== 小结 ===")
    print("| 入参            | worker 里拿到的     | 典型用途              |")
    print("|---------------|------------------|---------------------|")
    print("| str / int / …  | 原值               | 常量配置、路径模板      |")
    print("| list/dict      | 原值（序列化拷贝）    | 小元数据、路径列表      |")
    print("| Future         | Future.result()    | 任务依赖、流水线        |")
    print("| list[Future]   | list[各结果]        | map 写出后再 submit 合并 |")

    client.close()
    cluster.close()


if __name__ == "__main__":
    demo()
