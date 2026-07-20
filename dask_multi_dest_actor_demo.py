"""
DEMO：多目的地真正并行写出 —— 每目的地一个 Actor（跑在 worker 上）

回答「为什么用 ThreadPoolExecutor / 能否把监听 submit 到集群」：

1) ThreadPoolExecutor（client 侧）
   - 实现简单；fut.result() / 写盘多为 I/O，线程可重叠等待
   - 受 GIL 影响主要在纯 Python CPU；写 CSV 通常仍够用
   - 数据会经 client（或从各 worker 拉到 client 再写）

2) 不能直接 client.submit(drain, futures_a)
   - 传入的 Future 会在 drain 启动前被全部 resolve 成具体 DF
   - drain 拿到的是 list[DataFrame]，as_completed 流式语义没了
   - 等于「全部算完再一次性写」，不是来一个写一个

3) 在 worker 上用 Future(key)+as_completed 自行监听
   - 理论上要传 key 字符串再拼 Future，并 secede() 避免占死 worker 槽
   - 实践里易卡住/难维护，不推荐作为主方案

4) 推荐集群内并行：每个目的地一个 Actor
   - Actor 住在某个 worker 上，append 在该进程执行（多进程，真并行）
   - 不同目的地 = 不同 Actor，可同时写 A.csv / B.csv
   - 同一 Actor 内方法调用串行 → 同文件不会写乱
   - 计算任务算完直接 writer.append(df)，无需中央 as_completed

Windows 下必须 if __name__ == "__main__"。
"""

from __future__ import annotations

import time
from pathlib import Path

import numpy as np
import pandas as pd
from dask.distributed import Client, LocalCluster, wait

OUTPUT_DIR = Path(__file__).parent / "output" / "multi_dest_actor_demo"
DEST_A = OUTPUT_DIR / "A.csv"
DEST_B = OUTPUT_DIR / "B.csv"


class CsvAppendActor:
    """单目的地写入器：同一 Actor 内串行 append，保证单文件安全。"""

    def __init__(self, dest: str) -> None:
        Path(dest).parent.mkdir(parents=True, exist_ok=True)
        self.dest = dest
        self.name = Path(dest).name
        self.first = True
        self.total = 0
        self.t0 = time.perf_counter()
        self._f = open(dest, "w", newline="", encoding="utf-8")

    def append(self, pdf: pd.DataFrame) -> int:
        t = time.perf_counter() - self.t0
        mode = "新建+header" if self.first else "追加"
        pdf.to_csv(self._f, index=False, header=self.first)
        self._f.flush()
        self.first = False
        self.total += len(pdf)
        # 故意拖慢写出，便于日志里看到 A/B 时间重叠（真并行）
        time.sleep(0.12)
        print(f"  t=+{t:.3f}s  [{self.name}] rows={len(pdf)} {mode}  total={self.total}")
        return self.total

    def finish(self) -> tuple[str, int]:
        self._f.close()
        return self.dest, self.total


def process_and_append(
    pdf: pd.DataFrame,
    writer: CsvAppendActor,
    tag: str,
    sleep_s: float,
) -> int:
    """算完后直接交给该目的地的 Actor 追加（数据送到 Actor 所在 worker）。"""
    time.sleep(sleep_s)
    out = pdf.copy()
    out["amount"] = out["qty"] * out["price"]
    out["dest_tag"] = tag
    writer.append(out).result()  # Actor 方法返回 Future，.result() 等写完
    return len(out)


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
    sleeps_a = [0.05, 0.18, 0.30]
    sleeps_b = [0.06, 0.15, 0.22, 0.28]

    cluster = LocalCluster(n_workers=4, threads_per_worker=1, dashboard_address=":8793")
    client = Client(cluster)
    print(f"Dashboard: {client.dashboard_link}")
    print("每目的地一个 Actor（集群内进程级并行写出）\n")

    # Actor 会落在某个 worker 进程里；A/B 通常在不同 worker → 真并行写盘
    writer_a = client.submit(CsvAppendActor, str(DEST_A), actor=True).result()
    writer_b = client.submit(CsvAppendActor, str(DEST_B), actor=True).result()

    futures_a = [
        client.submit(
            process_and_append,
            parts_a[i],
            writer_a,
            "A",
            sleeps_a[i],
            key=f"part-A-{i}",
        )
        for i in range(len(parts_a))
    ]
    futures_b = [
        client.submit(
            process_and_append,
            parts_b[i],
            writer_b,
            "B",
            sleeps_b[i],
            key=f"part-B-{i}",
        )
        for i in range(len(parts_b))
    ]

    t0 = time.perf_counter()
    wait(futures_a + futures_b)
    for f in futures_a + futures_b:
        f.result()  # 抛出计算/写入错误

    ra = writer_a.finish().result()
    rb = writer_b.finish().result()
    elapsed = time.perf_counter() - t0

    print(f"\n总耗时: {elapsed:.2f}s")
    print(f"  A: {ra}")
    print(f"  B: {rb}")

    df_a = pd.read_csv(DEST_A)
    df_b = pd.read_csv(DEST_B)
    assert len(df_a) == 300 and set(df_a["dest_tag"]) == {"A"}
    assert len(df_b) == 200 and set(df_b["dest_tag"]) == {"B"}
    print("校验通过。A/B 的 t=+ 应交错 → 两 Actor 并行写。")

    client.close()
    cluster.close()


if __name__ == "__main__":
    main()
