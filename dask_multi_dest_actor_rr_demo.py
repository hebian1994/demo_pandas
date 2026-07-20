"""
DEMO：按 Worker 列表轮询创建 Actor + client.map 计算结果 Future 再追加

流程（贴近真实用法）:
  1. 原始分区 DataFrame 列表
  2. client.map(process_partition, ...) → 得到 result Futures（不是 DF）
  3. client.submit(append_to_actor, part_future, writer)
     → submit 会把 Future 依赖解析成 DF 再调用；数据再送到 Actor 所在 worker 写盘
  4. Actor 按 worker 轮询创建，相邻目的地尽量不同进程

Windows 下必须 if __name__ == "__main__"。
"""

from __future__ import annotations

import time
from itertools import cycle
from pathlib import Path

import numpy as np
import pandas as pd
from dask.distributed import Client, Future, LocalCluster, wait

OUTPUT_DIR = Path(__file__).parent / "output" / "multi_dest_actor_rr_demo"


class ActorRoundRobin:
    """按 worker 地址轮询创建 Actor。"""

    def __init__(self, client: Client) -> None:
        workers = sorted(client.scheduler_info()["workers"])
        if not workers:
            raise RuntimeError("集群无 worker")
        self._client = client
        self._workers = workers
        self._rr = cycle(workers)
        print(f"Worker 轮询顺序 ({len(workers)}):")
        for i, w in enumerate(workers):
            print(f"  [{i}] {w}")

    def create(self, cls, *args, **kwargs):
        w = next(self._rr)
        actor = self._client.submit(
            cls,
            *args,
            **kwargs,
            actor=True,
            workers=w,
            allow_other_workers=False,
        ).result()
        return actor, w


class CsvAppendActor:
    """单目的地写入器：同一 Actor 内串行 append；每次 append 单独 open/close。"""

    def __init__(self, dest: str) -> None:
        Path(dest).parent.mkdir(parents=True, exist_ok=True)
        self.dest = dest
        self.name = Path(dest).name
        self.first = True
        self.total = 0
        self.t0 = time.perf_counter()

    def where(self) -> str:
        from distributed.worker import get_worker

        return get_worker().address

    def append(self, pdf: pd.DataFrame) -> int:
        t = time.perf_counter() - self.t0
        mode = "新建+header" if self.first else "追加"
        # 首次 "w" 覆盖创建；之后 "a" 追加。每次调用单独打开/关闭。
        file_mode = "w" if self.first else "a"
        with open(self.dest, file_mode, newline="", encoding="utf-8") as f:
            pdf.to_csv(f, index=False, header=self.first)
        self.first = False
        self.total += len(pdf)
        time.sleep(0.10)
        print(f"  t=+{t:.3f}s  [{self.name}] rows={len(pdf)} {mode} total={self.total}")
        return self.total

    def finish(self) -> tuple[str, int]:
        return self.dest, self.total


def process_partition(pdf: pd.DataFrame, tag: str, sleep_s: float) -> pd.DataFrame:
    """分区业务计算：只返回 DataFrame，由 client.map 得到 Future。"""
    time.sleep(sleep_s)
    out = pdf.copy()
    out["amount"] = out["qty"] * out["price"]
    out["dest_tag"] = tag
    return out


def append_to_actor(pdf: pd.DataFrame, writer: CsvAppendActor) -> int:
    """
    写出包装：入参在运行时已是 DataFrame。
    调用方应传入「计算完成的 Future」；submit 会先等 Future 完成再注入 pdf。
    """
    writer.append(pdf).result()
    return len(pdf)


def main() -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    n_workers = 3
    dest_names = ["A.csv", "B.csv", "C.csv", "D.csv"]
    dest_paths = {name: OUTPUT_DIR / name for name in dest_names}
    for p in dest_paths.values():
        if p.exists():
            p.unlink()

    rng = np.random.default_rng(0)
    # 原始输入分区（DataFrame）；计算后才会变成 Future
    input_parts: dict[str, list[pd.DataFrame]] = {}
    sleeps: dict[str, list[float]] = {}
    base = 0
    for di, name in enumerate(dest_names):
        input_parts[name] = []
        sleeps[name] = []
        for pi in range(2):
            n = 80
            input_parts[name].append(
                pd.DataFrame(
                    {
                        "order_id": np.arange(base, base + n),
                        "qty": rng.integers(1, 5, n),
                        "price": rng.random(n) * 10,
                    }
                )
            )
            sleeps[name].append(0.05 + 0.04 * pi + 0.01 * di)
            base += n

    cluster = LocalCluster(
        n_workers=n_workers,
        threads_per_worker=1,
        dashboard_address=":8794",
    )
    client = Client(cluster)
    print(f"\nDashboard: {client.dashboard_link}")
    print(f"目的地: {dest_names}  workers={n_workers}\n")

    rr = ActorRoundRobin(client)

    writers: dict[str, CsvAppendActor] = {}
    planned_workers: dict[str, str] = {}
    print("\n创建 Actor（轮询）:")
    for name in dest_names:
        actor, planned = rr.create(CsvAppendActor, str(dest_paths[name]))
        actual = actor.where().result()
        writers[name] = actor
        planned_workers[name] = planned
        ok = "OK" if actual == planned else "MISMATCH"
        print(f"  {name}: planned={planned}")
        print(f"         actual ={actual}  [{ok}]")

    seq = [planned_workers[n] for n in dest_names]
    print("\n落点序列:", seq)
    for i in range(len(seq) - 1):
        if seq[i] == seq[i + 1]:
            print(f"  警告: {dest_names[i]} 与 {dest_names[i+1]} 落在同一 worker")
        else:
            print(f"  {dest_names[i]} → {dest_names[i+1]} 不同 worker（轮询生效）")
    if seq[0] == seq[3]:
        print("  A 与 D 同 worker：目的地多于 worker 时的预期回绕")

    # --- 1) client.map 分区计算 → Futures（不是 DataFrame）---
    print("\nclient.map 计算分区 → Futures:")
    part_futures: dict[str, list[Future]] = {}
    for name in dest_names:
        tag = name[0]
        n = len(input_parts[name])
        part_futures[name] = client.map(
            process_partition,
            input_parts[name],
            [tag] * n,
            sleeps[name],
            key=[f"part-{tag}-{i}" for i in range(n)],
        )
        print(f"  {name}: {[f.key for f in part_futures[name]]}  "
              f"(type={type(part_futures[name][0]).__name__})")

    # --- 2) 每个计算 Future 再 submit 给对应目的地 Actor ---
    # submit(append_to_actor, part_fut, writer)：part_fut 是依赖，
    # 调度器等其完成后把「结果 DF」传给 append_to_actor，再送到 Actor 所在 worker。
    print("\nsubmit(append_to_actor, part_future, writer):")
    write_futures: list[Future] = []
    for name in dest_names:
        tag = name[0]
        for i, part_fut in enumerate(part_futures[name]):
            assert isinstance(part_fut, Future)
            write_futures.append(
                client.submit(
                    append_to_actor,
                    part_fut,          # ← Future，不是 pdf
                    writers[name],
                    key=f"write-{tag}-{i}",
                )
            )

    t0 = time.perf_counter()
    wait(write_futures)
    for f in write_futures:
        f.result()

    results = {name: writers[name].finish().result() for name in dest_names}
    elapsed = time.perf_counter() - t0
    print(f"\n总耗时: {elapsed:.2f}s")
    for name, (path, nrows) in results.items():
        df = pd.read_csv(path)
        assert len(df) == nrows == 160
        assert set(df["dest_tag"]) == {name[0]}
        print(f"  {name}: rows={nrows} worker={planned_workers[name]}")

    print("\n校验通过。")
    client.close()
    cluster.close()


if __name__ == "__main__":
    main()
