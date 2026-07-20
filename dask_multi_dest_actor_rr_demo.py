"""
DEMO：按 Worker 列表轮询创建 Actor，让相邻目的地尽量落在不同进程

- ActorRoundRobin：sorted(workers) + cycle，每次 create 钉到下一个 worker
- allow_other_workers=False，禁止漂走
- 打印每个 Actor 实际所在 worker，核对是否轮询散开
- 多目的地 CSV：算完后 append 到对应 Actor（粘性绑定，作业结束才 finish）

Windows 下必须 if __name__ == "__main__"。
"""

from __future__ import annotations

import time
from itertools import cycle
from pathlib import Path

import numpy as np
import pandas as pd
from dask.distributed import Client, LocalCluster, wait

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
    """单目的地写入器：同一 Actor 内串行 append。"""

    def __init__(self, dest: str) -> None:
        Path(dest).parent.mkdir(parents=True, exist_ok=True)
        self.dest = dest
        self.name = Path(dest).name
        self.first = True
        self.total = 0
        self.t0 = time.perf_counter()
        self._f = open(dest, "w", newline="", encoding="utf-8")

    def where(self) -> str:
        """在 Actor 所在进程内取 worker 地址，供核对。"""
        from distributed.worker import get_worker

        return get_worker().address

    def append(self, pdf: pd.DataFrame) -> int:
        t = time.perf_counter() - self.t0
        mode = "新建+header" if self.first else "追加"
        pdf.to_csv(self._f, index=False, header=self.first)
        self._f.flush()
        self.first = False
        self.total += len(pdf)
        time.sleep(0.10)  # 拖慢写出，便于观察跨 worker 并行
        print(f"  t=+{t:.3f}s  [{self.name}] rows={len(pdf)} {mode} total={self.total}")
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
    time.sleep(sleep_s)
    out = pdf.copy()
    out["amount"] = out["qty"] * out["price"]
    out["dest_tag"] = tag
    writer.append(out).result()
    return len(out)


def main() -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    # 4 个目的地、3 个 worker → 轮询后 D 会与 A 同 worker
    n_workers = 3
    dest_names = ["A.csv", "B.csv", "C.csv", "D.csv"]
    dest_paths = {name: OUTPUT_DIR / name for name in dest_names}
    for p in dest_paths.values():
        if p.exists():
            p.unlink()

    rng = np.random.default_rng(0)
    # 每个目的地 2 个分区
    parts: dict[str, list[pd.DataFrame]] = {}
    sleeps: dict[str, list[float]] = {}
    base = 0
    for di, name in enumerate(dest_names):
        parts[name] = []
        sleeps[name] = []
        for pi in range(2):
            n = 80
            parts[name].append(
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

    # --- 轮询创建 Actor，并核对落点 ---
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

    # 相邻两次不应相同（最后 D 与 A 同机是预期：4 dest / 3 worker）
    seq = [planned_workers[n] for n in dest_names]
    print("\n落点序列:", seq)
    for i in range(len(seq) - 1):
        if seq[i] == seq[i + 1]:
            print(f"  警告: {dest_names[i]} 与 {dest_names[i+1]} 落在同一 worker")
        else:
            print(f"  {dest_names[i]} → {dest_names[i+1]} 不同 worker（轮询生效）")
    if seq[0] == seq[3]:
        print("  A 与 D 同 worker：目的地多于 worker 时的预期回绕")

    # --- 提交计算并写到对应 Actor ---
    print("\n写出:")
    futures = []
    for name in dest_names:
        tag = name[0]  # A/B/C/D
        for i, pdf in enumerate(parts[name]):
            futures.append(
                client.submit(
                    process_and_append,
                    pdf,
                    writers[name],
                    tag,
                    sleeps[name][i],
                    key=f"part-{tag}-{i}",
                )
            )

    t0 = time.perf_counter()
    wait(futures)
    for f in futures:
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
