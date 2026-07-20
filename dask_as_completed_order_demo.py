"""
验证 distributed.as_completed：按「完成时刻」通知，不是按 futures 列表下标顺序。

做法：submit 若干任务，故意让 index 大的先睡完（先完成），
打印 as_completed 的产出顺序，并与列表下标顺序对比。

Windows 下必须 if __name__ == "__main__"。
"""

from __future__ import annotations

import time

from dask.distributed import Client, LocalCluster, as_completed


def slow_task(index: int, sleep_s: float) -> dict:
    """睡完返回自己的下标与耗时。"""
    t0 = time.perf_counter()
    time.sleep(sleep_s)
    return {
        "index": index,
        "sleep_s": sleep_s,
        "finished_at": time.perf_counter() - t0,
    }


def main() -> None:
    # index=0 睡最久，index=4 睡最短 → 5 个 worker 并行时，完成序应为 4,3,2,1,0
    sleeps = [0.50, 0.40, 0.30, 0.20, 0.10]
    list_order = list(range(len(sleeps)))
    expected_completion_order = sorted(list_order, key=lambda i: sleeps[i])

    # worker 数 = 任务数，避免短任务排在长任务后面才开工，从而打乱「纯 sleep」完成序
    cluster = LocalCluster(
        n_workers=len(sleeps),
        threads_per_worker=1,
        dashboard_address=":8788",
    )
    client = Client(cluster)
    print(f"Dashboard: {client.dashboard_link}")
    print(f"workers={len(sleeps)}（与任务数相同，保证同时开工）")
    print(f"任务 sleep 秒: {sleeps}")
    print(f"futures 列表下标顺序: {list_order}")
    print(f"按 sleep 推断的完成顺序: {expected_completion_order}")
    print()

    futures = [
        client.submit(slow_task, i, sleeps[i], key=f"slow-{i}")
        for i in range(len(sleeps))
    ]
    print("result_futures 下标顺序:", [f.key for f in futures])

    got_order: list[int] = []
    print("\nas_completed 通知顺序:")
    t0 = time.perf_counter()
    for fut in as_completed(futures):
        result = fut.result()
        elapsed = time.perf_counter() - t0
        got_order.append(result["index"])
        print(
            f"  t=+{elapsed:.3f}s  通知到 index={result['index']}  "
            f"(该任务 sleep={result['sleep_s']}s)"
        )

    print()
    print(f"列表下标顺序:     {list_order}")
    print(f"as_completed 顺序: {got_order}")
    print(f"期望完成顺序:     {expected_completion_order}")

    if got_order == expected_completion_order:
        print("\n结论: as_completed 按完成先后通知（不是按列表下标）。")
    elif got_order == list_order:
        print("\n结论: 异常 — 看起来像按列表下标（不符合预期）。")
    else:
        print("\n结论: 顺序与严格 sleep 排序不完全一致（调度抖动），"
              f"但仍不是列表序: {got_order != list_order}")

    # 对比：若按列表下标逐个 result()，会卡在最慢的第一个上
    print("\n--- 对比：按 futures[i] 顺序 result() ---")
    futures2 = [
        client.submit(slow_task, i, sleeps[i], key=f"seq-{i}")
        for i in range(len(sleeps))
    ]
    t0 = time.perf_counter()
    seq_order: list[int] = []
    for i, fut in enumerate(futures2):
        result = fut.result()
        elapsed = time.perf_counter() - t0
        seq_order.append(result["index"])
        print(f"  t=+{elapsed:.3f}s  futures[{i}].result() → index={result['index']}")
    print(f"按列表 result 顺序: {seq_order}  ← 总是下标序，且被慢任务挡住")

    client.close()
    cluster.close()


if __name__ == "__main__":
    main()
