"""
DEMO：两个 Client 同连一个集群，其中一个 client.restart() 时，另一个会怎样？

结论（distributed 默认行为）:
  1. restart 会重启所有 worker，清空集群上的任务/内存数据
  2. 另一个 Client 一般不会因此断开连接（scheduler 还在）
  3. 但该 Client 上尚未完成的 Future 会失败/被取消
  4. restart 之后，另一个 Client 仍可继续 submit 新任务

前置:
  终端 1:  python dask_local_cluster_server.py
  终端 2:  python dask_restart_two_clients_demo.py

Windows 下必须 if __name__ == "__main__"。
"""

from __future__ import annotations

import time
from pathlib import Path

from dask.distributed import Client, Future

ADDRESS_FILE = Path(__file__).resolve().parent / ".dask_scheduler_address"
DEFAULT_SCHEDULER = "tcp://127.0.0.1:8786"


def connect(name: str) -> Client:
    if ADDRESS_FILE.exists():
        addr = ADDRESS_FILE.read_text(encoding="utf-8").strip()
    else:
        addr = DEFAULT_SCHEDULER
    try:
        client = Client(addr)
    except OSError as e:
        raise SystemExit(
            f"无法连接 {addr}: {e}\n"
            "请先运行:  python dask_local_cluster_server.py"
        ) from e
    print(f"  [{name}] 已连接  id={client.id}  scheduler={client.scheduler.address}")
    return client


def slow_add(x: int, sleep_s: float = 8.0) -> int:
    time.sleep(sleep_s)
    return x + 1


def future_status(fut: Future) -> str:
    try:
        return fut.status
    except Exception as e:  # noqa: BLE001
        return f"<status-error {type(e).__name__}: {e}>"


def try_result(fut: Future, label: str) -> None:
    print(f"  [{label}] status={future_status(fut)}")
    try:
        val = fut.result(timeout=5)
        print(f"  [{label}] result() → {val}")
    except Exception as e:  # noqa: BLE001
        print(f"  [{label}] result() 报错: {type(e).__name__}: {e}")


def try_submit(client: Client, label: str) -> None:
    try:
        fut = client.submit(lambda x: x * 2, 21)
        print(f"  [{label}] submit 成功, status={fut.status}, result={fut.result(timeout=10)}")
    except Exception as e:  # noqa: BLE001
        print(f"  [{label}] submit/result 报错: {type(e).__name__}: {e}")


def main() -> None:
    print("=== 两个 Client 连接同一集群 ===")
    client_a = connect("A")
    client_b = connect("B")
    print(f"  dashboard = {client_a.dashboard_link}")

    print("\n=== B 提交一个较慢的任务（8s）===")
    fut_b = client_b.submit(slow_add, 100, pure=False)
    print(f"  [B] Future key={fut_b.key}  status={fut_b.status}")
    time.sleep(0.5)  # 确保任务已进调度

    print("\n=== A 调用 client.restart() ===")
    t0 = time.perf_counter()
    client_a.restart()
    print(f"  [A] restart 完成  耗时 {time.perf_counter() - t0:.2f}s")
    print(f"  [A] 仍连接? scheduler={client_a.scheduler.address}")
    print(f"  [B] 仍连接? scheduler={client_b.scheduler.address}")

    print("\n=== 看 B 上那个未完成的 Future ===")
    try_result(fut_b, "B-旧Future")

    print("\n=== restart 后两边再 submit 新任务 ===")
    try_submit(client_a, "A-新任务")
    try_submit(client_b, "B-新任务")

    print("\n=== 小结 ===")
    print("  - B 通常不会因为 A.restart() 而整个 Client 报错断开")
    print("  - B 上旧 Future 会失败/取消（上面应能看到异常）")
    print("  - A/B 都可以继续向已重启的 worker 提交新任务")

    client_a.close()
    client_b.close()
    print("\nDEMO 完成。")


if __name__ == "__main__":
    main()
