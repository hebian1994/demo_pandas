"""
DEMO Client：连上「Queue 停机」Server，跑任务后向 Queue 发 STOP

前置: Server 已启动（单独终端或由 dask_queue_stop_demo.py 拉起）

Windows 下必须 if __name__ == "__main__"。
"""

from __future__ import annotations

import time
from pathlib import Path

from dask.distributed import Client

from _queue_control import (
    CONTROL_READY_FILE,
    SCHEDULER_ADDRESS_FILE,
    STOP_MSG,
    connect_control_queue,
)


def wait_server_ready(timeout_s: float = 60.0) -> str:
    deadline = time.perf_counter() + timeout_s
    while time.perf_counter() < deadline:
        if CONTROL_READY_FILE.exists() and SCHEDULER_ADDRESS_FILE.exists():
            return SCHEDULER_ADDRESS_FILE.read_text(encoding="utf-8").strip()
        time.sleep(0.2)
    raise SystemExit(
        f"等待 Server 超时（未出现 {CONTROL_READY_FILE.name}）。\n"
        "请先运行:  python dask_queue_stop_server.py\n"
        "或一键:    python dask_queue_stop_demo.py"
    )


def run_jobs(client: Client) -> None:
    print("\n=== 提交任务 ===")
    futs = client.map(lambda x: x * x, range(20))
    results = client.gather(futs)
    print(f"  map(x*x, 0..19) → {results[:5]} … {results[-3:]}")
    s = client.submit(sum, futs).result()
    print(f"  sum = {s}")


def main() -> None:
    print("=== 等待 Server ready ===")
    addr = wait_server_ready()
    print(f"  scheduler = {addr}")

    client = Client(addr)
    print(f"  dashboard = {client.dashboard_link}")

    try:
        run_jobs(client)
    finally:
        client.close()
        print("  Client 已断开")

    print(f"\n=== 向停机 Queue 发送 {STOP_MSG!r} ===")
    q = connect_control_queue()
    q.put(STOP_MSG)
    print("  已 put。Server 应随后关闭集群并退出。")


if __name__ == "__main__":
    main()
