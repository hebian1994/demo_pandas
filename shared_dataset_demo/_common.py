"""
多 Client 共享集群数据集：连接 / 停机 Queue / 常量。
"""

from __future__ import annotations

import time
from multiprocessing.managers import BaseManager
from pathlib import Path
from queue import Empty, Queue
from typing import Any

from dask.distributed import Client

DIR = Path(__file__).resolve().parent
SCHEDULER_ADDRESS_FILE = DIR / ".shared_ds_scheduler"
CONTROL_READY_FILE = DIR / ".shared_ds_ready"
PID_FILE = DIR / ".shared_ds_pid"
LOG_FILE = DIR / ".shared_ds_server.log"

# 与 localcluster_demo 端口错开，可同时实验
SCHEDULER_PORT = 8806
DASHBOARD_ADDRESS = ":8807"
CONTROL_HOST = "127.0.0.1"
CONTROL_PORT = 50010
AUTHKEY = b"shared-dataset-demo"
STOP_MSG = "STOP"

# 挂在 scheduler 上的共享数据集名字
DATASET_NAME = "sales_shared"


class ControlManager(BaseManager):
    pass


def start_control_server() -> tuple[Any, Queue]:
    q: Queue = Queue()
    ControlManager.register("get_queue", callable=lambda: q)
    mgr = ControlManager(address=(CONTROL_HOST, CONTROL_PORT), authkey=AUTHKEY)
    server = mgr.get_server()
    import threading

    threading.Thread(target=server.serve_forever, name="control-manager", daemon=True).start()
    return mgr, q


def connect_control_queue(timeout_s: float = 30.0) -> Any:
    ControlManager.register("get_queue")
    mgr = ControlManager(address=(CONTROL_HOST, CONTROL_PORT), authkey=AUTHKEY)
    deadline = time.perf_counter() + timeout_s
    last_err: Exception | None = None
    while time.perf_counter() < deadline:
        try:
            mgr.connect()
            return mgr.get_queue()
        except Exception as e:  # noqa: BLE001
            last_err = e
            time.sleep(0.2)
    raise SystemExit(f"无法连接停机 Queue: {last_err}")


def wait_stop(q: Queue, poll_s: float = 0.5) -> None:
    while True:
        try:
            msg = q.get(timeout=poll_s)
        except Empty:
            continue
        if msg == STOP_MSG:
            return
        print(f"  [control] 忽略: {msg!r}")


def cleanup_runtime_files() -> None:
    for p in (SCHEDULER_ADDRESS_FILE, CONTROL_READY_FILE, PID_FILE):
        if p.exists():
            p.unlink()


def wait_server_ready(timeout_s: float = 60.0) -> str:
    deadline = time.perf_counter() + timeout_s
    while time.perf_counter() < deadline:
        if CONTROL_READY_FILE.exists() and SCHEDULER_ADDRESS_FILE.exists():
            return SCHEDULER_ADDRESS_FILE.read_text(encoding="utf-8").strip()
        time.sleep(0.2)
    raise SystemExit(
        "等待 Server 超时。请先:  python server.py\n"
        "或一键:  python run_demo.py"
    )


def connect_client(name: str = "client") -> Client:
    addr = wait_server_ready()
    client = Client(addr)
    print(f"  [{name}] 已连接  id={client.id}  scheduler={addr}")
    print(f"  [{name}] dashboard={client.dashboard_link}")
    return client
