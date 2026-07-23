"""
停机 Queue 共用：multiprocessing Manager 上的远程 Queue。

Server 注册并监听；Client connect 后 put(STOP_MSG)。
"""

from __future__ import annotations

from multiprocessing.managers import BaseManager
from pathlib import Path
from queue import Empty, Queue
from typing import Any

DIR = Path(__file__).resolve().parent
SCHEDULER_ADDRESS_FILE = DIR / ".dask_queue_stop_scheduler"
CONTROL_READY_FILE = DIR / ".dask_queue_stop_ready"
PID_FILE = DIR / ".dask_queue_stop_pid"
LOG_FILE = DIR / ".dask_queue_stop_server.log"

SCHEDULER_PORT = 8796
DASHBOARD_ADDRESS = ":8797"
CONTROL_HOST = "127.0.0.1"
CONTROL_PORT = 50000
AUTHKEY = b"dask-localcluster-demo"
STOP_MSG = "STOP"


class ControlManager(BaseManager):
    pass


def start_control_server() -> tuple[ControlManager, Queue, Any]:
    """
    在本进程创建 Queue，并启动 Manager 服务（后台线程 serve_forever）。
    返回 (manager, queue, server)。
    """
    q: Queue = Queue()
    ControlManager.register("get_queue", callable=lambda: q)
    mgr = ControlManager(address=(CONTROL_HOST, CONTROL_PORT), authkey=AUTHKEY)
    server = mgr.get_server()

    import threading

    threading.Thread(target=server.serve_forever, name="control-manager", daemon=True).start()
    return mgr, q, server


def connect_control_queue(timeout_s: float = 30.0) -> Any:
    """Client 侧：连上 Manager 并拿到远程 Queue 代理。"""
    import time

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
    raise SystemExit(f"无法连接停机 Queue {CONTROL_HOST}:{CONTROL_PORT}: {last_err}")


def wait_queue(q: Queue, expected: str = STOP_MSG, poll_s: float = 0.5) -> str:
    """阻塞直到收到 expected（或任意非空消息时返回）。"""
    while True:
        try:
            msg = q.get(timeout=poll_s)
        except Empty:
            continue
        if msg == expected:
            return msg
        print(f"  [control] 忽略未知消息: {msg!r}")


def cleanup_runtime_files() -> None:
    for p in (SCHEDULER_ADDRESS_FILE, CONTROL_READY_FILE, PID_FILE):
        if p.exists():
            p.unlink()
