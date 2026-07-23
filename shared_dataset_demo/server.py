"""
静默后台 LocalCluster（无 CMD 黑框）。多 Client 通过 publish_dataset 共享数据。

用法:
  python server.py        # 拉起后台后立刻退出
  python publisher.py     # Client A：造数 → persist → publish
  python consumer.py --name alice
  python consumer.py --name bob
  python stop_server.py   # Queue 停机

或:  python run_demo.py
"""

from __future__ import annotations

import argparse
import os
import subprocess
import sys
import time
from pathlib import Path

from _common import (
    CONTROL_HOST,
    CONTROL_PORT,
    CONTROL_READY_FILE,
    DASHBOARD_ADDRESS,
    LOG_FILE,
    PID_FILE,
    SCHEDULER_ADDRESS_FILE,
    SCHEDULER_PORT,
    STOP_MSG,
    cleanup_runtime_files,
    start_control_server,
    wait_stop,
)

CREATE_NO_WINDOW = 0x08000000


def _daemon_python() -> str:
    if sys.platform != "win32":
        return sys.executable
    exe = Path(sys.executable)
    for name in ("pythonw.exe", "pythonw"):
        cand = exe.with_name(name)
        if cand.exists():
            return str(cand)
    return sys.executable


def spawn_and_exit() -> None:
    cleanup_runtime_files()
    if LOG_FILE.exists():
        LOG_FILE.unlink()

    cmd = [_daemon_python(), str(Path(__file__).resolve()), "--daemon"]
    log_f = open(LOG_FILE, "w", encoding="utf-8")  # noqa: SIM115
    kwargs: dict = {
        "cwd": str(Path(__file__).resolve().parent),
        "stdin": subprocess.DEVNULL,
        "stdout": log_f,
        "stderr": subprocess.STDOUT,
        "close_fds": True,
    }
    if sys.platform == "win32":
        kwargs["creationflags"] = CREATE_NO_WINDOW
    else:
        kwargs["start_new_session"] = True

    proc = subprocess.Popen(cmd, **kwargs)
    log_f.close()

    deadline = time.perf_counter() + 60.0
    while time.perf_counter() < deadline:
        if CONTROL_READY_FILE.exists() and PID_FILE.exists():
            pid = PID_FILE.read_text(encoding="utf-8").strip()
            addr = SCHEDULER_ADDRESS_FILE.read_text(encoding="utf-8").strip()
            print("=== 共享数据集 DEMO：Server 已静默在后台 ===")
            print(f"  pid={pid}  scheduler={addr}")
            print(f"  dashboard=http://127.0.0.1{DASHBOARD_ADDRESS}/status")
            print(f"  stop queue={CONTROL_HOST}:{CONTROL_PORT}")
            print(f"  log={LOG_FILE}")
            print()
            print("  python publisher.py")
            print("  python consumer.py --name alice")
            print("  python consumer.py --name bob")
            print("  python stop_server.py")
            return
        if proc.poll() is not None:
            tail = LOG_FILE.read_text(encoding="utf-8", errors="replace")[-2000:]
            raise SystemExit(f"启动失败 exit={proc.returncode}\n{tail}")
        time.sleep(0.2)
    raise SystemExit("等待 ready 超时: " + str(LOG_FILE))


def run_daemon() -> None:
    PID_FILE.write_text(str(os.getpid()), encoding="utf-8")
    _, stop_q = start_control_server()
    print(f"停机 Queue {CONTROL_HOST}:{CONTROL_PORT}", flush=True)

    from dask.distributed import LocalCluster

    cluster = LocalCluster(
        n_workers=2,
        threads_per_worker=2,
        memory_limit="1GB",
        processes=False,
        scheduler_port=SCHEDULER_PORT,
        dashboard_address=DASHBOARD_ADDRESS,
        silence_logs=40,
    )
    addr = cluster.scheduler_address
    SCHEDULER_ADDRESS_FILE.write_text(addr, encoding="utf-8")
    CONTROL_READY_FILE.write_text("ready", encoding="utf-8")
    print(f"LocalCluster 就绪 {addr}", flush=True)
    print(f"等 STOP={STOP_MSG!r}", flush=True)
    try:
        wait_stop(stop_q)
        print("收到 STOP，关闭…", flush=True)
    finally:
        cluster.close()
        cleanup_runtime_files()
        print("daemon 退出", flush=True)


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--daemon", action="store_true")
    args = p.parse_args()
    if args.daemon:
        run_daemon()
    else:
        spawn_and_exit()


if __name__ == "__main__":
    main()
