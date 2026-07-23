"""
一键 DEMO：静默 Server → Publisher 发布 → 两个 Consumer 共享读取 → 停机

用法:
  python run_demo.py
"""

from __future__ import annotations

import subprocess
import sys
import time
from pathlib import Path

from _common import CONTROL_READY_FILE, PID_FILE

DIR = Path(__file__).resolve().parent


def run(script: str, *args: str) -> None:
    cmd = [sys.executable, str(DIR / script), *args]
    print(f"\n>>> {' '.join(cmd)}")
    rc = subprocess.call(cmd, cwd=str(DIR))
    if rc != 0:
        raise SystemExit(f"{script} 失败 exit={rc}")


def main() -> None:
    print("=== 多 Client 共享集群数据集 DEMO ===")
    run("server.py")

    run("publisher.py")
    # 两个独立进程 = 两个 Client，同时读同一 published 数据集
    run("consumer.py", "--name", "alice", "--mode", "by_region")
    run("consumer.py", "--name", "bob", "--mode", "by_sku")
    run("consumer.py", "--name", "carol", "--mode", "totals")

    run("stop_server.py")

    deadline = time.perf_counter() + 20.0
    while time.perf_counter() < deadline:
        if not CONTROL_READY_FILE.exists() and not PID_FILE.exists():
            print("\n后台 Server 已退出。DEMO 完成。")
            return
        time.sleep(0.2)
    print("\n警告: Server 可能未退出，请看 .shared_ds_server.log")


if __name__ == "__main__":
    main()
