"""
DEMO 一键：启动后台 Server（前台立刻结束）→ Client 干活并发 STOP

用法:
  python dask_queue_stop_demo.py

Windows 下必须 if __name__ == "__main__"。
"""

from __future__ import annotations

import subprocess
import sys
import time
from pathlib import Path

from _queue_control import CONTROL_READY_FILE, PID_FILE

DIR = Path(__file__).resolve().parent
SERVER = DIR / "dask_queue_stop_server.py"
CLIENT = DIR / "dask_queue_stop_client.py"


def main() -> None:
    print("=== 1) 启动 Server（应马上回到命令行语义：子进程已分离）===")
    rc = subprocess.call([sys.executable, str(SERVER)], cwd=str(DIR))
    if rc != 0:
        raise SystemExit(f"Server 启动失败 exit={rc}")

    pid = PID_FILE.read_text(encoding="utf-8").strip() if PID_FILE.exists() else "?"
    print(f"  后台 daemon pid ≈ {pid}")

    print("\n=== 2) 跑 Client（任务结束会发 STOP）===")
    client_rc = subprocess.call([sys.executable, str(CLIENT)], cwd=str(DIR))
    if client_rc != 0:
        raise SystemExit(f"Client 失败 exit={client_rc}")

    print("\n=== 3) 等后台 Server 退出（ready/pid 文件消失）===")
    deadline = time.perf_counter() + 30.0
    while time.perf_counter() < deadline:
        if not CONTROL_READY_FILE.exists() and not PID_FILE.exists():
            print("  后台 Server 已退出。")
            break
        time.sleep(0.2)
    else:
        print("  警告: 30s 内未看到 Server 清理完成，请看 .dask_queue_stop_server.log")

    print("\n一键 DEMO 完成。")


if __name__ == "__main__":
    main()
