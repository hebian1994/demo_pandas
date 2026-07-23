"""
DEMO Server：启动后立刻回到命令行；后台进程无控制台窗口（不弹 CMD 黑框）。

用法:
  python dask_queue_stop_server.py          # 拉起静默 daemon 后立刻退出
  python dask_queue_stop_client.py          # 干活 → Queue.put("STOP") → daemon 退出
  python dask_queue_stop_demo.py            # 上面两步一键跑完

说明:
  - Windows 用 CREATE_NO_WINDOW + 优先 pythonw，避免另开黑框
  - daemon 日志只写到 .dask_queue_stop_server.log
  - 后台 LocalCluster 用 processes=False，避免每个 worker 再弹控制台

Windows 下必须 if __name__ == "__main__"。
"""

from __future__ import annotations

import argparse
import os
import subprocess
import sys
from pathlib import Path

from _queue_control import (
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
    wait_queue,
)

# Windows: 不给子进程分配控制台窗口（比 DETACHED_PROCESS 更不容易弹黑框）
CREATE_NO_WINDOW = 0x08000000


def _daemon_python() -> str:
    """Windows 优先 pythonw.exe（无控制台子系统）。"""
    if sys.platform != "win32":
        return sys.executable
    exe = Path(sys.executable)
    for name in ("pythonw.exe", "pythonw"):
        cand = exe.with_name(name)
        if cand.exists():
            return str(cand)
    return sys.executable


def spawn_daemon_and_exit() -> None:
    """前台入口：静默拉起 --daemon，确认 ready 后本进程退出。"""
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

    import time

    deadline = time.perf_counter() + 60.0
    while time.perf_counter() < deadline:
        if CONTROL_READY_FILE.exists() and PID_FILE.exists():
            pid = PID_FILE.read_text(encoding="utf-8").strip()
            addr = SCHEDULER_ADDRESS_FILE.read_text(encoding="utf-8").strip()
            print("=== Server 已静默在后台运行，本进程退出 ===")
            print(f"  daemon pid = {pid}")
            print(f"  scheduler  = {addr}")
            print(f"  dashboard  = http://127.0.0.1{DASHBOARD_ADDRESS}/status")
            print(f"  stop queue = {CONTROL_HOST}:{CONTROL_PORT}  msg={STOP_MSG!r}")
            print(f"  log        = {LOG_FILE}  （无额外窗口，只写这个文件）")
            print()
            print("下一步:  python dask_queue_stop_client.py")
            return
        if proc.poll() is not None:
            tail = LOG_FILE.read_text(encoding="utf-8", errors="replace")[-2000:]
            raise SystemExit(
                f"后台 Server 启动失败（exit={proc.returncode}）。日志末尾:\n{tail}"
            )
        time.sleep(0.2)

    raise SystemExit("等待后台 Server ready 超时，请查看 " + str(LOG_FILE))


def run_daemon() -> None:
    """后台进程：无窗口；stdout 已被重定向到日志文件。"""
    PID_FILE.write_text(str(os.getpid()), encoding="utf-8")

    _, stop_q, _server = start_control_server()
    print(f"停机 Queue 监听 {CONTROL_HOST}:{CONTROL_PORT}", flush=True)

    from dask.distributed import LocalCluster

    # processes=False：线程 worker，Windows 上不会为每个 worker 再弹 CMD
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

    print(f"LocalCluster 就绪 scheduler={addr}", flush=True)
    print(f"等待停机信号 {STOP_MSG!r} …", flush=True)

    try:
        msg = wait_queue(stop_q, expected=STOP_MSG)
        print(f"收到停机信号: {msg!r}，关闭集群…", flush=True)
    finally:
        cluster.close()
        cleanup_runtime_files()
        print("LocalCluster 已关闭。daemon 退出。", flush=True)


def main() -> None:
    parser = argparse.ArgumentParser(description="LocalCluster 静默后台 Server（Queue 停机）")
    parser.add_argument(
        "--daemon",
        action="store_true",
        help="后台常驻模式（由本脚本自动拉起，一般不必手写）",
    )
    args = parser.parse_args()
    if args.daemon:
        run_daemon()
    else:
        spawn_daemon_and_exit()


if __name__ == "__main__":
    main()
