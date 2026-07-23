"""
DEMO：单独启动 LocalCluster（调度器常驻，供其他进程 Client 连接）

用法:
  终端 1:  python dask_local_cluster_server.py
  终端 2:  python dask_client_dataframe_demo.py

本脚本只负责起集群并打印连接地址；计算在 Client 脚本里提交。
Windows 下必须 if __name__ == "__main__"。
"""

from __future__ import annotations

from pathlib import Path

from dask.distributed import LocalCluster

# 固定端口，方便第二个脚本默认连接；被占用时改这里两处即可
SCHEDULER_PORT = 8786
DASHBOARD_ADDRESS = ":8787"
ADDRESS_FILE = Path(__file__).resolve().parent / ".dask_scheduler_address"


def main() -> None:
    # 500MB 数据 + map 中间结果，给每个 worker 留足内存
    cluster = LocalCluster(
        n_workers=2,
        threads_per_worker=2,
        memory_limit="2GB",
        processes=True,
        scheduler_port=SCHEDULER_PORT,
        dashboard_address=DASHBOARD_ADDRESS,
        silence_logs=40,
    )

    addr = cluster.scheduler_address
    ADDRESS_FILE.write_text(addr, encoding="utf-8")

    print("=== LocalCluster 已启动 ===")
    print(f"  scheduler  = {addr}")
    print(f"  dashboard  = http://127.0.0.1{DASHBOARD_ADDRESS}/status")
    print(f"  address 已写入: {ADDRESS_FILE}")
    print()
    print("请另开终端运行:")
    print("  python dask_client_dataframe_demo.py")
    print("  python dask_scatter_lifetime_demo.py   # scatter/persist 生命周期")
    print("  python dask_restart_two_clients_demo.py  # 双 Client + restart")
    print("  python dask_queue_stop_demo.py           # 后台 Server + Queue 停机")
    print("本窗口保持运行；按回车或 Ctrl+C 关闭集群。")

    try:
        input()
    except (EOFError, KeyboardInterrupt):
        print("\n正在关闭…")
    finally:
        cluster.close()
        if ADDRESS_FILE.exists():
            ADDRESS_FILE.unlink()
        print("LocalCluster 已关闭。")


if __name__ == "__main__":
    main()
