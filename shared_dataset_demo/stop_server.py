"""向停机 Queue 发 STOP，结束后台 Server。"""

from __future__ import annotations

from _common import STOP_MSG, connect_control_queue


def main() -> None:
    print(f"发送停机信号 {STOP_MSG!r} …")
    q = connect_control_queue()
    q.put(STOP_MSG)
    print("已发送。Server 将关闭集群并退出。")


if __name__ == "__main__":
    main()
