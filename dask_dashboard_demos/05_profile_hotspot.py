"""
DEMO 05：Profile 火焰图 —— 找出 CPU 热点代码。

怎么看（先读这段再开 /profile）:
  1. 图是「火焰图 / 调用栈」：竖着是谁调用谁，横宽是时间占比
  2. 底下几层几乎总是 threading / worker / _task_spec —— 这是 Dask 壳，可忽略
  3. 往上看，找到你自己的函数名（本 DEMO：burn_python / burn_numpy）
  4. 鼠标悬停：Name / Filename / Time / Percentage
  5. 条越宽越值得优化；慢路径应明显比快路径「胖」或墙钟时间更长

本项目已用 Python 3.13：Dashboard /profile 实时页可用（3.11 才会 disabled）。
仍会额外导出 HTML，方便存档对比。
"""

from __future__ import annotations

import sys
import time
from pathlib import Path

import numpy as np
from dask import delayed

sys.path.insert(0, str(Path(__file__).resolve().parent))

from _common import (
    local_client,
    print_cluster_info,
    print_watch_urls,
    wait_for_browser,
)

OUTPUT_DIR = Path(__file__).resolve().parent.parent / "output"
PROFILE_SLOW = OUTPUT_DIR / "profile_slow_python.html"
PROFILE_FAST = OUTPUT_DIR / "profile_fast_numpy.html"


@delayed
def burn_python(rounds: int, inner: int) -> int:
    """纯 Python 双重循环：采样应能打中本函数。"""
    s = 0
    for _ in range(rounds):
        for j in range(inner):
            s += (j * j) % 97
    return s


@delayed
def burn_numpy(n: int) -> int:
    """同等量级的向量化计算（小数组，避免 OOM）。"""
    x = np.arange(n, dtype=np.int64)
    s = 0
    for _ in range(80):
        s += int(np.sum((x * x) % 97))
    return s


def _print_howto(*, live_dashboard: bool) -> None:
    where = "/profile 实时页" if live_dashboard else "导出的 HTML"
    print(
        f"""
======== 火焰图怎么读（看 {where}）========
1) 纵轴 = 调用栈：底=外层，顶=真正干活
2) 横宽 = 时间占比：越宽越热
3) 悬停看 Name / Filename / Percentage
4) 忽略：threading / distributed/worker / dask/_task_spec
5) 找：burn_python（慢） vs burn_numpy（快）
6) 若顶层几乎只有 __call__ / worker：以墙钟 elapsed 对比为主
==========================================
"""
    )


def _iter_frames(node: dict, path: list[str] | None = None):
    path = path or []
    desc = node.get("description") or {}
    if isinstance(desc, dict):
        name = desc.get("name") or ""
        filename = desc.get("filename") or ""
        label = f"{name} ({Path(filename).name})" if filename else name or "?"
    else:
        label = str(desc)
    new_path = path + [label]
    count = int(node.get("count") or 0)
    yield count, new_path
    children = node.get("children") or {}
    items = children.values() if isinstance(children, dict) else children
    for child in items:
        if isinstance(child, dict):
            yield from _iter_frames(child, new_path)


def _summarize_profile(prof, label: str) -> None:
    data = prof[0] if isinstance(prof, tuple) else prof
    if not isinstance(data, dict):
        print(f"\n--- {label}：无法解析 profile 结构 ---")
        return

    rows = sorted(_iter_frames(data), key=lambda x: x[0], reverse=True)
    print(f"\n--- {label}：采样最多的栈（count 越大越热）---")
    shown = 0
    for count, path in rows:
        tip = path[-1]
        interesting = any(k in tip for k in ("burn_python", "burn_numpy")) or (
            count > 0 and shown < 6 and "threading" not in tip
        )
        if not interesting and "burn_" not in " ".join(path):
            continue
        mark = ""
        joined = " > ".join(path[-3:])
        if "burn_python" in joined:
            mark = "  << 慢路径业务函数"
        elif "burn_numpy" in joined:
            mark = "  << 快路径业务函数"
        print(f"  count={count:4d}  {joined}{mark}")
        shown += 1
        if shown >= 10:
            break
    if shown == 0:
        print("  (未解析到帧；请看 /profile 或 HTML 悬停，或看 elapsed 对比)")


def demo() -> None:
    OUTPUT_DIR.mkdir(exist_ok=True)
    # 仅 3.11 的 Dashboard Profile UI 会被 Dask 强制 disabled
    live_dashboard = sys.version_info[:2] != (3, 11)
    _print_howto(live_dashboard=live_dashboard)

    print(f"当前解释器: Python {sys.version.split()[0]} @ {sys.executable}")
    if live_dashboard:
        print("可用：打开 /profile 实时看火焰图（计算进行中盯着即可）。\n")
    else:
        print("Python 3.11：/profile 会 disabled，请看导出 HTML。\n")

    with local_client(n_workers=2, threads_per_worker=2, memory_limit="1GB") as client:
        print_cluster_info(client)
        print_watch_urls(client, ["/profile", "/status"], title="05 Profile 热点")

        print("[1/2] 慢路径 burn_python —— 此时打开 /profile")
        t0 = time.perf_counter()
        _ = client.compute(
            [burn_python(400, 80_000), burn_python(400, 80_000)],
            sync=True,
        )
        slow_s = time.perf_counter() - t0
        print(f"  elapsed={slow_s:.1f}s")
        time.sleep(1.2)
        prof_slow = client.profile(plot=True, filename=str(PROFILE_SLOW))
        print(f"  备份 HTML: {PROFILE_SLOW}")
        _summarize_profile(prof_slow, "慢路径")

        print("\n[2/2] 快路径 burn_numpy —— 继续看 /profile（或刷新后框选后半段）")
        t0 = time.perf_counter()
        _ = client.compute([burn_numpy(200_000), burn_numpy(200_000)], sync=True)
        fast_s = time.perf_counter() - t0
        print(f"  elapsed={fast_s:.1f}s")
        time.sleep(1.2)
        prof_fast = client.profile(plot=True, filename=str(PROFILE_FAST))
        print(f"  备份 HTML: {PROFILE_FAST}")
        _summarize_profile(prof_fast, "快路径")

        ratio = (slow_s / fast_s) if fast_s > 0 else float("inf")
        print(
            f"\n墙钟对比: 慢 {slow_s:.1f}s / 快 {fast_s:.1f}s ≈ {ratio:.1f}x\n"
            "重点在 /profile 找 burn_python / burn_numpy；HTML 仅作存档。"
        )
        wait_for_browser()


if __name__ == "__main__":
    demo()
