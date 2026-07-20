"""
Dask LocalCluster Dashboard 对照学习目录。

用法:
  python dask_dashboard_demos/index.py
  python dask_dashboard_demos/00_open_cluster.py
  ...
"""

from __future__ import annotations

DEMOS = [
    {
        "script": "00_open_cluster.py",
        "title": "开集群 / 认路",
        "urls": ["/status", "/workers", "/info"],
        "signal": "空闲集群，熟悉导航栏（勿开 /sitemap.json，当前会 500）",
    },
    {
        "script": "01_status_healthy.py",
        "title": "健康 Status",
        "urls": ["/status"],
        "signal": "Task Stream 密实、少白缝；Progress / Processing 均衡",
    },
    {
        "script": "02_task_stream_idle_and_transfer.py",
        "title": "Idle + Transfer",
        "urls": ["/status", "/tasks"],
        "signal": "白缝=空闲浪费；红条=worker 间传数据",
    },
    {
        "script": "03_memory_spill.py",
        "title": "内存 spill",
        "urls": ["/status", "/individual-workers-memory"],
        "signal": "Bytes Stored 色带变化；可能出现橙色磁盘条",
    },
    {
        "script": "04_workers_imbalance.py",
        "title": "Worker 倾斜",
        "urls": ["/workers", "/status"],
        "signal": "Occupancy / Processing 明显不均",
    },
    {
        "script": "05_profile_hotspot.py",
        "title": "Profile 热点",
        "urls": ["/profile", "/status"],
        "signal": "火焰图对比 burn_python vs burn_numpy（Py3.13 可直接看 /profile 实时页）",
    },
    {
        "script": "06_graph_and_groups.py",
        "title": "Graph / Groups",
        "urls": ["/graph", "/groups"],
        "signal": "小图依赖形状（/graph + /groups；venv 内已打 /groups KeyError 防护）",
    },
    {
        "script": "07_system_and_hardware.py",
        "title": "System / Hardware",
        "urls": ["/system", "/hardware"],
        "signal": "机器 CPU/磁盘曲线 vs 任务视图",
    },
    {
        "script": "08_fine_performance_metrics.py",
        "title": "Fine Performance Metrics",
        "urls": [
            "/individual-aggregate-time-per-action",
            "/individual-compute-time-per-key",
            "/status",
        ],
        "signal": "execute / transfer / idle 占比；More... 里同名页",
    },
    {
        "script": "09_performance_report.py",
        "title": "静态 performance_report",
        "urls": ["/status", "output/dask-performance-report.html"],
        "signal": "导出 HTML 做调参前后对比",
    },
]

NAV_PAGES = [
    ("/status", "总览：内存、Processing/CPU/Occupancy、Task Stream、Progress"),
    ("/workers", "各 worker 线程、内存、任务"),
    ("/tasks", "更长回溯的任务块视图"),
    ("/system", "集群级 CPU/内存/带宽/FD"),
    ("/profile", "统计采样火焰图"),
    ("/graph", "当前图依赖树"),
    ("/groups", "按 task group 看依赖/内存/进度"),
    ("/info", "worker 详情入口"),
    ("/hardware", "内存/磁盘/网络带宽探测"),
]


def main() -> None:
    print("=== Dashboard 导航栏主页（LocalCluster 调优核心）===")
    for path, desc in NAV_PAGES:
        print(f"  {path:<12} {desc}")

    print("\n=== 学习顺序 DEMO（默认 dashboard http://127.0.0.1:8787 ）===")
    for i, d in enumerate(DEMOS, 1):
        print(f"\n{i:02d}. {d['script']}")
        print(f"    {d['title']} — {d['signal']}")
        print(f"    对开: {', '.join(d['urls'])}")
        print(f"    运行: python dask_dashboard_demos/{d['script']}")

    print("\n提示: 每个脚本会打印完整 URL；看完按回车再跑下一个。")


if __name__ == "__main__":
    main()
