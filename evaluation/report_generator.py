"""
Smart Lane Detection — Report Generator
========================================
Produces visual charts (bar charts, FPS comparisons, heatmaps) and a
text summary from benchmark results.  All output is saved to disk so
it can be embedded in the web UI or viewed standalone.
"""

import os
import sys

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))
import config

# Use a non-interactive backend so the module works in headless
# environments and web servers.
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


# ── Detection Rate Bar Chart ────────────────────────────────────────────────

def generate_detection_chart(results, output_path):
    """Bar chart of detection success across benchmark conditions.

    Parameters
    ----------
    results : dict
        Output of :func:`evaluation.benchmark.run_benchmark`.
    output_path : str
        File path to save the chart image (e.g. ``.png``).

    Returns
    -------
    str or None
        The *output_path* on success, ``None`` on failure.
    """
    conditions = results.get("conditions", {})
    if not conditions:
        return None

    labels = list(conditions.keys())
    detected = [1 if conditions[k]["detection_success"] else 0 for k in labels]

    fig, ax = plt.subplots(figsize=(max(8, len(labels) * 0.8), 5))
    colors = ["#00C853" if d else "#FF1744" for d in detected]
    ax.bar(range(len(labels)), detected, color=colors, edgecolor="white",
           linewidth=0.5)
    ax.set_xticks(range(len(labels)))
    ax.set_xticklabels(labels, rotation=45, ha="right", fontsize=8)
    ax.set_ylabel("Detected (1) / Not Detected (0)")
    ax.set_title("Lane Detection Success by Condition")
    ax.set_ylim(-0.1, 1.4)
    ax.axhline(y=1, color="gray", linestyle="--", linewidth=0.5)
    fig.tight_layout()

    os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)
    fig.savefig(output_path, dpi=150)
    plt.close(fig)
    return output_path


# ── FPS Comparison Chart ────────────────────────────────────────────────────

def generate_fps_chart(results, output_path):
    """Bar chart comparing FPS across benchmark conditions.

    Parameters
    ----------
    results : dict
        Benchmark results dict.
    output_path : str
        File path for the chart image.

    Returns
    -------
    str or None
        *output_path* on success.
    """
    conditions = results.get("conditions", {})
    if not conditions:
        return None

    labels = list(conditions.keys())
    fps_vals = [conditions[k].get("fps", 0) for k in labels]

    fig, ax = plt.subplots(figsize=(max(8, len(labels) * 0.8), 5))
    bar_colors = ["#00C853" if f >= config.TARGET_FPS else "#FF9100"
                  for f in fps_vals]
    ax.bar(range(len(labels)), fps_vals, color=bar_colors, edgecolor="white",
           linewidth=0.5)
    ax.set_xticks(range(len(labels)))
    ax.set_xticklabels(labels, rotation=45, ha="right", fontsize=8)
    ax.set_ylabel("Frames Per Second")
    ax.set_title("Processing FPS by Condition")
    ax.axhline(y=config.TARGET_FPS, color="red", linestyle="--",
               linewidth=1, label=f"Target ({config.TARGET_FPS} FPS)")
    ax.legend(loc="upper right")
    fig.tight_layout()

    os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)
    fig.savefig(output_path, dpi=150)
    plt.close(fig)
    return output_path


# ── Condition Heatmap ───────────────────────────────────────────────────────

def generate_condition_heatmap(results, output_path):
    """Heatmap of performance metrics across all benchmark conditions.

    Rows = conditions, columns = metrics (detection, FPS-ratio,
    processing time).

    Parameters
    ----------
    results : dict
        Benchmark results dict.
    output_path : str
        File path for the heatmap image.

    Returns
    -------
    str or None
        *output_path* on success.
    """
    conditions = results.get("conditions", {})
    if not conditions:
        return None

    labels = list(conditions.keys())
    metric_names = ["Detection", "FPS Ratio", "Proc. Time (ms)"]

    data = []
    for k in labels:
        c = conditions[k]
        det = 1.0 if c["detection_success"] else 0.0
        fps_ratio = c.get("fps", 0) / config.TARGET_FPS if config.TARGET_FPS else 0
        proc_ms = c.get("processing_time", 0) * 1000
        data.append([det, min(fps_ratio, 2.0), proc_ms])

    data = np.array(data)

    fig, ax = plt.subplots(figsize=(6, max(4, len(labels) * 0.45)))
    im = ax.imshow(data, aspect="auto", cmap="RdYlGn")

    ax.set_xticks(range(len(metric_names)))
    ax.set_xticklabels(metric_names, fontsize=9)
    ax.set_yticks(range(len(labels)))
    ax.set_yticklabels(labels, fontsize=8)
    ax.set_title("Performance Heatmap")

    # Annotate cells
    for i in range(len(labels)):
        for j in range(len(metric_names)):
            val = data[i, j]
            fmt = f"{val:.2f}" if j != 2 else f"{val:.1f}"
            ax.text(j, i, fmt, ha="center", va="center", fontsize=7,
                    color="black" if val > 0.5 else "white")

    fig.colorbar(im, ax=ax, fraction=0.03, pad=0.04)
    fig.tight_layout()

    os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)
    fig.savefig(output_path, dpi=150)
    plt.close(fig)
    return output_path


# ── Full Report ─────────────────────────────────────────────────────────────

def generate_full_report(results, output_dir):
    """Generate all charts and a text summary file.

    Output files:

    * ``detection_chart.png``
    * ``fps_chart.png``
    * ``condition_heatmap.png``
    * ``summary.txt``

    Parameters
    ----------
    results : dict
        Benchmark results dict.
    output_dir : str
        Directory for all report artefacts.

    Returns
    -------
    dict
        Paths to each generated file and a ``success`` flag.
    """
    os.makedirs(output_dir, exist_ok=True)

    report = {"success": False}

    # Charts
    det_path = os.path.join(output_dir, "detection_chart.png")
    fps_path = os.path.join(output_dir, "fps_chart.png")
    heatmap_path = os.path.join(output_dir, "condition_heatmap.png")

    report["detection_chart"] = generate_detection_chart(results, det_path)
    report["fps_chart"] = generate_fps_chart(results, fps_path)
    report["condition_heatmap"] = generate_condition_heatmap(results, heatmap_path)

    # Text summary
    summary_path = os.path.join(output_dir, "summary.txt")
    try:
        summary = results.get("summary", {})
        conditions = results.get("conditions", {})

        lines = [
            "=" * 60,
            "  SMART LANE DETECTION — BENCHMARK REPORT",
            "=" * 60,
            "",
            f"Image: {results.get('image_path', 'N/A')}",
            f"Total conditions tested: {summary.get('total_conditions', 0)}",
            f"Conditions passed: {summary.get('passed', 0)}",
            f"Detection rate: {summary.get('detection_rate', 0):.1f}%",
            "",
            "-" * 60,
            f"{'Condition':<25} {'Detected':>9} {'FPS':>8} {'Time(ms)':>10} {'Warning':>10}",
            "-" * 60,
        ]

        for label, c in conditions.items():
            det = "Yes" if c["detection_success"] else "No"
            fps = f"{c.get('fps', 0):.1f}"
            t_ms = f"{c.get('processing_time', 0) * 1000:.1f}"
            warn = c.get("warning_level", "safe")
            lines.append(f"{label:<25} {det:>9} {fps:>8} {t_ms:>10} {warn:>10}")

        lines.extend(["", "=" * 60, ""])

        with open(summary_path, "w", encoding="utf-8") as f:
            f.write("\n".join(lines))

        report["summary_txt"] = summary_path
    except Exception as exc:
        report["summary_txt"] = None
        report["summary_error"] = str(exc)

    report["success"] = True
    return report
