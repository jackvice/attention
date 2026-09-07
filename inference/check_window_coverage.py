#!/usr/bin/env python3
"""Report the angular coverage of the five active-vision windows.

Throwaway analysis script: prints the yaw interval each window spans, the
overlap between neighbours, and how much of each window falls outside the
camera, then plots the same thing for the configured and tiling window widths.
"""
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

HFOV_SRC_DEG = np.rad2deg(2.8)

OLD_YAWS_DEG, OLD_WIN_DEG = (-64.0, -32.0, 0.0, 32.0, 64.0), 60.0
NEW_YAWS_DEG, NEW_WIN_DEG = (-60.0, -30.0, 0.0, 30.0, 60.0), 40.0


def intervals(yaws, win_deg):
    half = win_deg / 2.0
    return [(y - half, y + half) for y in yaws]


def report(yaws, win_deg, label):
    print(f"\n{label}: window HFOV {win_deg:g} deg at yaws "
          f"{[f'{y:+.0f}' for y in yaws]}, spacing {yaws[1] - yaws[0]:g} deg")
    spans = intervals(yaws, win_deg)
    edge = HFOV_SRC_DEG / 2.0
    for i, (lo, hi) in enumerate(spans):
        outside = max(0.0, edge - hi) * 0 + max(0.0, hi - edge) + max(0.0, -edge - lo)
        print(f"  w{i}: [{lo:+7.1f}, {hi:+7.1f}]   outside camera: "
              f"{outside:5.1f} deg ({100 * outside / win_deg:4.1f}%)")
    for i in range(len(spans) - 1):
        ov = spans[i][1] - spans[i + 1][0]
        print(f"  w{i} & w{i+1} overlap: {ov:+.1f} deg "
              f"({100 * ov / win_deg:+.1f}% of a window)")
    union_lo = min(s[0] for s in spans)
    union_hi = max(s[1] for s in spans)
    print(f"  union: [{union_lo:+.1f}, {union_hi:+.1f}] = {union_hi - union_lo:.1f} deg "
          f"vs camera {HFOV_SRC_DEG:.1f} deg")


def plot(path):
    edge = HFOV_SRC_DEG / 2.0
    fig, axes = plt.subplots(2, 1, figsize=(11, 5), sharex=True)
    for ax, yaws, win, title in (
        (axes[0], OLD_YAWS_DEG, OLD_WIN_DEG,
         f"Previous: {OLD_WIN_DEG:g} deg windows every "
         f"{OLD_YAWS_DEG[1] - OLD_YAWS_DEG[0]:g} deg (47% overlap, outer two spill past the camera)"),
        (axes[1], NEW_YAWS_DEG, NEW_WIN_DEG,
         f"Current: {NEW_WIN_DEG:g} deg windows every "
         f"{NEW_YAWS_DEG[1] - NEW_YAWS_DEG[0]:g} deg (25% overlap, exactly fills the camera)"),
    ):
        for i, (lo, hi) in enumerate(intervals(yaws, win)):
            ax.barh(i, hi - lo, left=lo, height=0.7,
                    color=f"C{i}", alpha=0.65, edgecolor="k", linewidth=0.8)
            ax.text((lo + hi) / 2, i, f"w{i}", ha="center", va="center", fontsize=9)
        ax.axvspan(-200, -edge, color="k", alpha=0.13)
        ax.axvspan(edge, 200, color="k", alpha=0.13)
        ax.axvline(-edge, color="k", ls="--", lw=1)
        ax.axvline(edge, color="k", ls="--", lw=1)
        ax.set_yticks(range(5))
        ax.set_yticklabels([f"{y:+.0f} deg" for y in yaws])
        ax.set_xlim(-105, 105)
        ax.set_title(title, fontsize=11)
        ax.grid(axis="x", alpha=0.3)
    axes[1].set_xlabel("yaw relative to camera axis (deg); shaded = outside the "
                       f"{HFOV_SRC_DEG:.0f} deg camera")
    fig.tight_layout()
    fig.savefig(path, dpi=130)
    print(f"\nWrote {path}")


if __name__ == "__main__":
    report(OLD_YAWS_DEG, OLD_WIN_DEG, "Previous")
    report(NEW_YAWS_DEG, NEW_WIN_DEG, "Current")
    plot("/tmp/window_coverage.png")
