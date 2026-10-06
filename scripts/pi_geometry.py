#!/usr/bin/env python
"""Camera geometry for a litter cart: Pi camera at a fixed height above the ground.

Answers two deployment questions:
  1. How much ground does the camera see ahead, as a function of down-tilt?
  2. How many pixels wide does a piece of trash appear, as a function of distance
     and inference resolution? (decides whether the Pi can run at 320/416 instead of 640)

Everything is a plain pinhole model; no GPU needed.
"""
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

plt.rcParams["font.sans-serif"] = ["Microsoft YaHei", "SimHei", "DejaVu Sans"]
plt.rcParams["axes.unicode_minus"] = False

H = 0.50          # camera height above ground [m]
HFOV = 66.0       # Pi Camera Module 3 horizontal field of view [deg]
VFOV = 41.0       # vertical field of view for a 16:9 readout [deg]
OBJ = 0.10        # nominal trash size (a can lying down) [m]
MIN_PX = 16       # rule of thumb: below this a detector gets unreliable

def ground_band(tilt_deg):
    """Near/far ground distance seen by the camera, looking at `tilt_deg` below horizontal."""
    t = np.radians(tilt_deg)
    half = np.radians(VFOV / 2)
    near_a = t + half                      # steepest ray -> closest ground point
    far_a = t - half                       # shallowest ray -> farthest ground point
    near = H / np.tan(near_a)
    far = np.inf if far_a <= 0 else H / np.tan(far_a)
    return near, far

def px_width(dist, obj=OBJ, imgsz=640):
    """Apparent width in pixels of an object of size `obj` at ground distance `dist`."""
    f = imgsz / (2 * np.tan(np.radians(HFOV / 2)))
    return f * obj / np.maximum(dist, 1e-6)

def px_width_vertical(dist, obj=OBJ, imgsz=640):
    """Apparent height in pixels (foreshortened) for an object of height `obj` standing up."""
    f = imgsz * (16 / 9) / (2 * np.tan(np.radians(VFOV / 2)))
    a_bottom = np.arctan2(H, dist)
    a_top = np.arctan2(np.maximum(H - obj, 0.0), dist)
    return f * np.tan(a_bottom - a_top)

def main():
    fig, axes = plt.subplots(1, 3, figsize=(17, 5.0))
    dist = np.linspace(0.3, 5.0, 500)

    # ---- panel 1: ground coverage band vs tilt
    ax = axes[0]
    tilts = np.arange(15, 61, 1)
    near = np.array([ground_band(t)[0] for t in tilts])
    far = np.array([min(ground_band(t)[1], 8.0) for t in tilts])
    ax.fill_between(tilts, near, far, color="#cfe0ff", label="可见地面范围")
    for t, y in ((30, 6.6), (35, 5.3)):
        n, f = ground_band(t)
        ax.axvline(t, color="#2f6fed", ls="--", lw=1)
        ax.annotate("俯角 %d°: %.2f–%.2f m" % (t, n, f), (t + 0.6, y), fontsize=8,
                    color="#2f6fed", ha="left")
    ax.set_xlabel("相机俯角 (度)")
    ax.set_ylabel("离车前方的地面距离 (m)")
    ax.set_title("0.5 m 高度时，相机看到的地面范围\n(66° 水平 / 41° 垂直视场)")
    ax.set_ylim(0, 8)
    ax.grid(alpha=0.3)
    ax.legend(fontsize=9)

    # ---- panel 2: pixel width vs distance at several input resolutions
    ax = axes[1]
    for imgsz, c in ((640, "#2f6fed"), (416, "#f0a020"), (320, "#d04040")):
        w = px_width(dist, OBJ, imgsz)
        ax.plot(dist, w, color=c, lw=2, label="输入 %d px" % imgsz)
        reach = dist[np.where(w >= MIN_PX)[0][-1]] if (w >= MIN_PX).any() else 0
        ax.plot([reach], [MIN_PX], "o", color=c, ms=6)
        ax.annotate("%.1f m" % reach, (reach, MIN_PX), textcoords="offset points",
                    xytext=(4, 6), fontsize=8, color=c)
    ax.axhline(MIN_PX, color="gray", ls=":", lw=1)
    ax.text(3.2, MIN_PX + 2, "16 px 可靠下限", fontsize=8, color="gray")
    ax.set_xlabel("离车前方的地面距离 (m)")
    ax.set_ylabel("10 cm 垃圾的像素宽度")
    ax.set_title("物体在画面里有多大\n(10 cm 罐头，宽度方向)")
    ax.set_ylim(0, 90)
    ax.grid(alpha=0.3)
    ax.legend(fontsize=9)

    # ---- panel 3: what is actually usable = field of view AND big enough in pixels
    ax = axes[2]
    tilts3 = [25, 30, 35, 40, 45]
    res = [(640, "#2f6fed"), (416, "#f0a020"), (320, "#d04040")]
    reach = {}
    for imgsz, _ in res:
        w = px_width(dist, OBJ, imgsz)
        ok = dist[w >= MIN_PX]
        reach[imgsz] = float(ok.max()) if len(ok) else 0.0
    xpos = np.arange(len(tilts3))
    bw = 0.26
    for k, (imgsz, c) in enumerate(res):
        depths = []
        for t in tilts3:
            n, f = ground_band(t)
            f = 8.0 if f == np.inf else f
            depths.append(max(0.0, min(f, reach[imgsz]) - n))
        bars = ax.bar(xpos + (k - 1) * bw, depths, bw, color=c,
                      label="输入 %d px（细看可到 %.1f m）" % (imgsz, reach[imgsz]))
        ax.bar_label(bars, fmt="%.1fm", fontsize=7)
    ax.set_xticks(xpos)
    ax.set_xticklabels(["%d°" % t for t in tilts3])
    ax.set_xlabel("相机俯角 (度)")
    ax.set_ylabel("有效检测纵深 (m)")
    ax.set_title("真正能用的纵深 = 视场范围 与 物体足够大 的重叠\n(超出 %.1f m 的部分算无效)" % reach[640])
    ax.grid(axis="y", alpha=0.3)
    ax.legend(fontsize=8)
    ax.text(0.02, 0.97, "俯角 30°~35° 最平衡：\n约 0.4~2.0 m 可用，\n320 px 输入也基本够。",
            transform=ax.transAxes, fontsize=8.5, va="top",
            bbox=dict(fc="#fff6d5", ec="#e0c060"))

    fig.tight_layout()
    out = "results/pi_geometry.png"
    fig.savefig(out, dpi=150)
    print("saved", out)
    for t in (25, 30, 35, 40, 45):
        n, f = ground_band(t)
        print("tilt %2d deg -> ground %.2f .. %s m" % (t, n, "inf" if f == np.inf else "%.2f" % f))
    for imgsz in (640, 416, 320):
        w = px_width(dist, OBJ, imgsz)
        ok = dist[w >= MIN_PX]
        print("imgsz %3d -> 10cm object reaches >=16px out to %.2f m" % (imgsz, ok.max() if len(ok) else 0))
    for d in (0.5, 1.0, 1.5, 2.0, 3.0):
        print("  d=%.1fm  width@640=%5.1f  width@320=%5.1f  height@640=%5.1f"
              % (d, px_width(d), px_width(d, imgsz=320), px_width_vertical(d)))

if __name__ == "__main__":
    main()