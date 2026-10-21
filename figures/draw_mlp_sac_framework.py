"""Draw the MLP-SAC framework used by this repo (HistoryMLP + SAC v2)."""

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from matplotlib import patches
from matplotlib.font_manager import FontProperties
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch

CJK = FontProperties(family='Noto Sans CJK SC')
OUT = Path(__file__).with_name('mlp_sac_framework.png')

INK = '#1F2933'
MUTED = '#52606D'
BLUE = '#1D4E89'
TEAL = '#0F6E6B'
SOFT = '#6B7280'
EDGE = '#243B53'


def edge_point(center, radius, toward):
    v = np.asarray(toward, dtype=float) - np.asarray(center, dtype=float)
    n = np.linalg.norm(v)
    if n < 1e-6:
        return tuple(center)
    return tuple(np.asarray(center, dtype=float) + v / n * radius)


def arrow(ax, start, end, color=INK, lw=1.35, ls='-', rad=0.0, ms=12):
    ax.add_patch(
        FancyArrowPatch(
            start,
            end,
            arrowstyle='-|>',
            mutation_scale=ms,
            lw=lw,
            color=color,
            linestyle=ls,
            connectionstyle=f'arc3,rad={rad}',
            shrinkA=0,
            shrinkB=0,
            zorder=2,
            capstyle='round',
        )
    )


def label(ax, x, y, text, size=9, color=INK, ha='center', va='center', math=False, bg=False):
    ax.text(
        x,
        y,
        text,
        ha=ha,
        va=va,
        fontsize=size,
        color=color,
        fontproperties=None if math else CJK,
        zorder=6,
        bbox=dict(facecolor='white', edgecolor='none', pad=0.12, alpha=0.92) if bg else None,
    )


def mlp_glyph(ax, cx, cy, scale=1.0):
    w, h = 4.6 * scale, 3.4 * scale
    xs = [cx - w * 0.38, cx, cx + w * 0.38]
    counts = (3, 4, 2)
    layers = []
    for x, n in zip(xs, counts):
        ys = np.linspace(cy - h / 2, cy + h / 2, n)
        layers.append([(x, y) for y in ys])
    for left, right in zip(layers, layers[1:]):
        for p in left:
            for q in right:
                ax.plot([p[0], q[0]], [p[1], q[1]], color='#9AA5B1', lw=0.55, zorder=4, solid_capstyle='round')
    for layer in layers:
        ax.scatter(
            [p[0] for p in layer],
            [p[1] for p in layer],
            s=16 * scale,
            c='white',
            edgecolors=EDGE,
            linewidths=0.7,
            zorder=5,
        )


def net_circle(ax, cx, cy, r, face):
    ax.add_patch(
        patches.Circle((cx, cy), r, facecolor=face, edgecolor=EDGE, lw=1.35, zorder=3)
    )
    mlp_glyph(ax, cx, cy + 0.15, scale=r / 8.4)


def min_node(ax, cx, cy):
    ax.add_patch(patches.Circle((cx, cy), 2.15, facecolor='white', edgecolor=EDGE, lw=1.15, zorder=3))
    label(ax, cx, cy, 'min', size=8, math=True)


def rounded(ax, x, y, w, h, face, ec=EDGE, lw=1.15):
    ax.add_patch(
        FancyBboxPatch(
            (x, y),
            w,
            h,
            boxstyle='round,pad=0.25,rounding_size=0.6',
            facecolor=face,
            edgecolor=ec,
            lw=lw,
            zorder=3,
        )
    )


def cylinder(ax, cx, cy, w, h, face='#E4E7EB'):
    hw = w / 2
    cap_h = w * 0.22
    bottom = cy - h / 2
    top = cy + h / 2
    ax.add_patch(
        patches.Rectangle(
            (cx - hw, bottom),
            w,
            h - cap_h / 2,
            facecolor=face,
            edgecolor=EDGE,
            lw=1.2,
            zorder=3,
        )
    )
    ax.add_patch(
        patches.Ellipse((cx, bottom), w, cap_h, facecolor=face, edgecolor=EDGE, lw=1.2, zorder=3)
    )
    ax.add_patch(
        patches.Ellipse((cx, top - cap_h / 2), w, cap_h, facecolor='#F8FAFC', edgecolor=EDGE, lw=1.2, zorder=4)
    )


def main():
    fig, ax = plt.subplots(figsize=(15.8, 9.15), dpi=180)
    ax.set_xlim(0, 124)
    ax.set_ylim(0, 73.5)
    ax.set_aspect('equal')
    ax.axis('off')
    fig.patch.set_facecolor('white')
    ax.set_facecolor('white')

    t1, t2 = (16.5, 50.5), (16.5, 20.5)
    c1, c2 = (52.5, 50.5), (52.5, 20.5)
    r = 7.15
    mt, mc = (33.2, 35.5), (69.2, 35.5)
    pol = (88.6, 35.5)
    rp = (112.2, 52.2)
    env = (104.6, 14.6, 15.2, 8.6)

    face_t = '#F3F4F6'
    face_c = '#FFFFFF'
    face_p = '#E7F0FA'

    for center, face in ((t1, face_t), (t2, face_t), (c1, face_c), (c2, face_c)):
        net_circle(ax, *center, r, face)
    net_circle(ax, *pol, r, face_p)
    min_node(ax, *mt)
    min_node(ax, *mc)

    # captions under circles
    captions = [
        (t1, '目标评估网络', r'$\bar{Q}_1(s^{\prime},a^{\prime})$'),
        (t2, '目标评估网络', r'$\bar{Q}_2(s^{\prime},a^{\prime})$'),
        (c1, '当前评估网络', r'$Q_1(s,a)$'),
        (c2, '当前评估网络', r'$Q_2(s,a)$'),
        (pol, '策略网络', r'$\pi_\theta(a\mid s)$'),
    ]
    for (cx, cy), name, formula in captions:
        label(ax, cx, cy - r - 1.55, name, size=10)
        label(ax, cx, cy - r - 3.35, formula, size=10, math=True)

    label(ax, pol[0], pol[1] - r - 5.05, 'TanhNormal', size=8, color=MUTED)

    # target Q -> min
    for src in (t1, t2):
        arrow(ax, edge_point(src, r, mt), edge_point(mt, 2.15, src))
    label(ax, 22.4, 30.8, '防止过估计', size=8, color=BLUE, bg=True)

    # current Q -> min
    for src in (c1, c2):
        arrow(ax, edge_point(src, r, mc), edge_point(mc, 2.15, src))

    # TD target chip sits in the open band between the two Q columns
    rounded(ax, 36.7, 31.45, 13.8, 8.15, '#F8FBFF', ec=BLUE, lw=1.05)
    label(ax, 43.6, 37.4, 'TD 目标 · MSE', size=8.5, color=BLUE)
    label(ax, 43.6, 35.5, r'$y=r+\gamma(1-d)\,v$', size=8, math=True, color=BLUE)
    label(ax, 43.6, 33.75, r'$v=\min(\bar{Q}_1,\bar{Q}_2)-\alpha\log\pi$', size=6.3, math=True, color=BLUE)
    label(ax, 43.6, 32.1, '梯度只更新当前 Q', size=7.2, color=BLUE)

    arrow(ax, edge_point(mt, 2.15, (36.7, 35.5)), (36.7, 35.5), color=BLUE)
    arrow(ax, (50.5, 37.3), edge_point(c1, r, (50.5, 46)), color=BLUE, rad=-0.15)
    arrow(ax, (50.5, 33.7), edge_point(c2, r, (50.5, 25)), color=BLUE, rad=0.15)

    # actor objective
    arrow(ax, edge_point(mc, 2.15, pol), edge_point(pol, r, mc), color=TEAL)
    label(ax, 72.6, 40.6, r'$\max\;\min_i Q_i-\alpha\log\pi$', size=7.6, math=True, color=TEAL, bg=True)

    # soft update: current -> target, once per gradient step
    arrow(
        ax,
        edge_point(c1, r, (t1[0] + r, t1[1])),
        edge_point(t1, r, (c1[0] - r, c1[1])),
        color=SOFT,
        ls=(0, (3.2, 2.2)),
        rad=0.0,
    )
    arrow(
        ax,
        edge_point(c2, r, (t2[0] + r, t2[1])),
        edge_point(t2, r, (c2[0] - r, c2[1])),
        color=SOFT,
        ls=(0, (3.2, 2.2)),
        rad=0.0,
    )
    label(ax, 39.6, 53.55, '每步软更新', size=8.5, color=SOFT, bg=True)
    label(ax, 34.6, 16.55, r'$\bar\omega \leftarrow \tau\omega+(1-\tau)\bar\omega$', size=7.6, math=True, color=SOFT, bg=True)
    label(ax, 34.6, 14.55, r'$\tau=0.005$', size=8, math=True, color=SOFT, bg=True)

    # alpha sits above the policy and below the a' feedback lane
    rounded(ax, 80.2, 46.15, 16.8, 7.6, '#FFF7ED', ec='#C2410C', lw=1.05)
    label(ax, 88.6, 51.55, '温度系数', size=9, color='#C2410C')
    label(ax, 88.6, 49.65, r'$\alpha=\exp(\log\alpha)$', size=8, math=True, color='#C2410C')
    label(ax, 87.55, 47.85, '目标熵', size=8, color='#C2410C', ha='right')
    label(ax, 88.15, 47.85, r'$\bar{H}=0.3$', size=8, math=True, color='#C2410C', ha='left')
    arrow(ax, (88.6, 46.15), edge_point(pol, r, (88.6, 44)), color='#C2410C')
    label(ax, 96.2, 44.35, r'$-\log\alpha\,(\log\pi+\bar{H})$', size=7.2, math=True, color='#C2410C', ha='left', bg=True)

    # replay + env
    cylinder(ax, rp[0], rp[1], 13.2, 12.4, '#E6E9ED')
    label(ax, rp[0], rp[1] + 1.15, '经验回放库', size=10)
    label(ax, rp[0], rp[1] - 1.15, '均匀采样', size=8.5, color=MUTED)

    rounded(ax, env[0], env[1], env[2], env[3], '#EEF2F6')
    label(ax, env[0] + env[2] / 2, env[1] + env[3] / 2 + 1.15, '环境', size=11)
    label(ax, env[0] + env[2] / 2, env[1] + env[3] / 2 - 1.25, 'PROCS', size=8, color=MUTED)

    # policy -> env -> replay
    arrow(ax, edge_point(pol, r, (env[0], env[1] + env[3] / 2)), (env[0], env[1] + env[3] / 2), rad=0.18)
    label(ax, 99.6, 29.35, '动作', size=8.5)
    label(ax, 99.6, 27.45, r'$a\sim\pi_\theta$', size=8.5, math=True)
    # the mixed label above used math=False; redraw as math-aware split
    arrow(
        ax,
        (env[0] + env[2] / 2, env[1] + env[3]),
        (rp[0], rp[1] - 6.2),
        rad=-0.05,
    )
    label(ax, 116.6, 33.5, '转移存入', size=8.5, ha='left')
    label(ax, 116.6, 31.5, r'$(s,a,r,s^{\prime},d)$', size=8, math=True, ha='left')

    # uniform sample along the top, into the target-side update
    sample_y = 67.4
    ax.plot([rp[0] - 1.5, 14.2], [sample_y, sample_y], color=INK, lw=1.45, solid_capstyle='round', zorder=2)
    arrow(ax, (14.2, sample_y), edge_point(t1, r, (14.2, sample_y)), lw=1.45)
    label(ax, 58, 70.15, '均匀经验采样', size=10, bg=True)
    label(ax, 51.6, 68.45, '批次', size=8.5, color=MUTED, bg=True)
    label(ax, 62.4, 68.45, r'$(s,\,a,\,r,\,s^{\prime},\,d)$', size=8.5, math=True, color=MUTED, bg=True)

    # a' is resampled from the current policy; stop-grad into the target backup.
    # The vertical leg stays left of the temperature box.
    ay = 63.15
    pol_edge = edge_point(pol, r, (76.2, ay))
    ax.plot([pol_edge[0], 76.2], [pol_edge[1], ay], color=TEAL, lw=1.25, solid_capstyle='round', zorder=2)
    ax.plot([76.2, mt[0]], [ay, ay], color=TEAL, lw=1.25, solid_capstyle='round', zorder=2)
    arrow(ax, (mt[0], ay), (mt[0], mt[1] + 2.15), color=TEAL, lw=1.25)
    label(ax, 46.5, 61.35, r"$a^{\prime}\sim\pi_\theta(\cdot\mid s^{\prime})$", size=8, math=True, color=TEAL, bg=True)
    label(ax, 62.8, 61.35, '重参数采样，停梯度', size=8, color=TEAL, bg=True)

    label(ax, 62, 4.55, '四个 Q 与策略的前端都是 HistoryMLP：长度 20 的历史窗口展平后过 MLP。目标网络不反传。', size=8.5, color=MUTED)
    label(ax, 62, 2.35, '观测先做运行均值归一化，再用同一套统计量处理 s 与 s′。', size=8.5, color=MUTED)

    fig.savefig(OUT, bbox_inches='tight', facecolor='white', pad_inches=0.25)
    print(OUT)


if __name__ == '__main__':
    main()
