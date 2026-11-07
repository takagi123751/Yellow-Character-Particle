# -*- coding: utf-8 -*-
"""
黄色小精灵粒子动画
依赖：pip install moderngl glfw numpy
运行：python yellow_character.py（需要 OpenGL 3.3 图形界面 / 显示器）
"""

import math
import os
import sys
from pathlib import Path

try:
    import glfw
    import moderngl
    import numpy as np
except ModuleNotFoundError:
    # If VS Code starts a global Python that lacks the dependencies, relaunch
    # this file with the project's isolated environment.
    py_path = Path(__file__).resolve()
    venv_python = py_path.parent / ".venv" / "Scripts" / "python.exe"
    if not venv_python.is_file():
        venv_python = py_path.parents[1] / ".venv" / "Scripts" / "python.exe"
    current_python = Path(sys.executable).resolve()
    if venv_python.is_file() and current_python != venv_python.resolve():
        os.execv(str(venv_python), [str(venv_python), str(py_path), *sys.argv[1:]])
    raise


# ============================================================
# 全局参数
# ============================================================

# 窗口
WIN_W, WIN_H = 1400, 850

# 粒子数量
N_CHAR_BODY = 120000
N_CHAR_BELLY = 18000
N_CHAR_ARM = 18000
N_CHAR_HAND = 6000
N_CHAR_LEG = 16000
N_CHAR_FOOT = 12000
N_CHAR_EYE = 6000
N_CHAR_MOUTH = 3000
N_CHAR_TEETH = 2000
N_CHAR = (
    N_CHAR_BODY + N_CHAR_BELLY + N_CHAR_ARM + N_CHAR_HAND + N_CHAR_LEG
    + N_CHAR_FOOT + N_CHAR_EYE + N_CHAR_MOUTH + N_CHAR_TEETH
)
N_STARS = 5000           # 星星

UP = np.array([0.0, 0.0, 1.0])

# 人物
CHAR_CZ = 135.0
CHAR_MERGE_END = 6.8

# 颜色均从参考三视图取样，不做色相偏移。
C_YELLOW_HIGHLIGHT = np.array([255, 232, 90], dtype=np.float32) / 255.0
C_YELLOW_MID = np.array([250, 201, 70], dtype=np.float32) / 255.0
C_YELLOW_SHADOW = np.array([223, 138, 31], dtype=np.float32) / 255.0
C_YELLOW_ARM = np.array([223, 109, 12], dtype=np.float32) / 255.0
C_BELLY_LIGHT = np.array([250, 236, 209], dtype=np.float32) / 255.0
C_BELLY_SHADOW = np.array([220, 192, 153], dtype=np.float32) / 255.0
C_EYE_LIGHT = np.array([207, 192, 88], dtype=np.float32) / 255.0
C_EYE_WHITE = np.array([255, 251, 227], dtype=np.float32) / 255.0
C_EYE_GREEN = np.array([94, 130, 55], dtype=np.float32) / 255.0
C_PUPIL = np.array([18, 1, 5], dtype=np.float32) / 255.0
C_HAND = np.array([86, 71, 32], dtype=np.float32) / 255.0
C_FOOT = np.array([164, 138, 87], dtype=np.float32) / 255.0
C_MOUTH = np.array([81, 22, 0], dtype=np.float32) / 255.0

# 相机
CAM_DIST = 500.0
CAM_H = 220.0

rng = np.random.default_rng(23)


# ============================================================
# 通用工具函数
# ============================================================

def hsv2rgb(h, s, v):
    h = np.asarray(h)
    s = np.asarray(s)
    v = np.asarray(v)
    i = np.floor(h * 6.0).astype(int)
    f = h * 6.0 - i
    p = v * (1.0 - s)
    q = v * (1.0 - f * s)
    t = v * (1.0 - (1.0 - f) * s)
    i = i % 6
    r = np.choose(i, [v, q, p, p, t, v])
    g = np.choose(i, [t, v, v, q, p, p])
    b = np.choose(i, [p, p, t, v, v, q])
    return np.stack([r, g, b], 1)


def sstep(a, b, x):
    """平滑阶跃（smoothstep）。"""
    t = np.clip((x - a) / (b - a), 0.0, 1.0)
    return t * t * (3.0 - 2.0 * t)


# ============================================================
# 人物建模与初始化
# ============================================================

def mix_rgb(a, b, amount):
    amount = np.asarray(amount, dtype=np.float32)
    return a + (b - a) * amount[..., None]


def color_ramp(low, middle, high, amount):
    amount = np.clip(np.asarray(amount, dtype=np.float32), 0.0, 1.0)
    first = mix_rgb(low, middle, np.minimum(amount / 0.62, 1.0))
    second = mix_rgb(middle, high, np.maximum(0.0, (amount - 0.62) / 0.38))
    return np.where((amount < 0.62)[..., None], first, second)


BODY_PROFILE_V = np.array(
    [
        0.0, 0.08, 0.20, 0.38, 0.54, 0.63, 0.70,
        0.75, 0.80, 0.86, 0.90, 0.94, 0.975, 1.0,
    ]
)
BODY_PROFILE_RX = np.array(
    [
        0.0, 35.0, 58.0, 70.0, 69.0, 61.0, 50.0,
        43.0, 43.0, 45.0, 42.0, 33.0, 18.0, 0.0,
    ]
)


def body_radii(v):
    v = np.clip(np.asarray(v), 0.0, 1.0)
    rx = np.interp(v, BODY_PROFILE_V, BODY_PROFILE_RX)
    ry = rx * (0.68 + 0.06 * np.sin(np.pi * v))
    return rx, ry


def body_front_y(z):
    v = np.clip((np.asarray(z) + 55.0) / 160.0, 0.0, 1.0)
    return body_radii(v)[1]


def sample_ellipsoid(center, radii, count, rl):
    direction = rl.normal(0.0, 1.0, (count, 3))
    direction /= np.linalg.norm(direction, axis=1, keepdims=True)
    return np.asarray(center) + direction * np.asarray(radii)


def build_character(rl):
    points = []
    colors = []
    sizes = []
    brightness = []

    # 主体：梨形身体与头部连成一体。
    v = rl.random(N_CHAR_BODY)
    theta = rl.uniform(0.0, 2.0 * np.pi, N_CHAR_BODY)
    fill = rl.random(N_CHAR_BODY)
    radial = np.where(fill < 0.58, 1.0, np.cbrt(rl.random(N_CHAR_BODY)))
    rx, ry = body_radii(v)
    x = rx * radial * np.cos(theta)
    y = ry * radial * np.sin(theta)
    z = -55.0 + 160.0 * v
    x += rl.normal(0.0, 0.20, N_CHAR_BODY)
    y += rl.normal(0.0, 0.20, N_CHAR_BODY)
    z += rl.normal(0.0, 0.20, N_CHAR_BODY)

    frontness = np.clip(y / np.maximum(ry, 1e-6), -1.0, 1.0)
    body_mix = np.clip(0.18 + 0.50 * v + 0.30 * frontness, 0.0, 1.0)
    body_color = color_ramp(
        C_YELLOW_SHADOW,
        C_YELLOW_MID,
        C_YELLOW_HIGHLIGHT,
        body_mix,
    )

    # 腹部米白椭圆，边界来自参考图正视图。
    belly_radius = np.sqrt((x / 50.0) ** 2 + ((z + 8.0) / 42.0) ** 2)
    belly = (belly_radius < 1.0) & (y > -1.0)
    belly_color = mix_rgb(C_BELLY_SHADOW, C_BELLY_LIGHT, 1.0 - belly_radius)
    body_color[belly] = belly_color[belly]
    y[belly] += 0.45

    points.append(np.stack([x, y, z], axis=1))
    colors.append(body_color)
    body_size = 2.45 + 0.95 * rl.random(N_CHAR_BODY)
    body_size[belly] += 0.40
    sizes.append(body_size)
    body_bright = np.full(N_CHAR_BODY, 0.92, dtype=np.float32)
    body_bright[belly] = 0.98
    brightness.append(body_bright)

    # 独立的浅米色腹部粒子层，确保正面不会被身体内部粒子淹没。
    belly_theta = rl.uniform(0.0, 2.0 * np.pi, N_CHAR_BELLY)
    belly_radial = np.sqrt(rl.random(N_CHAR_BELLY))
    belly_x = 50.0 * belly_radial * np.cos(belly_theta)
    belly_z = -8.0 + 42.0 * belly_radial * np.sin(belly_theta)
    belly_y = body_front_y(belly_z) + 1.15 + rl.normal(0.0, 0.15, N_CHAR_BELLY)
    belly_distance = np.sqrt((belly_x / 50.0) ** 2 + ((belly_z + 8.0) / 42.0) ** 2)
    points.append(np.stack([belly_x, belly_y, belly_z], axis=1))
    colors.append(mix_rgb(C_BELLY_SHADOW, C_BELLY_LIGHT, 1.0 - belly_distance))
    sizes.append(2.55 + 0.90 * rl.random(N_CHAR_BELLY))
    brightness.append(np.full(N_CHAR_BELLY, 1.0, dtype=np.float32))

    # 捧腹姿势：肘部向外，前臂向腹部收拢。
    arm_per_side = N_CHAR_ARM // 2
    arm_parts = []
    arm_cols = []
    arm_front_y = float(body_front_y(-27.0))
    for side in (-1.0, 1.0):
        t = rl.random(arm_per_side)
        angle = rl.uniform(0.0, 2.0 * np.pi, arm_per_side)
        p0 = np.array([side * 50.0, 0.0, 23.0])
        p1 = np.array([side * 84.0, 3.0, -10.0])
        p2 = np.array([side * 41.0, arm_front_y + 9.0, -31.0])
        om = 1.0 - t
        center = (
            om[:, None] ** 2 * p0
            + 2.0 * om[:, None] * t[:, None] * p1
            + t[:, None] ** 2 * p2
        )
        tangent = (
            2.0 * om[:, None] * (p1 - p0)
            + 2.0 * t[:, None] * (p2 - p1)
        )
        tangent /= np.linalg.norm(tangent, axis=1, keepdims=True) + 1e-9
        basis_1 = np.cross(tangent, np.array([0.0, 1.0, 0.0]))
        basis_1 /= np.linalg.norm(basis_1, axis=1, keepdims=True) + 1e-9
        basis_2 = np.cross(tangent, basis_1)
        radius = 12.5 - 5.5 * t
        offset = (
            radius[:, None] * np.cos(angle)[:, None] * basis_1
            + radius[:, None] * 0.82 * np.sin(angle)[:, None] * basis_2
        )
        arm_parts.append(center + offset)
        surface_light = 0.5 + 0.5 * np.cos(angle)
        arm_mix = np.clip(0.48 + 0.28 * surface_light + 0.10 * t, 0.0, 1.0)
        arm_cols.append(mix_rgb(C_YELLOW_ARM, C_YELLOW_MID, arm_mix))
    points.append(np.concatenate(arm_parts, axis=0))
    colors.append(np.concatenate(arm_cols, axis=0))
    sizes.append(2.20 + 0.85 * rl.random(N_CHAR_ARM))
    brightness.append(np.full(N_CHAR_ARM, 0.88, dtype=np.float32))

    # 双手贴在腹部两侧，保留三根短手指。
    hand_parts = []
    hand_sizes = []
    palm_count = 900
    finger_count = 700
    for side in (-1.0, 1.0):
        palm = sample_ellipsoid(
            [side * 41.0, arm_front_y + 12.0, -31.0],
            [10.0, 5.8, 11.5],
            palm_count,
            rl,
        )
        hand_parts.append(palm)
        hand_sizes.append(np.full(palm_count, 2.25))
        for x_offset in (-4.5, 0.0, 4.5):
            finger = sample_ellipsoid(
                [side * 41.0 + x_offset, arm_front_y + 14.0, -41.0],
                [2.8, 3.3, 5.4],
                finger_count,
                rl,
            )
            hand_parts.append(finger)
            hand_sizes.append(np.full(finger_count, 1.95))
    hand_points = np.concatenate(hand_parts, axis=0)[:N_CHAR_HAND]
    points.append(hand_points)
    hand_shade = np.clip((-hand_points[:, 2] + 4.0) / 18.0, 0.0, 1.0)
    colors.append(mix_rgb(C_FOOT, C_HAND, hand_shade))
    sizes.append(np.concatenate(hand_sizes)[:N_CHAR_HAND])
    brightness.append(np.full(N_CHAR_HAND, 1.0, dtype=np.float32))

    # 双腿从梨形身体下方伸出。
    leg_per_side = N_CHAR_LEG // 2
    leg_parts = []
    leg_cols = []
    for side in (-1.0, 1.0):
        t = rl.random(leg_per_side)
        angle = rl.uniform(0.0, 2.0 * np.pi, leg_per_side)
        radius = 12.5 - 5.5 * t
        cx = side * (29.0 + 14.0 * t)
        cz = -43.0 - 67.0 * t
        px = cx + radius * np.cos(angle)
        py = radius * 0.82 * np.sin(angle)
        pz = cz + rl.normal(0.0, 0.18, leg_per_side)
        leg_parts.append(np.stack([px, py, pz], axis=1))
        surface_light = 0.5 + 0.5 * np.cos(angle)
        leg_mix = np.clip(0.25 + 0.35 * surface_light + 0.18 * t, 0.0, 1.0)
        leg_cols.append(mix_rgb(C_YELLOW_SHADOW, C_YELLOW_MID, leg_mix))
    points.append(np.concatenate(leg_parts, axis=0))
    colors.append(np.concatenate(leg_cols, axis=0))
    sizes.append(2.20 + 0.80 * rl.random(N_CHAR_LEG))
    brightness.append(np.full(N_CHAR_LEG, 0.88, dtype=np.float32))

    # 宽脚掌与三枚脚趾。
    foot_parts = []
    foot_sizes = []
    per_side_total = N_CHAR_FOOT // 2
    foot_count = 1920
    toe_count = (per_side_total - foot_count) // 3
    for side in (-1.0, 1.0):
        foot_parts.append(
            sample_ellipsoid([side * 43.0, 13.0, -112.0], [17.0, 25.0, 6.0], foot_count, rl)
        )
        foot_sizes.append(np.full(foot_count, 2.35))
        for x_offset in (-10.0, 0.0, 10.0):
            foot_parts.append(
                sample_ellipsoid(
                    [side * 43.0 + x_offset, 36.0, -112.0],
                    [5.0, 7.0, 3.2],
                    toe_count,
                    rl,
                )
            )
            foot_sizes.append(np.full(toe_count, 2.05))
    foot_points = np.concatenate(foot_parts, axis=0)[:N_CHAR_FOOT]
    points.append(foot_points)
    foot_mix = np.clip((foot_points[:, 1] + 10.0) / 50.0, 0.0, 1.0)
    colors.append(mix_rgb(C_HAND, C_FOOT, foot_mix))
    sizes.append(np.concatenate(foot_sizes)[:N_CHAR_FOOT])
    brightness.append(np.full(N_CHAR_FOOT, 1.0, dtype=np.float32))

    # 双层绿色眼睛、黑色瞳孔和高光。
    eye_parts = []
    eye_cols = []
    eye_sizes = []
    eye_bright = []
    per_eye = N_CHAR_EYE // 2
    layer_counts = (1700, 900, 350, 50)
    for eye_x in (-19.0, 19.0):
        z_center = 70.0
        front_y = float(body_front_y(z_center))
        colors_layer = np.empty((per_eye, 3), dtype=np.float32)
        sizes_layer = np.empty(per_eye, dtype=np.float32)
        bright_layer = np.empty(per_eye, dtype=np.float32)
        start = 0
        for count, radius_band, color, size, bright in zip(
            layer_counts,
            ((0.68, 1.0), (0.39, 0.68), (0.0, 0.39), (0.0, 0.11)),
            (C_EYE_LIGHT, C_EYE_GREEN, C_PUPIL, C_EYE_WHITE),
            (2.45, 2.50, 2.50, 2.65),
            (0.96, 1.0, 1.0, 1.0),
        ):
            stop = start + count
            low, high = radius_band
            rho = 12.8 * np.sqrt(
                low * low + rl.random(count) * (high * high - low * low)
            )
            if high <= 0.11:
                angle = rl.uniform(0.0, 2.0 * np.pi, count)
                rho *= 0.20
                ex = eye_x - 2.4 + rho * np.cos(angle)
                ez = z_center + 2.0 + rho * np.sin(angle)
            else:
                angle = rl.uniform(0.0, 2.0 * np.pi, count)
                ex = eye_x + 0.92 * rho * np.cos(angle)
                ez = z_center + 1.12 * rho * np.sin(angle)
            ey = front_y + 1.1 + 0.45 * max(0.0, 1.0 - high)
            ey = ey + 0.004 * np.maximum(0.0, 12.8 ** 2 - rho ** 2)
            eye_parts.append(np.stack([ex, np.full(count, ey), ez], axis=1))
            colors_layer[start:stop] = color
            sizes_layer[start:stop] = size
            bright_layer[start:stop] = bright
            start = stop
        eye_cols.append(colors_layer)
        eye_sizes.append(sizes_layer)
        eye_bright.append(bright_layer)
    points.extend(eye_parts)
    colors.extend(eye_cols)
    sizes.extend(eye_sizes)
    brightness.extend(eye_bright)

    # 大笑张开的深棕色口腔。
    mouth_angle = rl.uniform(0.0, 2.0 * np.pi, N_CHAR_MOUTH)
    mouth_radius = np.sqrt(rl.random(N_CHAR_MOUTH))
    mouth_x = 18.0 * mouth_radius * np.cos(mouth_angle)
    mouth_z = 49.0 + 10.0 * mouth_radius * np.sin(mouth_angle)
    mouth_y = body_front_y(mouth_z) + 3.2 + rl.normal(0.0, 0.08, N_CHAR_MOUTH)
    points.append(np.stack([mouth_x, mouth_y, mouth_z], axis=1))
    colors.append(np.tile(C_MOUTH, (N_CHAR_MOUTH, 1)))
    sizes.append(np.full(N_CHAR_MOUTH, 2.75, dtype=np.float32))
    brightness.append(np.full(N_CHAR_MOUTH, 1.0, dtype=np.float32))

    # 视频大笑动作中的上下牙齿。
    teeth_half = N_CHAR_TEETH // 2
    tooth_x = np.concatenate(
        [rl.uniform(-15.0, 15.0, teeth_half), rl.uniform(-15.0, 15.0, teeth_half)]
    )
    upper = np.arange(N_CHAR_TEETH) < teeth_half
    tooth_z = np.where(
        upper,
        54.3 + 0.6 * np.cos(tooth_x / 15.0 * np.pi),
        43.8 - 0.6 * np.cos(tooth_x / 15.0 * np.pi),
    )
    tooth_z += rl.normal(0.0, 0.18, N_CHAR_TEETH)
    tooth_y = body_front_y(tooth_z) + 4.6 + rl.normal(0.0, 0.06, N_CHAR_TEETH)
    points.append(np.stack([tooth_x, tooth_y, tooth_z], axis=1))
    colors.append(np.tile(C_EYE_WHITE, (N_CHAR_TEETH, 1)))
    sizes.append(np.full(N_CHAR_TEETH, 2.55, dtype=np.float32))
    brightness.append(np.full(N_CHAR_TEETH, 1.0, dtype=np.float32))

    part_groups = np.concatenate(
        [
            np.full(len(points[0]), 0, dtype=np.int8),  # body
            np.full(len(points[1]), 1, dtype=np.int8),  # belly
            np.full(len(points[2]), 2, dtype=np.int8),  # arms
            np.full(len(points[3]), 3, dtype=np.int8),  # hands
            np.full(len(points[4]), 4, dtype=np.int8),  # legs
            np.full(len(points[5]), 5, dtype=np.int8),  # feet
            np.full(sum(len(part) for part in eye_parts), 6, dtype=np.int8),
            np.full(len(points[-2]), 7, dtype=np.int8),  # mouth
            np.full(len(points[-1]), 8, dtype=np.int8),  # teeth
        ]
    )
    return (
        np.concatenate(points, axis=0),
        np.concatenate(colors, axis=0),
        np.concatenate(sizes, axis=0),
        np.concatenate(brightness, axis=0),
        part_groups,
    )


def init_character():
    rl = np.random.default_rng(7)
    local_points, colors, sizes, brightness, part_groups = build_character(rl)
    assert len(local_points) == N_CHAR

    # 粒子从散开的星尘聚合成人物。
    rb = 95.0 * np.sqrt(rl.random(N_CHAR))
    theta = rl.uniform(0.0, 2.0 * np.pi, N_CHAR)
    start_x = rb * np.cos(theta)
    start_y = rb * np.sin(theta)
    start_z = CHAR_CZ + rl.normal(0.0, 28.0, N_CHAR)
    depart_t = 2.7 + 0.9 * rl.random(N_CHAR)
    travel_t = 2.6 + 0.5 * rl.random(N_CHAR)

    return dict(
        local_x=local_points[:, 0],
        local_y=local_points[:, 1],
        local_z=local_points[:, 2],
        colors=colors,
        sizes=sizes,
        brightness=brightness,
        part_groups=part_groups,
        start_x=start_x,
        start_y=start_y,
        start_z=start_z,
        peak_height=42.0 + 34.0 * np.exp(-rb / 80.0),
        depart_t=depart_t,
        travel_t=travel_t,
        breathe_phase=rl.uniform(0.0, 2.0 * math.pi, N_CHAR),
    )


# ============================================================
# 星空
# ============================================================

def init_stars():
    n = N_STARS
    nf = int(n * 0.55)
    nm = int(n * 0.30)
    nn = n - nf - nm
    parts = []

    # 远景
    d = rng.normal(size=(nf, 3))
    d /= np.linalg.norm(d, axis=1, keepdims=True)
    r = rng.uniform(1050.0, 1500.0, nf)
    parts.append(dict(
        world=d * r[:, None],
        col=hsv2rgb(rng.uniform(0.55, 0.72, nf), rng.uniform(0.05, 0.35, nf),
                    rng.uniform(0.45, 0.75, nf)),
        size=rng.uniform(1.4, 3.0, nf), bright=rng.uniform(0.10, 0.22, nf),
        tw_speed=rng.uniform(0.2, 1.0, nf),
        tw_phase=rng.uniform(0.0, 2.0 * math.pi, nf),
    ))

    # 中景
    d = rng.normal(size=(nm, 3))
    d /= np.linalg.norm(d, axis=1, keepdims=True)
    r = rng.uniform(675.0, 1125.0, nm)
    parts.append(dict(
        world=d * r[:, None],
        col=hsv2rgb(rng.uniform(0.72, 0.95, nm), rng.uniform(0.30, 0.65, nm),
                    rng.uniform(0.55, 0.85, nm)),
        size=rng.uniform(2.7, 4.8, nm), bright=rng.uniform(0.22, 0.45, nm),
        tw_speed=rng.uniform(0.3, 1.5, nm),
        tw_phase=rng.uniform(0.0, 2.0 * math.pi, nm),
    ))

    # 近景
    d = rng.normal(size=(nn, 3))
    d /= np.linalg.norm(d, axis=1, keepdims=True)
    r = rng.uniform(420.0, 750.0, nn)
    sn = rng.uniform(3.8, 7.5, nn)
    bn = rng.uniform(0.35, 0.75, nn)
    neb = rng.random(nn) < 0.06
    sn = np.where(neb, rng.uniform(12.0, 24.0, nn), sn)
    bn = np.where(neb, rng.uniform(0.06, 0.18, nn), bn)
    parts.append(dict(
        world=d * r[:, None],
        col=hsv2rgb(rng.uniform(0.80, 0.99, nn), rng.uniform(0.40, 0.85, nn),
                    rng.uniform(0.70, 1.00, nn)),
        size=sn, bright=bn,
        tw_speed=rng.uniform(0.4, 1.8, nn),
        tw_phase=rng.uniform(0.0, 2.0 * math.pi, nn),
    ))

    w = np.concatenate([p['world'] for p in parts], axis=0)
    w[:, 2] += 120.0
    return dict(
        world=w,
        col=np.concatenate([p['col'] for p in parts], axis=0),
        size=np.concatenate([p['size'] for p in parts], axis=0),
        bright=np.concatenate([p['bright'] for p in parts], axis=0),
        tw_speed=np.concatenate([p['tw_speed'] for p in parts], axis=0),
        tw_phase=np.concatenate([p['tw_phase'] for p in parts], axis=0),
    )


# ============================================================
# 每帧更新
# ============================================================

def update_stars(t, s, ps):
    tw = 0.65 + 0.35 * np.sin(t * s['tw_speed'] + s['tw_phase'])
    return s['world'], s['col'], s['bright'] * tw, s['size'] * ps


def update_character(t, b, ps):
    # 固定正面，通过挤压、弹跳和手臂摆动表现捧腹大笑。
    laugh_time = max(0.0, t - CHAR_MERGE_END)
    laugh = 0.5 - 0.5 * math.cos(laugh_time * 5.2)
    laugh_small = 0.5 + 0.5 * math.sin(laugh_time * 4.1)

    local_x = b["local_x"].copy()
    local_y = b["local_y"].copy()
    local_z = b["local_z"].copy()

    arm_mask = (b["part_groups"] == 2) | (b["part_groups"] == 3)
    side_sign = np.where(local_x[arm_mask] < 0.0, -1.0, 1.0)
    shoulder_x = side_sign * 50.0
    arm_angle = side_sign * (0.045 + 0.105 * laugh_small)
    dx = local_x[arm_mask] - shoulder_x
    dz = local_z[arm_mask] - 23.0
    c_arm, s_arm = np.cos(arm_angle), np.sin(arm_angle)
    local_x[arm_mask] = shoulder_x + dx * c_arm - dz * s_arm
    local_z[arm_mask] = 23.0 + dx * s_arm + dz * c_arm

    eye_mask = b["part_groups"] == 6
    eye_open = 0.24 + 0.76 * (1.0 - laugh)
    local_z[eye_mask] = 70.0 + (local_z[eye_mask] - 70.0) * eye_open

    mouth_mask = (b["part_groups"] == 7) | (b["part_groups"] == 8)
    mouth_scale = 0.72 + 0.68 * laugh
    local_z[mouth_mask] = 48.0 + (local_z[mouth_mask] - 48.0) * mouth_scale

    roll = 0.045 * math.sin(laugh_time * 2.7)
    c_roll, s_roll = math.cos(roll), math.sin(roll)
    roll_x = local_x * c_roll - (local_z + 112.0) * s_roll
    roll_z = -112.0 + local_x * s_roll + (local_z + 112.0) * c_roll

    squash_x = 1.0 + 0.035 * laugh
    squash_z = 1.0 - 0.026 * laugh
    sway = 3.2 * math.sin(laugh_time * 2.7)
    target_x = local_y * (1.0 + 0.014 * laugh)
    target_y = -roll_x * squash_x + sway
    target_z = CHAR_CZ + roll_z * squash_z + 4.8 * laugh

    # 星尘聚合入场。
    pr = (t - b['depart_t']) / b['travel_t']
    pc = np.clip(pr, 0.0, 1.0)
    p = pc * pc * (3.0 - 2.0 * pc)
    cb = 0.25
    p1x = b['start_x'] * (1.0 - cb) + target_x * cb
    p1y = b['start_y'] * (1.0 - cb) + target_y * cb
    p1z = b['start_z'] + b['peak_height']
    om = 1.0 - p
    px = om * om * b['start_x'] + 2 * om * p * p1x + p * p * target_x
    py = om * om * b['start_y'] + 2 * om * p * p1y + p * p * target_y
    pz = om * om * b['start_z'] + 2 * om * p * p1z + p * p * target_z

    alpha = sstep(0.0, 0.12, pc)
    shimmer = 0.97 + 0.03 * np.sin(t * 2.1 + b["breathe_phase"])
    br = b["brightness"] * alpha * shimmer
    sz = b["sizes"] * ps
    world = np.stack([px, py, pz], axis=-1)
    return world, b["colors"], br, sz


# ============================================================
# 着色器
# ============================================================

VS = """
#version 330
in vec3 in_pos;
in vec3 in_col;
in float in_size;
in float in_bright;
uniform mat4 mvp;
out vec3 v_col;
out float v_bright;
void main() {
    gl_Position = mvp * vec4(in_pos, 1.0);
    v_col = in_col;
    v_bright = in_bright;
    gl_PointSize = clamp(in_size, 1.0, 120.0);
}
"""

FS = """
#version 330
in vec3 v_col;
in float v_bright;
out vec4 f_color;
uniform float point_mode;
void main() {
    vec2 d = gl_PointCoord - 0.5;
    float r2 = dot(d, d) * 4.0;
    float core = exp(-r2 * 5.0);
    float halo = exp(-r2 * 1.5) * 0.4;
    float mask = smoothstep(1.0, 0.86, r2);
    float intensity = (core + halo) * mask;
    if (intensity < 0.015) discard;
    if (point_mode < 0.5) {
        f_color = vec4(v_col * intensity * v_bright, 1.0);
    } else {
        f_color = vec4(v_col, clamp(intensity * v_bright, 0.0, 1.0));
    }
}
"""


# ============================================================
# 矩阵
# ============================================================

def persp(fov, asp, n, f):
    t = 1.0 / math.tan(fov / 2.0)
    m = np.zeros((4, 4), dtype='f4')
    m[0, 0] = t / asp
    m[1, 1] = t
    m[2, 2] = (f + n) / (n - f)
    m[2, 3] = 2.0 * f * n / (n - f)
    m[3, 2] = -1.0
    return m


def look_at(e, t, u):
    e = np.asarray(e, dtype='f4')
    t = np.asarray(t, dtype='f4')
    u = np.asarray(u, dtype='f4')
    f = t - e
    f /= np.linalg.norm(f)
    s = np.cross(f, u)
    s /= np.linalg.norm(s)
    up = np.cross(s, f)
    m = np.eye(4, dtype='f4')
    m[0, :3] = s
    m[1, :3] = up
    m[2, :3] = -f
    m[0, 3] = -s.dot(e)
    m[1, 3] = -up.dot(e)
    m[2, 3] = f.dot(e)
    return m


# ============================================================
# 主程序
# ============================================================

def _glfw_error_callback(code, description):
    print(f"GLFW 错误 {code}: {description}", file=sys.stderr)


def main():
    glfw.set_error_callback(_glfw_error_callback)
    if not glfw.init():
        raise RuntimeError("GLFW 初始化失败，请检查图形显示环境。")

    win = None
    try:
        glfw.window_hint(glfw.CONTEXT_VERSION_MAJOR, 3)
        glfw.window_hint(glfw.CONTEXT_VERSION_MINOR, 3)
        glfw.window_hint(glfw.OPENGL_PROFILE, glfw.OPENGL_CORE_PROFILE)
        if sys.platform == "darwin":
            glfw.window_hint(glfw.OPENGL_FORWARD_COMPAT, glfw.TRUE)

        win = glfw.create_window(WIN_W, WIN_H, "黄色小精灵", None, None)
        if win is None:
            raise RuntimeError("无法创建 OpenGL 窗口，请确认显卡驱动支持 OpenGL 3.3。")

        glfw.make_context_current(win)
        glfw.swap_interval(1)

        ctx = moderngl.create_context()
        ctx.enable(moderngl.BLEND)
        ctx.disable(moderngl.DEPTH_TEST)
        ctx.enable(moderngl.PROGRAM_POINT_SIZE)
        ctx.blend_func = moderngl.ONE, moderngl.ONE

        character = init_character()
        stars = init_stars()

        n_c, n_s = N_CHAR, N_STARS
        n_total = n_c + n_s
        print(f"总粒子: {n_total}  (人物 {n_c} + 星星 {n_s})")

        prog = ctx.program(vertex_shader=VS, fragment_shader=FS)
        vbo = ctx.buffer(reserve=n_total * 8 * 4)
        va = ctx.vertex_array(
            prog,
            [(vbo, "3f 3f 1f 1f", "in_pos", "in_col", "in_size", "in_bright")],
        )
        frame = np.empty((n_total, 8), dtype="f4")

        while not glfw.window_should_close(win):
            if glfw.get_key(win, glfw.KEY_ESCAPE) == glfw.PRESS:
                break

            t = glfw.get_time()
            fw, fh = glfw.get_framebuffer_size(win)
            if fw <= 0 or fh <= 0:
                glfw.wait_events_timeout(0.05)
                continue

            # Keep the OpenGL viewport synchronized after window resizing.
            ctx.viewport = (0, 0, fw, fh)
            ps = fh / float(WIN_H)

            bw_r = update_character(t, character, ps)
            sw_r = update_stars(t, stars, ps)

            i = 0
            for w, c, b, s in (bw_r, sw_r):
                n = w.shape[0]
                frame[i:i + n, 0:3] = w
                frame[i:i + n, 3:6] = c
                frame[i:i + n, 6] = s
                frame[i:i + n, 7] = b
                i += n

            vbo.write(frame.tobytes())
            ctx.clear(0.004, 0.002, 0.006, depth=1.0)

            # 固定正面视角，让捧腹大笑动作始终朝向观众。
            dist_mod = CAM_DIST + 18.0 * math.sin(t * 0.075 + 1.3)
            eye = np.array([
                dist_mod,
                0.0,
                CAM_H + 27.0 * math.sin(t * 0.11),
            ])
            tgt = np.array([0.0, 0.0, 100.0])
            asp = fw / float(fh)
            mvp = persp(math.radians(45.0), asp, 0.5, 3000.0) @ look_at(eye, tgt, UP)

            prog["mvp"].write(mvp.astype("f4").tobytes(order="F"))
            # Stars retain the original additive glow.
            prog["point_mode"].value = 0.0
            ctx.blend_func = moderngl.ONE, moderngl.ONE
            va.render(moderngl.POINTS, n_s, n_c)

            # The character uses point-center depth so the far side does not
            # show through its face and belly.
            prog["point_mode"].value = 1.0
            ctx.blend_func = moderngl.SRC_ALPHA, moderngl.ONE_MINUS_SRC_ALPHA
            ctx.enable(moderngl.DEPTH_TEST)
            va.render(moderngl.POINTS, n_c, 0)
            ctx.disable(moderngl.DEPTH_TEST)

            glfw.swap_buffers(win)
            glfw.poll_events()
    finally:
        if win is not None:
            glfw.destroy_window(win)
        glfw.terminate()


if __name__ == "__main__":
    main()
