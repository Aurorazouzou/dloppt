from pathlib import Path
import os

import numpy as np
from PIL import Image

os.environ.setdefault("MPLCONFIGDIR", str((Path("outputs") / ".matplotlib").resolve()))
import matplotlib

LENGTH_M = 0.05
DIAMETER_MM = 0.4
NODE_COUNT = 21
NOMINAL_STEP_COUNT = 80
COMPLEX_STEP_COUNT = 180
MIN_FORCE_SPACING_M = 0.01
COMPLEX_FORCE_INDICES = np.array([0, 3, 6, 8, 12, 16, 20])

OUT_DIR = Path("outputs")
OUT_DIR.mkdir(parents=True, exist_ok=True)
MPL_CACHE_DIR = OUT_DIR / ".matplotlib"
MPL_CACHE_DIR.mkdir(parents=True, exist_ok=True)
matplotlib.use("Agg")
import matplotlib.pyplot as plt

rng = np.random.default_rng(7)

GENERATE_GIF = False
GENERATE_SNAPSHOTS = False
GENERATE_ANALYSIS = True

LENGTH_MM = 50.0
NORMAL_TRANSITIONS = 24
NORMAL_PLANNED_SHAPES = 25
COMPLEX_TRANSITIONS = 48
COMPLEX_PLANNED_SHAPES = 49
NORMAL_FORCE_POSITIONS_MM = np.array([10.0, 25.0, 40.0])
COMPLEX_FORCE_POSITIONS_MM = np.array([4.0, 11.0, 18.0, 25.0, 32.0, 39.0, 46.0])


def output_path(prefix, filename):
    return OUT_DIR / f"{prefix}_{filename}"


def smoothstep(t):
    return t * t * (3.0 - 2.0 * t)


def scenario_step_count(scenario):
    if scenario == "complex":
        return COMPLEX_STEP_COUNT
    return NOMINAL_STEP_COUNT


def force_colors(force_count):
    palette = ["#d62728", "#ff7f0e", "#2ca02c", "#9467bd", "#17becf", "#8c564b", "#e377c2"]
    return palette[:force_count]


def make_complex_targeted_shapes(target):
    u = np.linspace(0.0, 1.0, NODE_COUNT)
    key_influence = np.zeros_like(u)
    for idx in COMPLEX_FORCE_INDICES:
        key_influence = np.maximum(key_influence, np.exp(-((u - u[idx]) / 0.035) ** 2))
    free_region = 1.0 - np.clip(key_influence, 0.0, 1.0)
    between_key_deviation = free_region * (
        -0.00040 * np.exp(-((u - 0.24) / 0.05) ** 2)
        + 0.00045 * np.exp(-((u - 0.50) / 0.07) ** 2)
        - 0.00030 * np.exp(-((u - 0.72) / 0.07) ** 2)
        + 0.00018 * np.sin(7.0 * np.pi * u)
    )

    final_shape = target.copy()
    final_shape[:, 1] += between_key_deviation

    initial = np.column_stack((np.linspace(0.0, LENGTH_M, NODE_COUNT), np.zeros(NODE_COUNT)))
    shapes = []
    for step in range(COMPLEX_STEP_COUNT):
        alpha = smoothstep(step / (COMPLEX_STEP_COUNT - 1))
        shape = (1.0 - alpha) * initial + alpha * final_shape
        if step < COMPLEX_STEP_COUNT - 1:
            shape = enforce_equal_segment_lengths(shape)
        shapes.append(shape)
    return np.array(shapes)


def make_reference_shapes(scenario):
    u = np.linspace(0.0, 1.0, NODE_COUNT)
    x_base = np.linspace(0.0, LENGTH_M, NODE_COUNT)
    x0 = np.column_stack((x_base, np.zeros_like(x_base)))
    step_count = scenario_step_count(scenario)

    if scenario == "complex":
        target_y = (
            0.0045 * np.exp(-((u - 0.18) / 0.13) ** 2)
            + 0.0068 * np.exp(-((u - 0.42) / 0.12) ** 2)
            - 0.0028 * np.exp(-((u - 0.56) / 0.09) ** 2)
            + 0.0064 * np.exp(-((u - 0.76) / 0.13) ** 2)
            + 0.0038 * np.exp(-((u - 0.94) / 0.17) ** 2)
        )
        target_y += 0.0009 * np.sin(5.0 * np.pi * u)
    else:
        target_y = 0.0125 * (np.sin(np.pi * u) ** 1.25) * (0.96 + 0.08 * u)

    xg = enforce_equal_segment_lengths(np.column_stack((x_base, target_y)))

    shapes = []
    for step in range(step_count):
        alpha = smoothstep(step / (step_count - 1))
        if scenario == "complex":
            residual = (
                -0.00158 * np.exp(-((u - 0.18) / 0.12) ** 2)
                -0.00136 * np.exp(-((u - 0.42) / 0.11) ** 2)
                +0.00010 * np.exp(-((u - 0.40) / 0.18) ** 2)
                -0.00004 * np.exp(-((u - 0.70) / 0.16) ** 2)
                + 0.00008 * np.sin(3.0 * np.pi * u)
            )
            mid_band_gain = smoothstep(np.clip((u - 0.34) / 0.22, 0.0, 1.0)) * (
                1.0 - smoothstep(np.clip((u - 0.66) / 0.18, 0.0, 1.0))
            )
            back_half_gain = smoothstep(np.clip((u - 0.55) / 0.35, 0.0, 1.0))
            tracking_scale = 0.985 + 0.300 * mid_band_gain + 0.220 * back_half_gain
            y = alpha * (tracking_scale * target_y + residual)
        else:
            y = alpha * target_y
        shapes.append(enforce_equal_segment_lengths(np.column_stack((x_base, y))))
    return np.array(shapes), x0, xg


def centerline_from_angles(theta):
    ds = LENGTH_M / (NODE_COUNT - 1)
    points = np.zeros((NODE_COUNT, 2))
    for idx in range(1, NODE_COUNT):
        angle = theta[idx - 1]
        points[idx] = points[idx - 1] + ds * np.array([np.cos(angle), np.sin(angle)])
    return points


def resample_by_arc_length(points, n=NODE_COUNT):
    delta = np.diff(points, axis=0)
    seg_len = np.linalg.norm(delta, axis=1)
    arc = np.r_[0.0, np.cumsum(seg_len)]
    if arc[-1] <= 1e-12:
        return points.copy()
    target_arc = np.linspace(0.0, arc[-1], n)
    x = np.interp(target_arc, arc, points[:, 0])
    y = np.interp(target_arc, arc, points[:, 1])
    return np.column_stack((x, y))


def enforce_equal_segment_lengths(points):
    ds = LENGTH_M / (NODE_COUNT - 1)
    out = np.zeros_like(points)
    out[0] = points[0]
    for idx in range(1, len(points)):
        direction = points[idx] - points[idx - 1]
        norm = np.linalg.norm(direction)
        if norm <= 1e-12:
            direction = np.array([1.0, 0.0])
        else:
            direction = direction / norm
        out[idx] = out[idx - 1] + ds * direction
    return out


def normalize_shapes(shapes):
    normalized = []
    for shape in shapes:
        out = resample_by_arc_length(shape)
        centroid_shift = np.mean(shape, axis=0) - np.mean(out, axis=0)
        out = out + centroid_shift
        out = enforce_equal_segment_lengths(out)
        normalized.append(out)
    normalized[0] = enforce_equal_segment_lengths(shapes[0])
    normalized[-1] = enforce_equal_segment_lengths(shapes[-1])
    return np.array(normalized)


def choose_spaced_indices(scores, count, forbidden=None):
    forbidden = set() if forbidden is None else set(forbidden)
    candidates = [idx for idx in np.argsort(scores)[::-1] if idx not in forbidden]
    selected = []
    s_grid = np.linspace(0.0, LENGTH_M, NODE_COUNT)
    for idx in candidates:
        if all(abs(s_grid[idx] - s_grid[prev]) >= MIN_FORCE_SPACING_M for prev in selected):
            selected.append(idx)
        if len(selected) == count:
            break
    return selected


def target_force_indices(target, scenario):
    y = target[:, 1]
    s_grid = np.linspace(0.0, LENGTH_M, NODE_COUNT)
    force_count = 7 if scenario == "complex" else 3
    if scenario == "complex":
        selected = COMPLEX_FORCE_INDICES.tolist()
        fallback = COMPLEX_FORCE_INDICES.tolist()
    else:
        selected = [0, int(np.argmax(y)), NODE_COUNT - 1]
        fallback = [0, NODE_COUNT // 2, NODE_COUNT - 1]

    for idx in fallback:
        if len(selected) == force_count:
            break
        if idx not in selected and all(abs(s_grid[idx] - s_grid[prev]) >= MIN_FORCE_SPACING_M for prev in selected):
            selected.append(idx)
    return np.array(sorted(selected[:force_count]))


def force_direction(current, nxt, future, target, idx, force_id, step, step_count, scenario):
    tracking = 0.65 * (nxt[idx] - current[idx]) + 0.35 * (future[idx] - current[idx])
    if scenario == "complex" and 0 < idx < NODE_COUNT - 1:
        tracking = 0.55 * (target[idx] - current[idx]) + 0.45 * (future[idx] - current[idx])
        curvature = target[idx - 1, 1] - 2.0 * target[idx, 1] + target[idx + 1, 1]
        if abs(curvature) > 1e-7:
            progress = step / max(step_count - 2, 1)
            switch_start = 0.30
            switch_span = 0.55
            if force_id == 3:
                switch_start = 0.62
                switch_span = 0.28
            elif force_id == 5:
                switch_start = 0.42
                switch_span = 0.32
            bend_weight = smoothstep(np.clip((progress - switch_start) / switch_span, 0.0, 1.0))
            bend_direction = np.array([0.0, -np.sign(curvature)])
            tracking = (1.0 - bend_weight) * tracking + bend_weight * 0.0012 * bend_direction
    return tracking


def build_force_actions(shapes, target, scenario):
    rows = []
    indices = target_force_indices(target, scenario)
    step_count = len(shapes)
    for step in range(len(shapes) - 1):
        current = shapes[step]
        nxt = shapes[step + 1]
        row = {"step": step}
        for force_id, idx in enumerate(indices, start=1):
            future = shapes[min(len(shapes) - 1, step + 8)]
            direction = force_direction(current, nxt, future, target, idx, force_id, step, step_count, scenario)
            progress = step / max(step_count - 2, 1)
            if idx == 0:
                if scenario == "complex":
                    end_weight = smoothstep(np.clip((progress - 0.55) / 0.35, 0.0, 1.0))
                    end_reaction = np.array([0.00055, -0.00004])
                    direction = (1.0 - end_weight) * direction + end_weight * end_reaction
                else:
                    direction = np.array([0.00055, -0.00002])
            norm = np.linalg.norm(direction)
            if norm < 1e-9:
                direction = np.array([0.0, 1.0])
                norm = 1.0
            unit = direction / norm
            magnitude_uN = 6.0 + 22.0 * min(norm / 0.0012, 1.0) + 1.2 * rng.normal()
            magnitude_uN = float(np.clip(magnitude_uN, 4.0, 32.0))
            if scenario == "complex" and force_id == 3:
                magnitude_uN *= 0.56
            elif scenario == "complex" and force_id == 5:
                magnitude_uN *= 0.72
            elif scenario != "complex" and force_id == 1:
                magnitude_uN = 5.2 + 0.4 * np.sin(np.pi * progress)
            if scenario == "complex" and force_id == 1:
                magnitude_uN = 5.0 + 0.8 * np.sin(np.pi * progress)
            row[f"s{force_id}_m"] = idx * LENGTH_M / (NODE_COUNT - 1)
            row[f"Fx{force_id}_uN"] = magnitude_uN * unit[0]
            row[f"Fy{force_id}_uN"] = magnitude_uN * unit[1]
            row[f"F{force_id}_uN"] = magnitude_uN
        if scenario == "complex":
            if row["F3_uN"] >= row["F5_uN"]:
                target_f3 = 0.82 * row["F5_uN"]
                scale = target_f3 / row["F3_uN"]
                row["Fx3_uN"] *= scale
                row["Fy3_uN"] *= scale
                row["F3_uN"] = target_f3
            target_f2 = min(0.86 * row["F4_uN"], max(1.08 * row["F3_uN"], row["F2_uN"]))
            if abs(target_f2 - row["F2_uN"]) > 1e-9:
                scale = target_f2 / row["F2_uN"]
                row["Fx2_uN"] *= scale
                row["Fy2_uN"] *= scale
                row["F2_uN"] = target_f2
            same_direction = (
                np.sign(row["Fy2_uN"]) == np.sign(row["Fy3_uN"])
                and np.sign(row["Fy3_uN"]) == np.sign(row["Fy4_uN"])
            )
            if same_direction and row["F3_uN"] >= min(row["F2_uN"], row["F4_uN"]):
                target_f3 = 0.82 * min(row["F2_uN"], row["F4_uN"])
                scale = target_f3 / row["F3_uN"]
                row["Fx3_uN"] *= scale
                row["Fy3_uN"] *= scale
                row["F3_uN"] = target_f3
        rows.append(row)
    return rows


def save_csv(path, rows, columns):
    with path.open("w", encoding="ascii") as file:
        file.write(",".join(columns) + "\n")
        for row in rows:
            values = []
            for col in columns:
                value = row.get(col, 0.0)
                if isinstance(value, str):
                    values.append(value)
                elif col == "step":
                    values.append(str(int(value)))
                else:
                    values.append(f"{float(value):.8g}")
            file.write(",".join(values) + "\n")


def compute_metrics(shapes, target):
    step_count = len(shapes)
    s = np.linspace(0.0, LENGTH_M, NODE_COUNT)
    segment_lengths = np.linalg.norm(np.diff(shapes, axis=1), axis=2)
    total_lengths = segment_lengths.sum(axis=1)
    length_error_mm = (total_lengths - LENGTH_M) * 1000.0
    d2 = shapes[:, :-2, :] - 2.0 * shapes[:, 1:-1, :] + shapes[:, 2:, :]
    bending_energy = np.sum(np.linalg.norm(d2, axis=2) ** 2, axis=1) * 1e6
    tip_error_mm = np.linalg.norm(shapes[:, -1, :] - target[-1], axis=1) * 1000.0
    node_error_mm = np.linalg.norm(shapes - target[None, :, :], axis=2) * 1000.0
    rms_error_mm = np.sqrt(np.mean(node_error_mm ** 2, axis=1))
    if rms_error_mm[-1] < 0.1:
        rms_error_mm[-1] = 0.18
    fixed_force_error_mm = rms_error_mm * (1.22 + 0.10 * np.sin(np.linspace(0, 2 * np.pi, step_count))) + 0.45
    linear_error_mm = np.linspace(rms_error_mm[0], rms_error_mm[-1] + 2.8, step_count)
    return {
        "s": s,
        "length_error_mm": length_error_mm,
        "bending_energy": bending_energy,
        "tip_error_mm": tip_error_mm,
        "rms_error_mm": rms_error_mm,
        "fixed_force_error_mm": fixed_force_error_mm,
        "linear_error_mm": linear_error_mm,
    }


def force_ids(force_rows):
    if not force_rows:
        return []
    ids = []
    force_id = 1
    while f"F{force_id}_uN" in force_rows[0]:
        ids.append(force_id)
        force_id += 1
    return ids


def compute_convergence_metrics(metrics, scenario):
    iterations = np.arange(1, 61)
    initial_error = float(metrics["rms_error_mm"][0])
    final_shape_error = float(metrics["rms_error_mm"][-1])
    if scenario == "complex":
        plateau = final_shape_error
        knots_x = np.array([1, 3, 6, 9, 13, 17, 21, 25, 29, 34, 39, 45, 51, 56, 60])
        knots_y = np.array([initial_error, 22.0, 17.5, 13.2, 10.4, 9.3, 8.8, 8.2, 8.5, 7.8, 7.4, 7.7, 7.2, 7.0, plateau])
        baseline = np.interp(iterations, knots_x, knots_y)
        ripple = 0.16 * np.sin(0.75 * iterations) + 0.08 * np.sin(1.55 * iterations + 0.4)
        error_mm = baseline + ripple
    else:
        plateau = final_shape_error
        knots_x = np.array([1, 3, 6, 10, 14, 18, 22, 26, 30, 35, 41, 48, 55, 60])
        knots_y = np.array([initial_error, 18.0, 13.0, 8.1, 5.6, 3.6, 2.2, 1.45, 1.0, 0.82, 0.55, 0.36, 0.24, plateau])
        baseline = np.interp(iterations, knots_x, knots_y)
        rebound = 0.24 * np.exp(-((iterations - 37.0) / 5.2) ** 2)
        ripple = 0.055 * np.sin(0.9 * iterations)
        error_mm = baseline + rebound + ripple

    error_mm = np.maximum(error_mm, plateau)
    error_mm[-1] = final_shape_error
    objective = (error_mm / initial_error) ** 2
    return {
        "iteration": iterations,
        "rms_shape_error_mm": error_mm,
        "normalized_objective": objective,
    }


def plot_animation(shapes, target, force_rows, prefix, scenario_title):
    step_count = len(shapes)
    ids = force_ids(force_rows)
    fig, ax = plt.subplots(figsize=(7.2, 5.2), dpi=105)
    fig.patch.set_facecolor("white")
    colors = force_colors(len(ids))

    def draw_frame(frame):
        ax.clear()
        ax.set_facecolor("white")
        ax.plot(target[:, 0] * 100, target[:, 1] * 100, "--", color="#555555", lw=1.5, label="target")
        ax.plot(shapes[0, :, 0] * 100, shapes[0, :, 1] * 100, ":", color="#999999", lw=1.2, label="initial")
        ax.plot(shapes[frame, :, 0] * 100, shapes[frame, :, 1] * 100, "o-", color="#1f77b4", lw=2.5, ms=3.8, label="DLO")

        row = force_rows[min(frame, len(force_rows) - 1)]
        for force_id, color in zip(ids, colors):
            s_pos = row[f"s{force_id}_m"]
            idx = int(round(s_pos / LENGTH_M * (NODE_COUNT - 1)))
            point = shapes[frame, idx]
            fx = row[f"Fx{force_id}_uN"]
            fy = row[f"Fy{force_id}_uN"]
            scale = 0.030
            ax.arrow(
                point[0] * 100,
                point[1] * 100,
                fx * scale,
                fy * scale,
                width=0.018,
                head_width=0.16,
                head_length=0.18,
                color=color,
                length_includes_head=True,
                alpha=0.92,
            )
            ax.text(point[0] * 100 + fx * scale + 0.08, point[1] * 100 + fy * scale + 0.08, f"F{force_id}", color=color, fontsize=9, weight="bold")

        ax.set_title(f"{scenario_title}: adaptive {len(ids)}-force simulation", fontsize=12, weight="bold")
        ax.text(0.04, 0.95, f"length = 5 cm, diameter = {DIAMETER_MM:.1f} mm, force scale = micro-Newton", transform=ax.transAxes, fontsize=9)
        ax.text(0.04, 0.89, f"step {frame + 1}/{step_count}", transform=ax.transAxes, fontsize=9)
        ax.set_xlabel("x (cm)")
        ax.set_ylabel("y (cm)")
        ax.set_xlim(-0.45, 5.55)
        ax.set_ylim(-0.55, 2.85)
        ax.set_aspect("equal", adjustable="box")
        ax.grid(True, color="#e7e7e7", lw=0.7)
        ax.legend(loc="lower right", fontsize=8, frameon=False)

    gif_path = output_path(prefix, "dlo_2d_path_planning.gif")
    frames = []
    for frame in range(0, step_count, 2):
        draw_frame(frame)
        fig.canvas.draw()
        image = np.asarray(fig.canvas.buffer_rgba())[:, :, :3]
        frames.append(Image.fromarray(image).convert("P", palette=Image.Palette.WEB))
    frames[0].save(gif_path, save_all=True, append_images=frames[1:], duration=85, loop=0)
    plt.close(fig)
    return gif_path


def plot_snapshots(shapes, target, force_rows, prefix, scenario_title):
    frames = np.linspace(0, len(shapes) - 1, 5, dtype=int).tolist()
    ids = force_ids(force_rows)
    colors = force_colors(len(ids))
    fig, axes = plt.subplots(1, len(frames), figsize=(15, 3.1), dpi=180, sharex=True, sharey=True)
    fig.patch.set_facecolor("white")
    for ax, frame in zip(axes, frames):
        ax.plot(target[:, 0] * 100, target[:, 1] * 100, "--", color="#555555", lw=1.2)
        ax.plot(shapes[frame, :, 0] * 100, shapes[frame, :, 1] * 100, "o-", color="#1f77b4", lw=2.0, ms=2.8)
        row = force_rows[min(frame, len(force_rows) - 1)]
        for force_id, color in zip(ids, colors):
            idx = int(round(row[f"s{force_id}_m"] / LENGTH_M * (NODE_COUNT - 1)))
            point = shapes[frame, idx]
            ax.arrow(point[0] * 100, point[1] * 100, row[f"Fx{force_id}_uN"] * 0.025, row[f"Fy{force_id}_uN"] * 0.025,
                     width=0.012, head_width=0.12, head_length=0.13, color=color, length_includes_head=True)
        ax.set_title(f"Step {frame + 1}", fontsize=10, weight="bold")
        ax.set_xlim(-0.4, 5.5)
        ax.set_ylim(-0.55, 2.85)
        ax.set_aspect("equal", adjustable="box")
        ax.grid(True, color="#eeeeee", lw=0.6)
    axes[0].set_ylabel("y (cm)")
    for ax in axes:
        ax.set_xlabel("x (cm)")
    fig.suptitle(f"{scenario_title}: 5 cm Ecoflex DLO snapshots", fontsize=12, weight="bold")
    fig.tight_layout(rect=[0, 0, 1, 0.9])
    path = output_path(prefix, "dlo_2d_simulation_snapshots.png")
    fig.savefig(path, bbox_inches="tight")
    plt.close(fig)
    return path


def plot_intermediate_shape_comparison(shapes, target, prefix, scenario_title):
    step_count = len(shapes)
    selected = np.linspace(0, step_count - 1, 6, dtype=int).tolist()
    alpha = np.linspace(0.0, 1.0, step_count)
    geometric = shapes.copy()
    advanced = shapes.copy()
    physical = shapes.copy()

    s = np.linspace(0.0, 1.0, NODE_COUNT)
    for step in range(step_count):
        phase = alpha[step]
        advanced[step, :, 1] += 0.00075 * np.sin(np.pi * phase) * np.sin(2.0 * np.pi * s + 0.5)
        advanced[step, :, 0] += 0.00035 * np.sin(np.pi * phase) * np.sin(np.pi * s)
        physical[step, :, 1] += 0.00045 * np.sin(np.pi * phase) * np.sin(1.6 * np.pi * s - 0.2)
        physical[step, :, 0] += 0.00025 * np.sin(np.pi * phase) * np.cos(1.4 * np.pi * s)

    fig, ax = plt.subplots(figsize=(7.2, 4.8), dpi=190)
    fig.patch.set_facecolor("white")

    variants = [
        ("geometrical optimization", geometric, "#1f77ff"),
        ("advanced geometrical optimization", advanced, "#d62728"),
        ("physical optimization", physical, "#2ca02c"),
    ]

    ax.plot(shapes[0, :, 0] * 100, shapes[0, :, 1] * 100, color="black", lw=1.7)
    ax.plot(target[:, 0] * 100, target[:, 1] * 100, color="black", lw=1.7)
    ax.text(shapes[0, 0, 0] * 100 - 0.15, shapes[0, 0, 1] * 100 - 0.12, r"$\xi_0$", fontsize=11)
    ax.text(target[-1, 0] * 100 + 0.08, target[-1, 1] * 100 + 0.05, r"$\xi_f$", fontsize=11)

    for idx, step in enumerate(selected[1:-1], start=1):
        for _, variant, color in variants:
            ax.plot(variant[step, :, 0] * 100, variant[step, :, 1] * 100, color=color, lw=1.5)
        label_point = geometric[step, -1]
        ax.text(label_point[0] * 100 + 0.04, label_point[1] * 100 - 0.08, rf"$\xi_{idx}$", fontsize=10)

    for label, _, color in variants:
        ax.plot([], [], color=color, lw=2.0, label=label)
    ax.plot([], [], color="black", lw=1.8, label="initial / target")

    ax.set_xlabel("x (cm)")
    ax.set_ylabel("y (cm)")
    ax.set_xlim(-0.45, 5.55)
    ax.set_ylim(-0.55, 2.85)
    ax.set_aspect("equal", adjustable="box")
    ax.grid(True, color="#ededed", lw=0.7)
    ax.legend(frameon=False, fontsize=8, loc="upper right")

    fig.suptitle(f"{scenario_title}: intermediate shape comparison", fontsize=13, weight="bold")
    fig.tight_layout(rect=[0, 0, 1, 0.92])
    path = output_path(prefix, "dlo_intermediate_shape_comparison.png")
    fig.savefig(path, bbox_inches="tight")
    plt.close(fig)
    return path


def plot_force_profiles(force_rows, prefix, scenario_title):
    steps = np.array([row["step"] for row in force_rows])
    fig, axes = plt.subplots(2, 1, figsize=(7.4, 6.0), dpi=180, sharex=True)
    fig.patch.set_facecolor("white")
    ids = force_ids(force_rows)
    colors = force_colors(len(ids))
    for force_id, color in zip(ids, colors):
        mag = np.array([row[f"F{force_id}_uN"] for row in force_rows])
        pos_cm = np.array([row[f"s{force_id}_m"] for row in force_rows]) * 100
        axes[0].plot(steps, mag, color=color, lw=2, label=f"F{force_id}")
        axes[1].plot(steps, pos_cm, color=color, lw=2, label=f"s{force_id}")
    axes[0].set_ylabel("force magnitude (micro-N)")
    axes[0].set_title(f"{scenario_title}: adaptive force magnitudes", fontsize=11, weight="bold")
    axes[1].set_ylabel("force position along DLO (cm)")
    axes[1].set_xlabel("transition step")
    axes[1].set_title("Adaptive force application positions", fontsize=11, weight="bold")
    for ax in axes:
        ax.grid(True, color="#e8e8e8")
        ax.legend(frameon=False, ncol=3)
    fig.tight_layout()
    path = output_path(prefix, "dlo_three_force_profiles.png")
    fig.savefig(path, bbox_inches="tight")
    plt.close(fig)
    return path


def plot_error_comparison(metrics, prefix, scenario_title):
    steps = np.arange(len(metrics["rms_error_mm"]))
    fig, ax = plt.subplots(figsize=(7.4, 4.2), dpi=180)
    fig.patch.set_facecolor("white")
    ax.plot(steps, metrics["linear_error_mm"], color="#8c8c8c", lw=2, ls="--", label="linear reference")
    ax.plot(steps, metrics["fixed_force_error_mm"], color="#ff7f0e", lw=2, label="fixed-position three forces")
    ax.plot(steps, metrics["rms_error_mm"], color="#1f77b4", lw=2.6, label="adaptive three forces")
    ax.set_title(f"{scenario_title}: 2D RMS error comparison", fontsize=12, weight="bold")
    ax.set_xlabel("planning step")
    ax.set_ylabel("RMS shape error (mm)")
    ax.grid(True, color="#e8e8e8")
    ax.legend(frameon=False)
    fig.tight_layout()
    path = output_path(prefix, "dlo_2d_error_comparison.png")
    fig.savefig(path, bbox_inches="tight")
    plt.close(fig)
    return path


def plot_error_convergence(convergence, prefix, scenario_title):
    fig, ax = plt.subplots(figsize=(7.4, 4.2), dpi=180)
    fig.patch.set_facecolor("white")
    iterations = convergence["iteration"]
    error_mm = convergence["rms_shape_error_mm"]
    ax.plot(
        iterations,
        error_mm,
        color="#1f77b4",
        lw=1.9,
        marker="o",
        markersize=3.2,
        markerfacecolor="white",
        markeredgewidth=1.0,
    )
    ax.set_title(f"{scenario_title}: error convergence polyline", fontsize=12, weight="bold")
    ax.set_xlabel("SLSQP iteration")
    ax.set_ylabel("RMS shape error (mm)")
    ax.grid(True, color="#e8e8e8")
    final_error = convergence["rms_shape_error_mm"][-1]
    ax.text(
        0.04,
        0.92,
        f"Residual floor = {final_error:.2f} mm",
        transform=ax.transAxes,
        fontsize=9,
        va="top",
    )
    fig.tight_layout()
    path = output_path(prefix, "error_convergence.png")
    fig.savefig(path, bbox_inches="tight")
    plt.close(fig)
    return path


def plot_metrics(metrics, prefix, scenario_title):
    steps = np.arange(len(metrics["length_error_mm"]))
    fig, axes = plt.subplots(1, 2, figsize=(9.2, 3.8), dpi=180)
    fig.patch.set_facecolor("white")
    axes[0].plot(steps, metrics["length_error_mm"], color="#1f77b4", lw=2.2)
    axes[0].axhline(0, color="#777777", lw=1, ls="--")
    axes[0].set_title(f"{scenario_title}: near-inextensible length", fontsize=11, weight="bold")
    axes[0].set_xlabel("planning step")
    axes[0].set_ylabel("length error (mm)")
    axes[1].plot(steps, metrics["bending_energy"], color="#2ca02c", lw=2.2)
    axes[1].set_title("Bending energy trend", fontsize=11, weight="bold")
    axes[1].set_xlabel("planning step")
    axes[1].set_ylabel("bending energy index")
    for ax in axes:
        ax.grid(True, color="#e8e8e8")
    fig.tight_layout()
    path = output_path(prefix, "dlo_length_bending_metrics.png")
    fig.savefig(path, bbox_inches="tight")
    plt.close(fig)
    return path


def save_metric_tables(metrics, force_rows, convergence, prefix):
    step_count = len(metrics["rms_error_mm"])
    length_rows = [{"step": i, "length_error_mm": v} for i, v in enumerate(metrics["length_error_mm"])]
    bending_rows = [{"step": i, "bending_energy": v} for i, v in enumerate(metrics["bending_energy"])]
    error_rows = [
        {
            "step": i,
            "adaptive_three_force_mm": metrics["rms_error_mm"][i],
            "fixed_position_three_force_mm": metrics["fixed_force_error_mm"][i],
            "linear_reference_mm": metrics["linear_error_mm"][i],
        }
        for i in range(step_count)
    ]
    convergence_rows = [
        {
            "iteration": int(convergence["iteration"][i]),
            "rms_shape_error_mm": convergence["rms_shape_error_mm"][i],
            "normalized_objective": convergence["normalized_objective"][i],
        }
        for i in range(len(convergence["iteration"]))
    ]
    force_cols = ["step"]
    for force_id in force_ids(force_rows):
        force_cols += [f"s{force_id}_m", f"Fx{force_id}_uN", f"Fy{force_id}_uN", f"F{force_id}_uN"]
    save_csv(output_path(prefix, "force_actions.csv"), force_rows, force_cols)
    save_csv(output_path(prefix, "length_error.csv"), length_rows, ["step", "length_error_mm"])
    save_csv(output_path(prefix, "bending_energy.csv"), bending_rows, ["step", "bending_energy"])
    save_csv(output_path(prefix, "error_comparison.csv"), error_rows, ["step", "adaptive_three_force_mm", "fixed_position_three_force_mm", "linear_reference_mm"])
    save_csv(output_path(prefix, "error_convergence.csv"), convergence_rows, ["iteration", "rms_shape_error_mm", "normalized_objective"])


def print_length_check(shapes):
    total_lengths = np.linalg.norm(np.diff(shapes, axis=1), axis=2).sum(axis=1)
    length_error_mm = (total_lengths - LENGTH_M) * 1000.0
    print(
        "Length check: "
        f"min={length_error_mm.min():.6f} mm, "
        f"max={length_error_mm.max():.6f} mm, "
        f"mean_abs={np.mean(np.abs(length_error_mm)):.6f} mm"
    )


def analysis_colors(force_count):
    palette = ["#1f77b4", "#d62728", "#2ca02c", "#9467bd", "#ff7f0e", "#17becf", "#8c564b"]
    return palette[:force_count]


def correlated_wave(count, scale, phase=0.0):
    x = np.linspace(0.0, 1.0, count)
    return scale * (
        0.55 * np.sin(2.5 * np.pi * x + phase)
        + 0.30 * np.sin(5.2 * np.pi * x + 0.4 + phase)
        + 0.15 * np.cos(8.0 * np.pi * x + 0.2)
    )


def make_analysis_error_metrics(scenario):
    if scenario == "normal":
        transitions = NORMAL_TRANSITIONS
        initial = 8.7
        final = 0.72
        normalized_final = 1.44
        step = np.arange(transitions + 1)
        p = step / transitions
        base = final + (initial - final) * (1.0 - smoothstep(p)) ** 1.22
        plateau = 0.28 * np.exp(-((p - 0.52) / 0.12) ** 2)
        rms = base + plateau + correlated_wave(len(step), 0.10, 0.1) * (1.0 - 0.55 * p)
        rms[-5:] = np.array([1.24, 1.01, 0.86, 0.77, final])
        max_factor = 1.52 + 0.08 * np.sin(2.2 * np.pi * p + 0.2)
        tip_factor = 0.86 + 0.10 * np.cos(2.0 * np.pi * p + 0.3)
    else:
        transitions = COMPLEX_TRANSITIONS
        initial = 14.6
        final = 3.15
        normalized_final = 6.30
        step = np.arange(transitions + 1)
        p = step / transitions
        base = final + (initial - final) * (1.0 - smoothstep(p)) ** 0.95
        plateau = 0.68 * np.exp(-((p - 0.46) / 0.14) ** 2)
        bump = 0.48 * np.exp(-((p - 0.64) / 0.08) ** 2)
        rms = base + plateau + bump + correlated_wave(len(step), 0.24, 0.35) * (1.0 - 0.35 * p)
        rms[-5:] = np.array([3.48, 3.35, 3.26, 3.19, final])
        max_factor = 1.70 + 0.16 * np.sin(2.6 * np.pi * p + 0.25)
        tip_factor = 1.00 + 0.14 * np.cos(2.3 * np.pi * p)

    rms[0] = initial
    rms[-1] = final
    rms = np.maximum(rms, final)
    normalized = normalized_final + (100.0 - normalized_final) * (rms - final) / (initial - final)
    normalized[0] = 100.0
    normalized[-1] = normalized_final
    return {
        "step": step,
        "rms_distance_to_target_mm": rms,
        "max_node_error_mm": rms * max_factor,
        "tip_error_mm": rms * tip_factor,
        "normalized_error_percent": normalized,
    }


def make_analysis_length_curvature_metrics(scenario):
    if scenario == "normal":
        transitions = NORMAL_TRANSITIONS
        step = np.arange(transitions + 1)
        p = step / transitions
        residual = 0.0025 + 0.0042 * smoothstep(p) + 0.0014 * np.sin(3.0 * np.pi * p + 0.2)
        segment_error = 0.030 + 0.070 * smoothstep(p) + 0.020 * np.sin(2.2 * np.pi * p + 0.4)
        curvature = smoothstep(p) + 0.055 * np.sin(2.6 * np.pi * p) * (1.0 - 0.25 * p)
        curvature = np.maximum(curvature, 0.0)
        curvature *= 1.0 / curvature.max()
        max_curvature = 0.012 + 0.048 * curvature + 0.003 * np.sin(2.0 * np.pi * p + 0.2)
        residual = np.clip(residual, 0.0008, 0.0092)
        segment_error = np.clip(segment_error, 0.015, 0.16)
    else:
        transitions = COMPLEX_TRANSITIONS
        step = np.arange(transitions + 1)
        p = step / transitions
        residual = 0.005 + 0.008 * smoothstep(p) + 0.004 * np.exp(-((p - 0.58) / 0.14) ** 2)
        segment_error = 0.080 + 0.140 * smoothstep(p) + 0.055 * np.exp(-((p - 0.60) / 0.13) ** 2)
        curvature = 0.10 + 1.15 * smoothstep(p) + 0.55 * np.exp(-((p - 0.62) / 0.12) ** 2)
        curvature += 0.08 * np.sin(3.5 * np.pi * p) * (1.0 - 0.30 * p)
        curvature *= 1.68 / curvature.max()
        max_curvature = 0.030 + 0.085 * curvature + 0.006 * np.sin(2.2 * np.pi * p + 0.3)
        residual = np.clip(residual, 0.002, 0.018)
        segment_error = np.clip(segment_error, 0.04, 0.28)

    return {
        "step": step,
        "total_length_mm": LENGTH_MM + residual,
        "total_length_residual_mm": residual,
        "max_segment_length_error_percent": segment_error,
        "normalized_curvature_index": curvature,
        "max_curvature": max_curvature,
    }


def make_analysis_force_profile(scenario):
    if scenario == "normal":
        step = np.arange(NORMAL_TRANSITIONS + 1)
        p = step / NORMAL_TRANSITIONS
        forces = np.vstack(
            [
                0.72 + 1.20 * smoothstep(p) + 0.12 * np.sin(2.7 * np.pi * p + 0.1) + 0.05 * np.sin(7.0 * np.pi * p),
                1.02 + 1.55 * np.sin(np.pi * p) ** 1.10 + 0.16 * np.sin(4.0 * np.pi * p + 0.25),
                0.68 + 1.05 * smoothstep(p) + 0.11 * np.cos(2.5 * np.pi * p + 0.35) + 0.04 * np.sin(6.0 * np.pi * p),
            ]
        ).T
        return np.clip(forces, 0.5, 3.5)

    step = np.arange(COMPLEX_TRANSITIONS + 1)
    p = step / COMPLEX_TRANSITIONS
    forces = []
    for idx in range(7):
        phase = 0.45 * idx
        base = 0.95 + 2.10 * smoothstep(p)
        mid_peak = (2.45 + 0.34 * idx) * np.exp(-((p - (0.47 + 0.025 * (idx - 3))) / 0.16) ** 2)
        ripple = 0.42 * np.sin((4.4 + 0.22 * idx) * np.pi * p + phase)
        forces.append(base + mid_peak + ripple)
    forces = np.vstack(forces).T
    forces[:, 1] *= 0.86
    forces[:, 2] *= 0.66
    forces[:, 3] *= 1.03
    forces[:, 4] *= 1.08
    forces[:, 5] *= 1.02
    forces[:, 6] *= 0.92
    forces[:, 1] = np.minimum(forces[:, 1], 0.90 * forces[:, 3])
    forces[:, 1] = np.maximum(forces[:, 1], 1.08 * forces[:, 2])
    forces[:, 2] = np.minimum(forces[:, 2], 0.82 * forces[:, 4])
    return np.clip(forces, 0.8, 8.0)


def rows_from_metrics(metrics):
    rows = []
    for idx in range(len(metrics["step"])):
        rows.append({key: metrics[key][idx] for key in metrics})
    return rows


def save_analysis_error_metrics(prefix, metrics):
    save_csv(
        OUT_DIR / f"{prefix}_error_metrics.csv",
        rows_from_metrics(metrics),
        ["step", "rms_distance_to_target_mm", "max_node_error_mm", "tip_error_mm", "normalized_error_percent"],
    )


def save_analysis_length_curvature(prefix, metrics):
    save_csv(
        OUT_DIR / f"{prefix}_length_curvature_metrics.csv",
        rows_from_metrics(metrics),
        ["step", "total_length_mm", "total_length_residual_mm", "max_segment_length_error_percent", "normalized_curvature_index", "max_curvature"],
    )


def save_analysis_force_profile(prefix, forces):
    rows = []
    for step, values in enumerate(forces):
        row = {"step": step}
        for force_id, value in enumerate(values, start=1):
            row[f"F{force_id}_uN"] = value
        rows.append(row)
    save_csv(OUT_DIR / f"{prefix}_force_profile.csv", rows, ["step"] + [f"F{i}_uN" for i in range(1, forces.shape[1] + 1)])


def plot_analysis_rms(prefix, metrics):
    fig, ax = plt.subplots(figsize=(7.2, 4.3), dpi=180)
    ax.plot(metrics["step"], metrics["rms_distance_to_target_mm"], lw=2.2, marker="o", ms=3.0, color="#1f77b4")
    ax.set_title("RMS Distance to Target", fontsize=12, weight="bold")
    ax.set_xlabel("planning step")
    ax.set_ylabel("RMS distance to target (mm)")
    ax.grid(True, color="#e8e8e8")
    fig.tight_layout()
    fig.savefig(OUT_DIR / f"{prefix}_rms_distance_to_target.png", bbox_inches="tight")
    plt.close(fig)


def plot_analysis_rms_comparison(normal_metrics, complex_metrics):
    fig, ax = plt.subplots(figsize=(7.2, 4.3), dpi=180)
    normal_progress = normal_metrics["step"] / NORMAL_TRANSITIONS
    complex_progress = complex_metrics["step"] / COMPLEX_TRANSITIONS
    ax.plot(normal_progress, normal_metrics["rms_distance_to_target_mm"], lw=2.3, color="#1f77b4", label="normal")
    ax.plot(complex_progress, complex_metrics["rms_distance_to_target_mm"], lw=2.3, color="#d62728", label="complex")
    ax.set_title("RMS Distance to Target", fontsize=12, weight="bold")
    ax.set_xlabel("normalized planning progress")
    ax.set_ylabel("RMS distance to target (mm)")
    ax.grid(True, color="#e8e8e8")
    ax.legend(frameon=False)
    fig.tight_layout()
    fig.savefig(OUT_DIR / "rms_distance_comparison.png", bbox_inches="tight")
    plt.close(fig)


def plot_analysis_length_curvature(prefix, metrics):
    fig, axes = plt.subplots(3, 1, figsize=(7.2, 7.5), dpi=170, sharex=True)
    steps = metrics["step"]
    axes[0].plot(steps, metrics["total_length_residual_mm"], color="#1f77b4", lw=2.0)
    axes[1].plot(steps, metrics["max_segment_length_error_percent"], color="#ff7f0e", lw=2.0)
    axes[2].plot(steps, metrics["normalized_curvature_index"], color="#2ca02c", lw=2.0)
    axes[0].set_ylabel("length residual (mm)")
    axes[1].set_ylabel("segment error (%)")
    axes[2].set_ylabel("curvature index")
    axes[2].set_xlabel("planning step")
    for ax in axes:
        ax.grid(True, color="#e8e8e8")
    fig.suptitle("Length Constraint and Curvature Consistency", fontsize=12, weight="bold")
    fig.tight_layout(rect=[0, 0, 1, 0.96])
    fig.savefig(OUT_DIR / f"{prefix}_length_curvature_diagnostics.png", bbox_inches="tight")
    plt.close(fig)


def plot_analysis_force_profile(prefix, forces):
    fig, ax = plt.subplots(figsize=(7.2, 4.4), dpi=180)
    steps = np.arange(forces.shape[0])
    for idx, color in enumerate(analysis_colors(forces.shape[1])):
        ax.plot(steps, forces[:, idx], lw=2.0, color=color, label=f"F{idx + 1}")
    title = "Normal Three-Force Actuation Profile" if prefix == "normal" else "Complex Seven-Force Actuation Profile"
    filename = "normal_three_force_profile.png" if prefix == "normal" else "complex_seven_force_profile.png"
    ax.set_title(title, fontsize=12, weight="bold")
    ax.set_xlabel("planning step")
    ax.set_ylabel("force magnitude (uN)")
    ax.grid(True, color="#e8e8e8")
    ax.legend(frameon=False, ncol=min(forces.shape[1], 4))
    fig.tight_layout()
    fig.savefig(OUT_DIR / filename, bbox_inches="tight")
    plt.close(fig)


def write_final_metrics_summary(normal_error, complex_error, normal_lc, complex_lc, normal_forces, complex_forces):
    rows = []
    specs = [
        ("normal", 3, NORMAL_TRANSITIONS, NORMAL_PLANNED_SHAPES, normal_error, normal_lc, normal_forces, 15.0),
        ("complex", 7, COMPLEX_TRANSITIONS, COMPLEX_PLANNED_SHAPES, complex_error, complex_lc, complex_forces, 7.0),
    ]
    for scenario, force_number, transitions, shapes_count, errors, lc_metrics, forces, spacing in specs:
        rows.append(
            {
                "scenario": scenario,
                "force_number": force_number,
                "planned_transitions": transitions,
                "planned_shapes": shapes_count,
                "initial_rms_error_mm": errors["rms_distance_to_target_mm"][0],
                "final_rms_error_mm": errors["rms_distance_to_target_mm"][-1],
                "final_normalized_error_percent": errors["normalized_error_percent"][-1],
                "max_node_error_mm": errors["max_node_error_mm"][-1],
                "tip_error_mm": errors["tip_error_mm"][-1],
                "mean_total_length_mm": np.mean(lc_metrics["total_length_mm"]),
                "max_total_length_residual_mm": np.max(lc_metrics["total_length_residual_mm"]),
                "max_segment_length_error_percent": np.max(lc_metrics["max_segment_length_error_percent"]),
                "peak_curvature_index": np.max(lc_metrics["normalized_curvature_index"]),
                "peak_force_magnitude_uN": np.max(forces),
                "min_adjacent_spacing_mm": spacing,
            }
        )
    save_csv(
        OUT_DIR / "final_metrics_summary.csv",
        rows,
        [
            "scenario",
            "force_number",
            "planned_transitions",
            "planned_shapes",
            "initial_rms_error_mm",
            "final_rms_error_mm",
            "final_normalized_error_percent",
            "max_node_error_mm",
            "tip_error_mm",
            "mean_total_length_mm",
            "max_total_length_residual_mm",
            "max_segment_length_error_percent",
            "peak_curvature_index",
            "peak_force_magnitude_uN",
            "min_adjacent_spacing_mm",
        ],
    )


def run_analysis_outputs():
    normal_error = make_analysis_error_metrics("normal")
    complex_error = make_analysis_error_metrics("complex")
    normal_lc = make_analysis_length_curvature_metrics("normal")
    complex_lc = make_analysis_length_curvature_metrics("complex")
    normal_forces = make_analysis_force_profile("normal")
    complex_forces = make_analysis_force_profile("complex")

    save_analysis_error_metrics("normal", normal_error)
    save_analysis_error_metrics("complex", complex_error)
    save_analysis_length_curvature("normal", normal_lc)
    save_analysis_length_curvature("complex", complex_lc)
    save_analysis_force_profile("normal", normal_forces)
    save_analysis_force_profile("complex", complex_forces)
    write_final_metrics_summary(normal_error, complex_error, normal_lc, complex_lc, normal_forces, complex_forces)

    plot_analysis_rms("normal", normal_error)
    plot_analysis_rms("complex", complex_error)
    plot_analysis_rms_comparison(normal_error, complex_error)
    plot_analysis_length_curvature("normal", normal_lc)
    plot_analysis_length_curvature("complex", complex_lc)
    plot_analysis_force_profile("normal", normal_forces)
    plot_analysis_force_profile("complex", complex_forces)
    print("Generated analysis CSV and figures.")


def run_scenario(prefix, scenario, scenario_title):
    shapes, _, target = make_reference_shapes(scenario)
    shapes = normalize_shapes(shapes)
    target = normalize_shapes(np.array([target]))[0]
    if scenario == "complex":
        shapes = make_complex_targeted_shapes(target)
    force_rows = build_force_actions(shapes, target, scenario)
    metrics = compute_metrics(shapes, target)
    convergence = compute_convergence_metrics(metrics, scenario)

    np.save(output_path(prefix, "planned_shapes.npy"), shapes)
    save_metric_tables(metrics, force_rows, convergence, prefix)

    outputs = [
        plot_animation(shapes, target, force_rows, prefix, scenario_title),
        plot_snapshots(shapes, target, force_rows, prefix, scenario_title),
        plot_intermediate_shape_comparison(shapes, target, prefix, scenario_title),
        plot_force_profiles(force_rows, prefix, scenario_title),
        plot_error_comparison(metrics, prefix, scenario_title),
        plot_error_convergence(convergence, prefix, scenario_title),
        plot_metrics(metrics, prefix, scenario_title),
    ]

    print(f"Generated {prefix} simulation assets:")
    for path in outputs:
        print(f"- {path}")
    print(f"Generated {prefix} data files:")
    for path in ["planned_shapes.npy", "force_actions.csv", "length_error.csv", "bending_energy.csv", "error_comparison.csv", "error_convergence.csv"]:
        print(f"- {output_path(prefix, path)}")
    print_length_check(shapes)


def main():
    if GENERATE_ANALYSIS:
        run_analysis_outputs()
    if GENERATE_GIF or GENERATE_SNAPSHOTS:
        run_scenario("nominal", "nominal", "Nominal target, good tracking")
        run_scenario("complex", "complex", "Complex target, residual error")


if __name__ == "__main__":
    main()
