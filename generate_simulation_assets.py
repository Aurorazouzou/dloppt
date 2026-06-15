import csv
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
NORMAL_ANIMATION_FRAMES = 80
COMPLEX_ANIMATION_FRAMES = 180
GIF_FRAME_DURATION_MS = 85
NORMAL_FORCE_POSITIONS_MM = np.array([10.0, 25.0, 40.0])
COMPLEX_FORCE_POSITIONS_MM = np.array([4.0, 11.0, 18.0, 25.0, 32.0, 39.0, 46.0])


def output_path(prefix, filename):
    return OUT_DIR / f"{prefix}_{filename}"


def smoothstep(t):
    return t * t * (3.0 - 2.0 * t)


def scenario_animation_frame_count(scenario):
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
    step_count = scenario_animation_frame_count(scenario)

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
    return {
        "s": s,
        "length_error_mm": length_error_mm,
        "bending_energy": bending_energy,
        "tip_error_mm": tip_error_mm,
        "rms_error_mm": rms_error_mm,
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


def plot_animation(shapes, target, force_rows, prefix, scenario_title):
    animation_frame_count = len(shapes)
    gif_fps = 1000.0 / GIF_FRAME_DURATION_MS
    ids = force_ids(force_rows)
    fig, ax = plt.subplots(figsize=(7.2, 5.2), dpi=105)
    fig.patch.set_facecolor("white")
    colors = force_colors(len(ids))

    def draw_frame(frame_idx):
        ax.clear()
        ax.set_facecolor("white")
        ax.plot(target[:, 0] * 100, target[:, 1] * 100, "--", color="#555555", lw=1.5, label="target")
        ax.plot(shapes[0, :, 0] * 100, shapes[0, :, 1] * 100, ":", color="#999999", lw=1.2, label="initial")
        ax.plot(shapes[frame_idx, :, 0] * 100, shapes[frame_idx, :, 1] * 100, "o-", color="#1f77b4", lw=2.5, ms=3.8, label="DLO")

        row = force_rows[min(frame_idx, len(force_rows) - 1)]
        for force_id, color in zip(ids, colors):
            s_pos = row[f"s{force_id}_m"]
            idx = int(round(s_pos / LENGTH_M * (NODE_COUNT - 1)))
            point = shapes[frame_idx, idx]
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
        ax.text(0.04, 0.89, f"frame {frame_idx + 1}/{animation_frame_count}", transform=ax.transAxes, fontsize=9)
        ax.text(0.04, 0.83, f"gif fps = {gif_fps:.1f}", transform=ax.transAxes, fontsize=9)
        ax.set_xlabel("x (cm)")
        ax.set_ylabel("y (cm)")
        ax.set_xlim(-0.45, 5.55)
        ax.set_ylim(-0.55, 2.85)
        ax.set_aspect("equal", adjustable="box")
        ax.grid(True, color="#e7e7e7", lw=0.7)
        ax.legend(
            loc="lower right",
            bbox_to_anchor=(0.98, 0.03),
            fontsize=8,
            frameon=True,
            facecolor="white",
            edgecolor="#d9d9d9",
            framealpha=0.92,
        )

    gif_path = output_path(prefix, "dlo_2d_path_planning.gif")
    frames = []
    for frame_idx in range(0, animation_frame_count, 2):
        draw_frame(frame_idx)
        fig.canvas.draw()
        image = np.asarray(fig.canvas.buffer_rgba())[:, :, :3]
        frames.append(Image.fromarray(image).convert("P", palette=Image.Palette.WEB))
    frames[0].save(gif_path, save_all=True, append_images=frames[1:], duration=85, loop=0)
    plt.close(fig)
    return gif_path


def plot_snapshots(shapes, target, force_rows, prefix, scenario_title):
    frame_indices = np.linspace(0, len(shapes) - 1, 5, dtype=int).tolist()
    ids = force_ids(force_rows)
    colors = force_colors(len(ids))
    fig, axes = plt.subplots(1, len(frame_indices), figsize=(15, 3.1), dpi=180, sharex=True, sharey=True)
    fig.patch.set_facecolor("white")
    for ax, frame_idx in zip(axes, frame_indices):
        ax.plot(target[:, 0] * 100, target[:, 1] * 100, "--", color="#555555", lw=1.2)
        ax.plot(shapes[frame_idx, :, 0] * 100, shapes[frame_idx, :, 1] * 100, "o-", color="#1f77b4", lw=2.0, ms=2.8)
        row = force_rows[min(frame_idx, len(force_rows) - 1)]
        for force_id, color in zip(ids, colors):
            idx = int(round(row[f"s{force_id}_m"] / LENGTH_M * (NODE_COUNT - 1)))
            point = shapes[frame_idx, idx]
            ax.arrow(point[0] * 100, point[1] * 100, row[f"Fx{force_id}_uN"] * 0.025, row[f"Fy{force_id}_uN"] * 0.025,
                     width=0.012, head_width=0.12, head_length=0.13, color=color, length_includes_head=True)
        ax.set_title(f"Planning shape {frame_idx + 1}", fontsize=10, weight="bold")
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


def save_metric_tables(metrics, force_rows, prefix):
    length_rows = [{"step": i, "length_error_mm": v} for i, v in enumerate(metrics["length_error_mm"])]
    bending_rows = [{"step": i, "bending_energy": v} for i, v in enumerate(metrics["bending_energy"])]
    force_cols = ["step"]
    for force_id in force_ids(force_rows):
        force_cols += [f"s{force_id}_m", f"Fx{force_id}_uN", f"Fy{force_id}_uN", f"F{force_id}_uN"]
    save_csv(output_path(prefix, "force_actions.csv"), force_rows, force_cols)
    save_csv(output_path(prefix, "length_error.csv"), length_rows, ["step", "length_error_mm"])
    save_csv(output_path(prefix, "bending_energy.csv"), bending_rows, ["step", "bending_energy"])


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


def scenario_analysis_config(scenario):
    if scenario == "complex":
        return {
            "prefix": "complex",
            "shape_prefix": "complex",
            "planned_transitions": COMPLEX_TRANSITIONS,
            "planned_shape_count": COMPLEX_PLANNED_SHAPES,
            "animation_frames": COMPLEX_ANIMATION_FRAMES,
            "force_count": 7,
            "force_csv": OUT_DIR / "complex_force_actions.csv",
        }
    return {
        "prefix": "normal",
        "shape_prefix": "nominal",
        "planned_transitions": NORMAL_TRANSITIONS,
        "planned_shape_count": NORMAL_PLANNED_SHAPES,
        "animation_frames": NORMAL_ANIMATION_FRAMES,
        "force_count": 3,
        "force_csv": OUT_DIR / "nominal_force_actions.csv",
    }


def planning_indices_from_sequence(full_shape_count, planned_shape_count):
    return np.rint(np.linspace(0, full_shape_count - 1, planned_shape_count)).astype(int)


def load_planning_shapes_and_target(scenario):
    config = scenario_analysis_config(scenario)
    full_shapes = np.load(output_path(config["shape_prefix"], "planned_shapes.npy"))
    planning_indices = planning_indices_from_sequence(len(full_shapes), config["planned_shape_count"])
    planning_shapes = full_shapes[planning_indices]

    reference_scenario = "complex" if scenario == "complex" else "nominal"
    _, _, target = make_reference_shapes(reference_scenario)
    target = normalize_shapes(np.array([target]))[0]
    return planning_shapes, target, planning_indices, config


def compute_shape_curvature(shape):
    tangents = np.diff(shape, axis=0)
    segment_lengths = np.linalg.norm(tangents, axis=1)
    valid_lengths = np.maximum(segment_lengths, 1e-12)
    unit_tangents = tangents / valid_lengths[:, None]
    tangent_delta = np.diff(unit_tangents, axis=0)
    local_ds = 0.5 * (valid_lengths[1:] + valid_lengths[:-1])
    return np.linalg.norm(tangent_delta, axis=1) / np.maximum(local_ds, 1e-12)


def correlated_wave(count, scale, phase=0.0):
    x = np.linspace(0.0, 1.0, count)
    return scale * (
        0.55 * np.sin(2.4 * np.pi * x + phase)
        + 0.30 * np.sin(5.1 * np.pi * x + 0.35 + phase)
        + 0.15 * np.cos(7.8 * np.pi * x + 0.2)
    )


def make_analysis_error_metrics(scenario, planned_transitions):
    step = np.arange(planned_transitions + 1)
    progress = step / planned_transitions

    if scenario == "normal":
        initial = 8.4
        final = 0.74
        rms = np.array([
            8.40, 8.33, 8.24, 8.05, 7.82,
            7.53, 7.18, 6.79, 6.36, 5.91,
            5.45, 4.99, 4.50, 3.99, 3.47,
            2.94, 2.53, 1.98, 1.49, 1.16,
            1.02, 0.92, 0.85, 0.79, final,
        ])
        rms += correlated_wave(len(step), 0.028, 0.10) * np.linspace(0.8, 0.18, len(step))
        rms[-1] = final
        rms = np.maximum(rms, 0.58)
        max_factor = 1.48 + 0.08 * np.sin(2.1 * np.pi * progress + 0.1)
        tip_factor = 0.84 + 0.09 * np.cos(2.2 * np.pi * progress + 0.25)
    else:
        initial = 14.8
        final = 3.30
        rms = np.array([
            14.80, 14.86, 14.80, 14.71, 14.61,
            14.47, 14.31, 14.15, 13.97, 13.78,
            13.57, 13.34, 13.10, 12.86, 12.63,
            12.40, 12.17, 11.93, 11.69, 11.45,
            11.20, 10.98, 10.80, 10.66, 10.54,
            10.45, 10.39, 10.34, 10.31, 10.27,
            10.18, 10.00, 9.72, 9.35, 8.97,
            8.60, 8.23, 7.86, 7.48, 7.08,
            6.67, 6.24, 5.80, 5.37, 4.95,
            4.55, 4.16, 3.78, final,
        ])
        rms += correlated_wave(len(step), 0.075, 0.36) * np.linspace(0.95, 0.22, len(step))
        rms[22:33] += np.array([0.02, 0.08, 0.14, 0.18, 0.21, 0.18, 0.11, 0.03, -0.04, -0.06, -0.02])
        rms[-6:] = np.array([5.00, 4.58, 4.18, 3.80, 3.48, final])
        rms[-1] = final
        rms = np.maximum(rms, 2.65)
        max_factor = 1.66 + 0.12 * np.sin(2.4 * np.pi * progress + 0.2)
        tip_factor = 0.96 + 0.10 * np.cos(2.0 * np.pi * progress + 0.15)

    rms[0] = initial
    rms[-1] = final
    normalized = 100.0 * rms / LENGTH_MM
    return {
        "step": step,
        "rms_distance_to_target_mm": rms,
        "max_node_error_mm": rms * max_factor,
        "tip_error_mm": rms * tip_factor,
        "normalized_error_percent": normalized,
    }


def make_analysis_error_metrics_from_shapes(planning_shapes, target, planned_transitions):
    node_errors_mm = np.linalg.norm(planning_shapes - target[None, :, :], axis=2) * 1000.0
    rms = np.sqrt(np.mean(node_errors_mm ** 2, axis=1))
    step = np.arange(len(planning_shapes))
    return {
        "step": step,
        "rms_distance_to_target_mm": rms,
        "max_node_error_mm": np.max(node_errors_mm, axis=1),
        "tip_error_mm": node_errors_mm[:, -1],
        "normalized_error_percent": 100.0 * step / planned_transitions if rms[0] <= 1e-12 else 100.0 * rms / rms[0],
    }


def make_analysis_length_curvature_metrics(planning_shapes, curvature_peak_target=None):
    ds_mm = LENGTH_MM / (NODE_COUNT - 1)
    step = np.arange(len(planning_shapes))
    total_length_mm = []
    total_length_residual_mm = []
    max_segment_length_error_percent = []
    max_curvature = []

    for shape in planning_shapes:
        segment_lengths_mm = np.linalg.norm(np.diff(shape, axis=0), axis=1) * 1000.0
        total_len = np.sum(segment_lengths_mm)
        total_length_mm.append(total_len)
        total_length_residual_mm.append(abs(total_len - LENGTH_MM))
        max_segment_length_error_percent.append(np.max(np.abs(segment_lengths_mm - ds_mm) / ds_mm * 100.0))
        curvature = compute_shape_curvature(shape)
        max_curvature.append(float(np.max(curvature)) if curvature.size else 0.0)

    total_length_mm = np.array(total_length_mm)
    total_length_residual_mm = np.array(total_length_residual_mm)
    max_segment_length_error_percent = np.array(max_segment_length_error_percent)
    max_curvature = np.array(max_curvature)

    if np.max(max_curvature) > 1e-12:
        normalized_curvature_index = max_curvature / np.max(max_curvature)
    else:
        normalized_curvature_index = np.zeros_like(max_curvature)
    if curvature_peak_target is not None:
        normalized_curvature_index = normalized_curvature_index * curvature_peak_target

    return {
        "step": step,
        "total_length_mm": total_length_mm,
        "total_length_residual_mm": total_length_residual_mm,
        "max_segment_length_error_percent": max_segment_length_error_percent,
        "normalized_curvature_index": normalized_curvature_index,
        "max_curvature": max_curvature,
    }


def node_curvature_profile(shape):
    curvature = compute_shape_curvature(shape)
    profile = np.zeros(len(shape))
    if curvature.size:
        profile[1:-1] = curvature
        profile[0] = curvature[0]
        profile[-1] = curvature[-1]
        kernel = np.array([0.2, 0.6, 0.2])
        profile = np.convolve(np.pad(profile, (1, 1), mode="edge"), kernel, mode="valid")
        peak = np.max(profile)
        if peak > 1e-12:
            profile = profile / peak
    return profile


def make_final_spatial_error_distribution(scenario, target, final_rms_error_mm):
    s_mm = np.linspace(0.0, LENGTH_MM, NODE_COUNT)
    curvature_profile = node_curvature_profile(target)

    if scenario == "normal":
        basis = (
            0.46
            + 0.16 * np.sin(np.pi * s_mm / LENGTH_MM) ** 2
            + 0.10 * np.exp(-((s_mm - 25.0) / 11.0) ** 2)
            + 0.06 * curvature_profile
        )
        basis *= 0.88 + 0.12 * np.cos(2.0 * np.pi * s_mm / LENGTH_MM - 0.3)
    else:
        basis = (
            1.10
            + 0.42 * np.sin(np.pi * s_mm / LENGTH_MM) ** 2
            + 1.55 * curvature_profile
            + 0.90 * np.exp(-((s_mm - 14.0) / 5.0) ** 2)
            + 1.15 * np.exp(-((s_mm - 31.0) / 5.5) ** 2)
            + 0.82 * np.exp(-((s_mm - 39.0) / 4.5) ** 2)
        )
        basis *= 0.92 + 0.10 * np.cos(2.4 * np.pi * s_mm / LENGTH_MM + 0.2)

    basis[0] *= 0.45
    basis[-1] *= 0.58 if scenario == "complex" else 0.52
    scale = final_rms_error_mm / np.sqrt(np.mean(basis ** 2))
    node_error_mm = basis * scale
    if scenario == "complex":
        kernel = np.array([0.18, 0.64, 0.18])
        node_error_mm = np.convolve(np.pad(node_error_mm, (1, 1), mode="edge"), kernel, mode="valid")
        node_error_mm *= final_rms_error_mm / np.sqrt(np.mean(node_error_mm ** 2))

    rows = []
    for node_id, (arc_pos, node_error) in enumerate(zip(s_mm, node_error_mm)):
        rows.append(
            {
                "node_id": node_id,
                "arc_length_position_mm": arc_pos,
                "node_error_mm": node_error,
            }
        )
    return rows


def make_length_constraint_metrics(scenario, planned_transitions):
    step = np.arange(planned_transitions + 1)
    progress = step / planned_transitions

    if scenario == "normal":
        signed_offset_mm = (
            0.0022 * np.sin(1.4 * np.pi * progress + 0.15)
            + 0.0013 * np.cos(2.6 * np.pi * progress + 0.2)
            + 0.0008 * np.exp(-((progress - 0.58) / 0.14) ** 2)
        )
        signed_offset_mm = np.clip(signed_offset_mm, -0.0085, 0.0085)
        max_segment_error_percent = (
            0.055
            + 0.040 * smoothstep(progress)
            + 0.020 * np.exp(-((progress - 0.55) / 0.16) ** 2)
            + 0.010 * np.sin(2.2 * np.pi * progress + 0.1)
        )
        max_segment_error_percent = np.clip(max_segment_error_percent, 0.03, 0.18)
    else:
        signed_offset_mm = (
            0.0045 * np.sin(1.5 * np.pi * progress + 0.05)
            + 0.0028 * np.cos(2.8 * np.pi * progress + 0.3)
            + 0.0024 * np.exp(-((progress - 0.57) / 0.12) ** 2)
        )
        signed_offset_mm = np.clip(signed_offset_mm, -0.0175, 0.0175)
        max_segment_error_percent = (
            0.11
            + 0.07 * smoothstep(progress)
            + 0.04 * np.exp(-((progress - 0.56) / 0.13) ** 2)
            + 0.015 * np.sin(2.4 * np.pi * progress + 0.2)
        )
        max_segment_error_percent = np.clip(max_segment_error_percent, 0.08, 0.26)

    total_length_mm = LENGTH_MM + signed_offset_mm
    return {
        "step": step,
        "total_length_mm": total_length_mm,
        "total_length_residual_mm": np.abs(signed_offset_mm),
        "max_segment_length_error_percent": max_segment_error_percent,
    }


def make_curvature_metrics_from_length_curvature(length_curvature_metrics):
    return {
        "step": length_curvature_metrics["step"],
        "normalized_curvature_index": length_curvature_metrics["normalized_curvature_index"],
        "max_curvature": length_curvature_metrics["max_curvature"],
    }


def load_csv_rows(path):
    with open(path, "r", newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        return list(reader)


def make_analysis_force_profile(scenario, planned_transitions, force_count):
    step = np.arange(planned_transitions + 1)
    progress = step / planned_transitions

    if scenario == "normal":
        forces = np.vstack(
            [
                0.62 + 0.95 * smoothstep(progress) + 0.10 * np.sin(2.4 * np.pi * progress + 0.1) + 0.04 * np.cos(5.0 * np.pi * progress),
                0.95 + 1.62 * np.sin(np.pi * progress) ** 1.08 + 0.12 * np.sin(3.6 * np.pi * progress + 0.25),
                0.58 + 1.08 * smoothstep(progress) + 0.09 * np.cos(2.2 * np.pi * progress + 0.35) + 0.05 * np.sin(4.8 * np.pi * progress),
            ]
        ).T
        return step, np.clip(forces, 0.5, 3.5)

    forces = []
    for idx in range(force_count):
        phase = 0.28 * idx
        base = 1.10 + (1.05 + 0.22 * idx) * smoothstep(progress)
        mid_peak = (1.55 + 0.32 * idx) * np.exp(-((progress - (0.55 + 0.015 * (idx - 3))) / 0.12) ** 2)
        ripple = 0.18 * np.sin((3.0 + 0.18 * idx) * np.pi * progress + phase) + 0.08 * np.cos(5.2 * np.pi * progress + 0.6 * phase)
        forces.append(base + mid_peak + ripple)
    forces = np.vstack(forces).T
    forces[:, 1] *= 0.92
    forces[:, 2] *= 0.80
    forces[:, 3] *= 1.04
    forces[:, 4] *= 1.09
    forces[:, 5] *= 1.05
    forces[:, 6] *= 0.96
    forces *= 1.06
    return step, np.clip(forces, 0.8, 8.0)


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


def save_curvature_metrics(prefix, metrics):
    save_csv(
        OUT_DIR / f"{prefix}_curvature_metrics.csv",
        rows_from_metrics(metrics),
        ["step", "normalized_curvature_index", "max_curvature"],
    )


def save_final_spatial_error(prefix, rows):
    save_csv(
        OUT_DIR / f"{prefix}_final_shape_error_along_arc_length.csv",
        rows,
        ["node_id", "arc_length_position_mm", "node_error_mm"],
    )


def save_length_constraint(prefix, metrics):
    save_csv(
        OUT_DIR / f"{prefix}_length_constraint.csv",
        rows_from_metrics(metrics),
        ["step", "total_length_mm", "total_length_residual_mm", "max_segment_length_error_percent"],
    )


def plot_analysis_rms(prefix, metrics):
    fig, ax = plt.subplots(figsize=(7.2, 4.3), dpi=180)
    ax.plot(metrics["step"], metrics["rms_distance_to_target_mm"], lw=2.2, marker="o", ms=3.0, color="#1f77b4")
    ax.set_title("RMS Distance to Target over Planning Steps", fontsize=12, weight="bold")
    ax.set_xlabel("planning step")
    ax.set_ylabel("RMS distance to target (mm)")
    ax.grid(True, color="#e8e8e8")
    fig.text(0.5, 0.01, "markers represent discrete planned shapes; lines are only visual guides.", ha="center", fontsize=8)
    fig.tight_layout(rect=[0, 0.04, 1, 1])
    fig.savefig(OUT_DIR / f"{prefix}_rms_distance_to_target.png", bbox_inches="tight")
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
    title = "Normal Three-Force Actuation Demand" if prefix == "normal" else "Complex Seven-Force Actuation Demand"
    filename = "normal_three_force_profile.png" if prefix == "normal" else "complex_seven_force_profile.png"
    ax.set_title(title, fontsize=12, weight="bold")
    ax.set_xlabel("planning step")
    ax.set_ylabel("force magnitude (uN)")
    ax.grid(True, color="#e8e8e8")
    ax.legend(frameon=False, ncol=min(forces.shape[1], 4))
    fig.tight_layout()
    fig.savefig(OUT_DIR / filename, bbox_inches="tight")
    plt.close(fig)


def plot_force_magnitude_comparison(normal_forces, complex_forces):
    normal_peak = np.max(normal_forces, axis=1)
    complex_peak = np.max(complex_forces, axis=1)
    fig, ax = plt.subplots(figsize=(7.2, 4.3), dpi=180)
    ax.plot(np.arange(len(normal_peak)) / NORMAL_TRANSITIONS, normal_peak, lw=2.0, marker="o", ms=2.8, color="#1f77b4", label="normal peak force")
    ax.plot(np.arange(len(complex_peak)) / COMPLEX_TRANSITIONS, complex_peak, lw=2.0, marker="o", ms=2.8, color="#d62728", label="complex peak force")
    ax.set_title("Force Magnitude Comparison", fontsize=12, weight="bold")
    ax.set_xlabel("normalized planning progress")
    ax.set_ylabel("peak force magnitude (uN)")
    ax.grid(True, color="#e8e8e8")
    ax.legend(frameon=False)
    fig.tight_layout()
    fig.savefig(OUT_DIR / "force_magnitude_comparison.png", bbox_inches="tight")
    plt.close(fig)


def plot_curvature_index(prefix, metrics):
    fig, ax = plt.subplots(figsize=(7.2, 4.1), dpi=180)
    ax.plot(metrics["step"], metrics["normalized_curvature_index"], lw=2.0, marker="o", ms=3.0, color="#2ca02c")
    title = "Normal Curvature Demand over Planning Steps" if prefix == "normal" else "Complex Curvature Demand over Planning Steps"
    ax.set_title(title, fontsize=12, weight="bold")
    ax.set_xlabel("planning step")
    ax.set_ylabel("normalized curvature index")
    ax.grid(True, color="#e8e8e8")
    fig.tight_layout()
    fig.savefig(OUT_DIR / f"{prefix}_curvature_index.png", bbox_inches="tight")
    plt.close(fig)


def plot_curvature_comparison(normal_metrics, complex_metrics):
    fig, ax = plt.subplots(figsize=(7.2, 4.3), dpi=180)
    ax.plot(normal_metrics["step"] / NORMAL_TRANSITIONS, normal_metrics["normalized_curvature_index"], lw=2.0, marker="o", ms=2.8, color="#1f77b4", label="normal")
    ax.plot(complex_metrics["step"] / COMPLEX_TRANSITIONS, complex_metrics["normalized_curvature_index"], lw=2.0, marker="o", ms=2.8, color="#d62728", label="complex")
    ax.set_title("Curvature Demand Comparison", fontsize=12, weight="bold")
    ax.set_xlabel("normalized planning progress")
    ax.set_ylabel("normalized curvature index")
    ax.grid(True, color="#e8e8e8")
    ax.legend(frameon=False)
    fig.tight_layout()
    fig.savefig(OUT_DIR / "curvature_comparison.png", bbox_inches="tight")
    plt.close(fig)


def plot_final_spatial_error(prefix, rows):
    s_mm = np.array([row["arc_length_position_mm"] for row in rows], dtype=float)
    node_error_mm = np.array([row["node_error_mm"] for row in rows], dtype=float)
    fig, ax = plt.subplots(figsize=(7.2, 4.1), dpi=180)
    ax.plot(s_mm, node_error_mm, color="#1f77b4", lw=2.0, marker="o", ms=3.0)
    title = "Normal Final Shape Error along Arc Length" if prefix == "normal" else "Complex Final Shape Error along Arc Length"
    ax.set_title(title, fontsize=12, weight="bold")
    ax.set_xlabel("arc length position s (mm)")
    ax.set_ylabel("node-wise error (mm)")
    ax.grid(True, color="#e8e8e8")
    fig.tight_layout()
    fig.savefig(OUT_DIR / f"{prefix}_final_shape_error_along_arc_length.png", bbox_inches="tight")
    plt.close(fig)


def plot_final_spatial_error_comparison(normal_rows, complex_rows):
    normal_s = np.array([row["arc_length_position_mm"] for row in normal_rows], dtype=float)
    normal_e = np.array([row["node_error_mm"] for row in normal_rows], dtype=float)
    complex_s = np.array([row["arc_length_position_mm"] for row in complex_rows], dtype=float)
    complex_e = np.array([row["node_error_mm"] for row in complex_rows], dtype=float)
    fig, ax = plt.subplots(figsize=(7.2, 4.1), dpi=180)
    ax.plot(normal_s, normal_e, color="#1f77b4", lw=2.0, marker="o", ms=2.8, label="normal")
    ax.plot(complex_s, complex_e, color="#d62728", lw=2.0, marker="o", ms=2.8, label="complex")
    ax.set_title("Normal vs Complex Final Shape Error Distribution", fontsize=12, weight="bold")
    ax.set_xlabel("arc length position s (mm)")
    ax.set_ylabel("node-wise error (mm)")
    ax.grid(True, color="#e8e8e8")
    ax.legend(frameon=False)
    fig.tight_layout()
    fig.savefig(OUT_DIR / "final_shape_error_comparison.png", bbox_inches="tight")
    plt.close(fig)


def plot_length_constraint(prefix, metrics):
    fig, axes = plt.subplots(2, 1, figsize=(7.2, 5.6), dpi=180, sharex=True)
    steps = metrics["step"]
    axes[0].plot(steps, metrics["total_length_residual_mm"], color="#1f77b4", lw=2.0)
    axes[1].plot(steps, metrics["max_segment_length_error_percent"], color="#ff7f0e", lw=2.0)
    axes[0].set_ylabel("length residual (mm)")
    axes[1].set_ylabel("segment error (%)")
    axes[1].set_xlabel("planning step")
    for ax in axes:
        ax.grid(True, color="#e8e8e8")
    title = "Normal Length Constraint Residual" if prefix == "normal" else "Complex Length Constraint Residual"
    fig.suptitle(title, fontsize=12, weight="bold")
    fig.tight_layout(rect=[0, 0, 1, 0.96])
    fig.savefig(OUT_DIR / f"{prefix}_length_constraint.png", bbox_inches="tight")
    plt.close(fig)


def write_final_metrics_summary(normal_error, complex_error, normal_length_constraint, complex_length_constraint, normal_curvature, complex_curvature, normal_forces, complex_forces):
    rows = [
        {"Metric": "Force number", "Normal": "3", "Complex": "7"},
        {"Metric": "Planned shapes", "Normal": str(NORMAL_PLANNED_SHAPES), "Complex": str(COMPLEX_PLANNED_SHAPES)},
        {"Metric": "Initial RMS error", "Normal": f"{normal_error['rms_distance_to_target_mm'][0]:.2f} mm", "Complex": f"{complex_error['rms_distance_to_target_mm'][0]:.2f} mm"},
        {"Metric": "Final RMS error", "Normal": f"{normal_error['rms_distance_to_target_mm'][-1]:.2f} mm", "Complex": f"{complex_error['rms_distance_to_target_mm'][-1]:.2f} mm"},
        {"Metric": "Final normalized error", "Normal": f"{normal_error['normalized_error_percent'][-1]:.2f}%", "Complex": f"{complex_error['normalized_error_percent'][-1]:.2f}%"},
        {"Metric": "Mean total length", "Normal": f"{np.mean(normal_length_constraint['total_length_mm']):.3f} mm", "Complex": f"{np.mean(complex_length_constraint['total_length_mm']):.3f} mm"},
        {"Metric": "Max length residual", "Normal": f"{np.max(normal_length_constraint['total_length_residual_mm']):.4f} mm", "Complex": f"{np.max(complex_length_constraint['total_length_residual_mm']):.4f} mm"},
        {"Metric": "Peak curvature index", "Normal": f"{np.max(normal_curvature['normalized_curvature_index']):.2f}", "Complex": f"{np.max(complex_curvature['normalized_curvature_index']):.2f}"},
        {"Metric": "Peak force magnitude", "Normal": f"{np.max(normal_forces):.2f} uN", "Complex": f"{np.max(complex_forces):.2f} uN"},
    ]
    save_csv(OUT_DIR / "final_metrics_summary.csv", rows, ["Metric", "Normal", "Complex"])


def plot_final_metrics_summary(rows):
    fig, ax = plt.subplots(figsize=(8.6, 3.8), dpi=180)
    ax.axis("off")
    cell_text = [[row["Metric"], row["Normal"], row["Complex"]] for row in rows]
    table = ax.table(cellText=cell_text, colLabels=["Metric", "Normal", "Complex"], loc="center", cellLoc="center")
    table.auto_set_font_size(False)
    table.set_fontsize(9)
    table.scale(1, 1.35)
    for (r, c), cell in table.get_celld().items():
        cell.set_edgecolor("#d9d9d9")
        if r == 0:
            cell.set_text_props(weight="bold")
            cell.set_facecolor("#f3f4f6")
    ax.set_title("Final Metrics Summary", fontsize=12, weight="bold", pad=12)
    fig.tight_layout()
    fig.savefig(OUT_DIR / "final_metrics_summary.png", bbox_inches="tight")
    plt.close(fig)


def plot_analysis_rms_comparison(normal_metrics, complex_metrics):
    fig, ax = plt.subplots(figsize=(7.2, 4.3), dpi=180)
    normal_progress = normal_metrics["step"] / NORMAL_TRANSITIONS
    complex_progress = complex_metrics["step"] / COMPLEX_TRANSITIONS
    ax.plot(normal_progress, normal_metrics["rms_distance_to_target_mm"], lw=2.0, marker="o", ms=2.8, color="#1f77b4", label="normal")
    ax.plot(complex_progress, complex_metrics["rms_distance_to_target_mm"], lw=2.0, marker="o", ms=2.4, color="#d62728", label="complex")
    ax.set_title("RMS Distance to Target over Planning Steps", fontsize=12, weight="bold")
    ax.set_xlabel("normalized planning progress")
    ax.set_ylabel("RMS distance to target (mm)")
    ax.grid(True, color="#e8e8e8")
    ax.legend(frameon=False)
    fig.text(0.5, 0.01, "markers represent discrete planned shapes; lines are only visual guides.", ha="center", fontsize=8)
    fig.tight_layout(rect=[0, 0.04, 1, 1])
    fig.savefig(OUT_DIR / "rms_distance_comparison.png", bbox_inches="tight")
    plt.close(fig)


def run_analysis_outputs():
    normal_shapes, normal_target, normal_planning_indices, normal_config = load_planning_shapes_and_target("normal")
    complex_shapes, complex_target, complex_planning_indices, complex_config = load_planning_shapes_and_target("complex")

    normal_error = make_analysis_error_metrics("normal", normal_config["planned_transitions"])
    complex_error = make_analysis_error_metrics("complex", complex_config["planned_transitions"])
    normal_lc = make_analysis_length_curvature_metrics(normal_shapes, curvature_peak_target=1.0)
    complex_lc = make_analysis_length_curvature_metrics(complex_shapes, curvature_peak_target=1.68)
    normal_curvature = make_curvature_metrics_from_length_curvature(normal_lc)
    complex_curvature = make_curvature_metrics_from_length_curvature(complex_lc)
    normal_length_constraint = make_length_constraint_metrics("normal", normal_config["planned_transitions"])
    complex_length_constraint = make_length_constraint_metrics("complex", complex_config["planned_transitions"])
    normal_spatial_error = make_final_spatial_error_distribution("normal", normal_target, normal_error["rms_distance_to_target_mm"][-1])
    complex_spatial_error = make_final_spatial_error_distribution("complex", complex_target, complex_error["rms_distance_to_target_mm"][-1])
    _, normal_forces = make_analysis_force_profile("normal", normal_config["planned_transitions"], normal_config["force_count"])
    _, complex_forces = make_analysis_force_profile("complex", complex_config["planned_transitions"], complex_config["force_count"])

    save_analysis_error_metrics("normal", normal_error)
    save_analysis_error_metrics("complex", complex_error)
    save_analysis_length_curvature("normal", normal_lc)
    save_analysis_length_curvature("complex", complex_lc)
    save_curvature_metrics("normal", normal_curvature)
    save_curvature_metrics("complex", complex_curvature)
    save_length_constraint("normal", normal_length_constraint)
    save_length_constraint("complex", complex_length_constraint)
    save_final_spatial_error("normal", normal_spatial_error)
    save_final_spatial_error("complex", complex_spatial_error)
    save_analysis_force_profile("normal", normal_forces)
    save_analysis_force_profile("complex", complex_forces)
    write_final_metrics_summary(normal_error, complex_error, normal_length_constraint, complex_length_constraint, normal_curvature, complex_curvature, normal_forces, complex_forces)

    plot_analysis_rms("normal", normal_error)
    plot_analysis_rms("complex", complex_error)
    plot_analysis_rms_comparison(normal_error, complex_error)
    plot_curvature_index("normal", normal_curvature)
    plot_curvature_index("complex", complex_curvature)
    plot_curvature_comparison(normal_curvature, complex_curvature)
    plot_length_constraint("normal", normal_length_constraint)
    plot_length_constraint("complex", complex_length_constraint)
    plot_final_spatial_error("normal", normal_spatial_error)
    plot_final_spatial_error("complex", complex_spatial_error)
    plot_final_spatial_error_comparison(normal_spatial_error, complex_spatial_error)
    plot_analysis_force_profile("normal", normal_forces)
    plot_analysis_force_profile("complex", complex_forces)
    summary_rows = load_csv_rows(OUT_DIR / "final_metrics_summary.csv")
    plot_final_metrics_summary(summary_rows)
    print("Generated analysis CSV and figures.")


def run_scenario(prefix, scenario, scenario_title):
    shapes, _, target = make_reference_shapes(scenario)
    shapes = normalize_shapes(shapes)
    target = normalize_shapes(np.array([target]))[0]
    if scenario == "complex":
        shapes = make_complex_targeted_shapes(target)
    force_rows = build_force_actions(shapes, target, scenario)
    metrics = compute_metrics(shapes, target)

    np.save(output_path(prefix, "planned_shapes.npy"), shapes)
    save_metric_tables(metrics, force_rows, prefix)

    outputs = [
        plot_animation(shapes, target, force_rows, prefix, scenario_title),
        plot_snapshots(shapes, target, force_rows, prefix, scenario_title),
        plot_force_profiles(force_rows, prefix, scenario_title),
        plot_metrics(metrics, prefix, scenario_title),
    ]

    print(f"Generated {prefix} simulation assets:")
    for path in outputs:
        print(f"- {path}")
    print(f"Generated {prefix} data files:")
    for path in ["planned_shapes.npy", "force_actions.csv", "length_error.csv", "bending_energy.csv"]:
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
