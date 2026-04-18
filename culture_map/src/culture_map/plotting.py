from pathlib import Path
from collections import deque

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from PIL import Image
from matplotlib import patheffects
from matplotlib.ticker import FormatStrFormatter

from .constants import (
    CATEGORY_COLORS,
    CATEGORY_LABEL_FONT_SIZES,
    CATEGORY_LABEL_POSITIONS,
    CULTURE_ZONE_DRAW_ORDER,
    MODEL_LABEL_FONT_SIZES,
    MODEL_LABEL_NUDGES,
    MUSLIM_COUNTRIES,
    TEMPLATE_2023_SPEC,
)
from .utils import ensure_dir


def _expand_polygon(points, expansion=1.12):
    centroid = points.mean(axis=0)
    return centroid + (points - centroid) * expansion


def _cross(origin, a_point, b_point):
    return (a_point[0] - origin[0]) * (b_point[1] - origin[1]) - (a_point[1] - origin[1]) * (b_point[0] - origin[0])


def _convex_hull(points):
    unique = sorted(set((float(x_value), float(y_value)) for x_value, y_value in points))
    if len(unique) <= 1:
        return np.asarray(unique, dtype=float)

    lower = []
    for point in unique:
        while len(lower) >= 2 and _cross(lower[-2], lower[-1], point) <= 0:
            lower.pop()
        lower.append(point)

    upper = []
    for point in reversed(unique):
        while len(upper) >= 2 and _cross(upper[-2], upper[-1], point) <= 0:
            upper.pop()
        upper.append(point)

    hull = lower[:-1] + upper[:-1]
    return np.asarray(hull, dtype=float)


def _sort_by_angle(points):
    centroid = points.mean(axis=0)
    angles = np.arctan2(points[:, 1] - centroid[1], points[:, 0] - centroid[0])
    return points[np.argsort(angles)]


def _chaikin(points, iterations=3):
    points = np.asarray(points, dtype=float)
    if len(points) < 3:
        return points

    current = np.vstack([points, points[0]])
    for _ in range(iterations):
        refined = []
        for index in range(len(current) - 1):
            left = current[index]
            right = current[index + 1]
            refined.append(0.75 * left + 0.25 * right)
            refined.append(0.25 * left + 0.75 * right)
        current = np.asarray(refined, dtype=float)
        current = np.vstack([current, current[0]])

    return current[:-1]


def smooth_region_polygon(points, expansion=1.12, smoothing=0.08, samples=200):
    points = np.asarray(points, dtype=float)
    if len(points) < 3:
        return points

    hull = _convex_hull(points)
    if len(hull) < 3:
        hull = _sort_by_angle(points)

    hull = _expand_polygon(hull, expansion=expansion)
    if len(hull) < 3:
        return hull

    return _chaikin(hull, iterations=3)


def _load_overlay_points(paths):
    frames = []
    for path in paths:
        frame = pd.read_csv(path)
        if not {"label", "RC1", "RC2"}.issubset(frame.columns):
            raise ValueError("Point file {} must contain label, RC1, RC2 columns".format(path))
        frames.append(frame[["label", "RC1", "RC2"]].copy())
    if not frames:
        return pd.DataFrame(columns=["label", "RC1", "RC2"])
    return pd.concat(frames, ignore_index=True)


def _filter_points(frame, exclude_labels=None):
    if frame is None:
        return frame
    exclude_labels = {label for label in (exclude_labels or []) if label}
    if not exclude_labels or len(frame) == 0:
        return frame
    return frame.loc[~frame["label"].isin(exclude_labels)].copy()


def _template_data_to_pixels(frame, spec=None):
    spec = spec or TEMPLATE_2023_SPEC
    transformed = frame.copy()
    x_fraction = (transformed["RC1"] - spec["data_x_min"]) / (spec["data_x_max"] - spec["data_x_min"])
    y_fraction = (transformed["RC2"] - spec["data_y_min"]) / (spec["data_y_max"] - spec["data_y_min"])
    transformed["x_px"] = spec["axis_left_px"] + x_fraction * (spec["axis_right_px"] - spec["axis_left_px"])
    transformed["y_px"] = spec["axis_bottom_px"] - y_fraction * (spec["axis_bottom_px"] - spec["axis_top_px"])
    return transformed


def _template_pixels_to_data(points, spec=None):
    spec = spec or TEMPLATE_2023_SPEC
    points = np.asarray(points, dtype=float)
    x_fraction = (points[:, 0] - spec["axis_left_px"]) / (spec["axis_right_px"] - spec["axis_left_px"])
    y_fraction = (spec["axis_bottom_px"] - points[:, 1]) / (spec["axis_bottom_px"] - spec["axis_top_px"])
    x_values = spec["data_x_min"] + x_fraction * (spec["data_x_max"] - spec["data_x_min"])
    y_values = spec["data_y_min"] + y_fraction * (spec["data_y_max"] - spec["data_y_min"])
    return np.column_stack([x_values, y_values])


def _wrap_angle_difference(values, reference):
    return (values - reference + np.pi) % (2 * np.pi) - np.pi


def _resample_closed_curve(points, samples=500):
    points = np.asarray(points, dtype=float)
    if len(points) < 3:
        return points
    closed = np.vstack([points, points[0]])
    diffs = np.diff(closed, axis=0)
    lengths = np.sqrt((diffs ** 2).sum(axis=1))
    cumulative = np.concatenate([[0.0], np.cumsum(lengths)])
    total = cumulative[-1]
    if total == 0:
        return points
    targets = np.linspace(0.0, total, samples, endpoint=False)
    x_values = np.interp(targets, cumulative, closed[:, 0])
    y_values = np.interp(targets, cumulative, closed[:, 1])
    return np.column_stack([x_values, y_values])


def _smooth_closed_curve(points, window=11, passes=2):
    points = np.asarray(points, dtype=float)
    if len(points) < 3:
        return points
    current = points.copy()
    half_window = window // 2
    for _ in range(passes):
        padded = np.vstack([current[-half_window:], current, current[:half_window]])
        smoothed = []
        for idx in range(len(current)):
            smoothed.append(padded[idx : idx + window].mean(axis=0))
        current = np.asarray(smoothed)
    return current


def _radial_expand_polygon(polygon, factor=1.02, pad=0.0):
    polygon = np.asarray(polygon, dtype=float)
    centroid = polygon.mean(axis=0)
    delta = polygon - centroid
    radii = np.sqrt((delta ** 2).sum(axis=1))
    safe = np.where(radii == 0.0, 1.0, radii)
    expanded = centroid + delta * ((radii * factor + pad) / safe)[:, None]
    return expanded


def _expand_polygon_to_include_points(polygon, points, margin=0.08, sigma=0.28):
    polygon = np.asarray(polygon, dtype=float)
    points = np.asarray(points, dtype=float)
    if len(points) == 0:
        return polygon

    centroid = polygon.mean(axis=0)
    vertex_delta = polygon - centroid
    vertex_angles = np.arctan2(vertex_delta[:, 1], vertex_delta[:, 0])
    vertex_radii = np.sqrt((vertex_delta ** 2).sum(axis=1))
    updated_radii = vertex_radii.copy()

    for point in points:
        point_delta = point - centroid
        point_radius = float(np.sqrt((point_delta ** 2).sum()))
        point_angle = float(np.arctan2(point_delta[1], point_delta[0]))
        differences = _wrap_angle_difference(vertex_angles, point_angle)
        influence = np.exp(-0.5 * (differences / sigma) ** 2)
        desired_radius = point_radius + margin
        updated_radii = np.maximum(updated_radii, vertex_radii + influence * np.maximum(0.0, desired_radius - vertex_radii))

    safe = np.where(vertex_radii == 0.0, 1.0, vertex_radii)
    expanded = centroid + vertex_delta * (updated_radii / safe)[:, None]
    return expanded


def _envelope_polygon_from_shape_and_points(base_polygon, include_points, margin=0.08, samples=720, bandwidth=0.24):
    base_polygon = np.asarray(base_polygon, dtype=float)
    include_points = np.asarray(include_points, dtype=float)
    centroid = base_polygon.mean(axis=0)

    base_delta = base_polygon - centroid
    base_angles = np.arctan2(base_delta[:, 1], base_delta[:, 0])
    base_radii = np.sqrt((base_delta ** 2).sum(axis=1))

    point_angles = np.array([])
    point_radii = np.array([])
    if len(include_points):
        point_delta = include_points - centroid
        point_angles = np.arctan2(point_delta[:, 1], point_delta[:, 0])
        point_radii = np.sqrt((point_delta ** 2).sum(axis=1)) + margin

    sample_angles = np.linspace(-np.pi, np.pi, samples, endpoint=False)
    radii = np.zeros_like(sample_angles)

    for idx, angle in enumerate(sample_angles):
        base_diff = np.abs(_wrap_angle_difference(base_angles, angle))
        nearby = base_radii[base_diff <= bandwidth]
        if len(nearby):
            base_radius = float(nearby.max())
        else:
            base_radius = float(base_radii[np.argmin(base_diff)])

        final_radius = base_radius
        if len(point_radii):
            point_diff = np.abs(_wrap_angle_difference(point_angles, angle))
            point_nearby = point_radii[point_diff <= bandwidth * 1.25]
            if len(point_nearby):
                final_radius = max(final_radius, float(point_nearby.max()))
        radii[idx] = final_radius

    kernel = np.array([1, 2, 3, 4, 3, 2, 1], dtype=float)
    kernel = kernel / kernel.sum()
    padded = np.r_[radii[-3:], radii, radii[:3]]
    smoothed = np.convolve(padded, kernel, mode="same")[3:-3]

    x_values = centroid[0] + smoothed * np.cos(sample_angles)
    y_values = centroid[1] + smoothed * np.sin(sample_angles)
    return np.column_stack([x_values, y_values])


def _format_model_label(label):
    if label.lower().startswith("gpt-"):
        return "GPT-" + label[4:]
    return label


def _model_label_bbox():
    return {
        "boxstyle": "round,pad=0.24,rounding_size=0.18",
        "facecolor": (1.0, 1.0, 1.0, 0.92),
        "edgecolor": "#555555",
        "linewidth": 0.9,
    }


def _estimate_template_label_box(label, font_size):
    text = str(label)
    width = max(72.0, font_size * (0.68 * len(text) + 2.8))
    height = max(24.0, font_size * 1.9)
    return width, height


def _rectangles_overlap(rect_a, rect_b, pad=0.0):
    left_a, top_a, right_a, bottom_a = rect_a
    left_b, top_b, right_b, bottom_b = rect_b
    return not (
        right_a + pad <= left_b
        or right_b + pad <= left_a
        or bottom_a + pad <= top_b
        or bottom_b + pad <= top_a
    )


def _triangle_rect(x_value, y_value, half_size=18.0):
    return (
        x_value - half_size,
        y_value - half_size,
        x_value + half_size,
        y_value + half_size,
    )


def _place_template_overlay_labels(overlay, width, height):
    if overlay is None or len(overlay) == 0:
        return {}

    placements = {}
    occupied_rects = []
    triangle_rects = {
        index: _triangle_rect(float(row["x_px"]), float(row["y_px"]))
        for index, row in overlay.iterrows()
    }

    ordered_indices = sorted(
        overlay.index.tolist(),
        key=lambda idx: (float(overlay.loc[idx, "y_px"]), -float(overlay.loc[idx, "x_px"])),
    )
    for index in ordered_indices:
        row = overlay.loc[index]
        display_label = _format_model_label(row["label"])
        font_size = MODEL_LABEL_FONT_SIZES.get(display_label, MODEL_LABEL_FONT_SIZES.get(row["label"], 18))
        box_width, box_height = _estimate_template_label_box(display_label, font_size)
        box_width *= 0.90
        box_height *= 0.90
        x_value = float(row["x_px"])
        y_value = float(row["y_px"])

        chosen = None
        x_offsets = [0.0, -8.0, 8.0, -16.0, 16.0, -24.0, 24.0, -32.0, 32.0, -40.0, 40.0]
        y_offsets = [32.0, 42.0, 54.0, 66.0, 80.0, 96.0, 114.0, 134.0, 156.0, 180.0]

        for y_offset in y_offsets:
            for x_offset in x_offsets:
                center_x = np.clip(x_value + x_offset, 10.0 + box_width / 2.0, width - 10.0 - box_width / 2.0)
                center_y = np.clip(y_value + y_offset, 10.0 + box_height / 2.0, height - 10.0 - box_height / 2.0)
                rect = (
                    center_x - box_width / 2.0,
                    center_y - box_height / 2.0,
                    center_x + box_width / 2.0,
                    center_y + box_height / 2.0,
                )
                triangle_hit = any(_rectangles_overlap(rect, tri_rect, pad=4.0) for tri_rect in triangle_rects.values())
                if triangle_hit:
                    continue
                box_hit = any(_rectangles_overlap(rect, other_rect, pad=4.0) for other_rect in occupied_rects)
                if box_hit:
                    continue
                score = abs(x_offset) * 3.0 + y_offset
                candidate = {
                    "score": score,
                    "text_x": center_x,
                    "text_y": center_y,
                    "ha": "center",
                    "va": "center",
                    "font_size": font_size,
                    "display_label": display_label,
                    "rect": rect,
                    "anchor_x": x_value,
                    "anchor_y": y_value,
                }
                if chosen is None or candidate["score"] < chosen["score"]:
                    chosen = candidate

        if chosen is None:
            center_x = np.clip(x_value, 10.0 + box_width / 2.0, width - 10.0 - box_width / 2.0)
            center_y = np.clip(y_value + 180.0, 10.0 + box_height / 2.0, height - 10.0 - box_height / 2.0)
            rect = (
                center_x - box_width / 2.0,
                center_y - box_height / 2.0,
                center_x + box_width / 2.0,
                center_y + box_height / 2.0,
            )
            chosen = {
                "text_x": center_x,
                "text_y": center_y,
                "ha": "center",
                "va": "center",
                "font_size": font_size,
                "display_label": display_label,
                "rect": rect,
                "anchor_x": x_value,
                "anchor_y": y_value,
            }

        occupied_rects.append(chosen["rect"])
        placements[index] = chosen
    return placements


def _category_expansion_params(category):
    params = {
        "factor": 1.01,
        "pad": 0.01,
        "margin": 0.08,
        "sigma": 0.32,
    }
    if category == "West & South Asia":
        params.update({"factor": 1.03, "pad": 0.03, "margin": 0.18, "sigma": 0.45})
    if category in ("English-Speaking", "Protestant Europe", "Confucian"):
        params.update({"margin": 0.10})
    return params


def _extract_longest_contour(mask):
    fig, ax = plt.subplots()
    contour_set = ax.contour(mask.astype(float), levels=[0.5])
    segments = contour_set.allsegs[0]
    plt.close(fig)
    if not segments:
        raise ValueError("No contour segments found")
    return max(segments, key=len)


def _fill_mask_holes(mask):
    mask = np.asarray(mask, dtype=bool)
    if not mask.any():
        return mask

    ys, xs = np.where(mask)
    pad = 4
    y0 = max(0, int(ys.min()) - pad)
    y1 = min(mask.shape[0], int(ys.max()) + pad + 1)
    x0 = max(0, int(xs.min()) - pad)
    x1 = min(mask.shape[1], int(xs.max()) + pad + 1)
    cropped = mask[y0:y1, x0:x1].copy()

    inverse = ~cropped
    visited = np.zeros_like(inverse, dtype=bool)
    queue = deque()

    height, width = inverse.shape
    for x_value in range(width):
        if inverse[0, x_value] and not visited[0, x_value]:
            visited[0, x_value] = True
            queue.append((0, x_value))
        if inverse[height - 1, x_value] and not visited[height - 1, x_value]:
            visited[height - 1, x_value] = True
            queue.append((height - 1, x_value))
    for y_value in range(height):
        if inverse[y_value, 0] and not visited[y_value, 0]:
            visited[y_value, 0] = True
            queue.append((y_value, 0))
        if inverse[y_value, width - 1] and not visited[y_value, width - 1]:
            visited[y_value, width - 1] = True
            queue.append((y_value, width - 1))

    neighbors = [(-1, 0), (1, 0), (0, -1), (0, 1)]
    while queue:
        y_value, x_value = queue.popleft()
        for dy_value, dx_value in neighbors:
            ny_value = y_value + dy_value
            nx_value = x_value + dx_value
            if 0 <= ny_value < height and 0 <= nx_value < width and inverse[ny_value, nx_value] and not visited[ny_value, nx_value]:
                visited[ny_value, nx_value] = True
                queue.append((ny_value, nx_value))

    holes = inverse & ~visited
    filled = cropped | holes

    result = mask.copy()
    result[y0:y1, x0:x1] = filled
    return result


def _majority_smooth(mask, iterations=2):
    mask = np.asarray(mask, dtype=bool)
    current = mask.copy()
    for _ in range(iterations):
        padded = np.pad(current.astype(np.uint8), 1, mode="edge")
        total = (
            padded[:-2, :-2] + padded[:-2, 1:-1] + padded[:-2, 2:]
            + padded[1:-1, :-2] + padded[1:-1, 1:-1] + padded[1:-1, 2:]
            + padded[2:, :-2] + padded[2:, 1:-1] + padded[2:, 2:]
        )
        current = total >= 5
    return current


def _dilate_mask(mask, iterations=1):
    current = np.asarray(mask, dtype=bool)
    for _ in range(iterations):
        padded = np.pad(current, 1, mode="edge")
        total = (
            padded[:-2, :-2] | padded[:-2, 1:-1] | padded[:-2, 2:]
            | padded[1:-1, :-2] | padded[1:-1, 1:-1] | padded[1:-1, 2:]
            | padded[2:, :-2] | padded[2:, 1:-1] | padded[2:, 2:]
        )
        current = total
    return current


def extract_template_polygons(template_image_path, spec=None):
    spec = spec or TEMPLATE_2023_SPEC
    image = np.asarray(Image.open(template_image_path).convert("RGB"))
    plot_crop = image[
        int(spec["axis_top_px"]) - 40 : int(spec["axis_bottom_px"]) + 40,
        int(spec["axis_left_px"]) - 40 : int(spec["axis_right_px"]) + 40,
    ]
    y_offset = int(spec["axis_top_px"]) - 40
    x_offset = int(spec["axis_left_px"]) - 40

    reference_colors = {
        "African-Islamic": (185, 181, 189),
        "Latin America": (153, 177, 181),
        "Orthodox Europe": (217, 90, 62),
        "Confucian": (231, 147, 76),
        "West & South Asia": (253, 202, 89),
        "Catholic Europe": (157, 195, 24),
        "English-Speaking": (201, 191, 33),
        "Protestant Europe": (228, 225, 65),
    }
    tolerances = {
        "African-Islamic": 20.0,
        "Latin America": 18.0,
        "Orthodox Europe": 20.0,
        "Confucian": 18.0,
        "West & South Asia": 22.0,
        "Catholic Europe": 20.0,
        "English-Speaking": 18.0,
        "Protestant Europe": 16.0,
    }

    polygons = {}
    for category in CULTURE_ZONE_DRAW_ORDER:
        color = np.asarray(reference_colors[category], dtype=float)
        distance = np.sqrt(((plot_crop.astype(float) - color) ** 2).sum(axis=2))
        mask = distance < tolerances[category]
        mask = _fill_mask_holes(mask)
        mask = _majority_smooth(mask, iterations=2)
        mask = _dilate_mask(mask, iterations=4)
        contour = _extract_longest_contour(mask)
        contour[:, 0] = contour[:, 0] + x_offset
        contour[:, 1] = contour[:, 1] + y_offset
        polygons[category] = _template_pixels_to_data(contour[:, [0, 1]], spec=spec)
    return polygons


def plot_culture_map(human_df, overlay_point_paths=None, paper_points=None, output_path=None, title=None, exclude_labels=None):
    overlay_point_paths = overlay_point_paths or []
    overlay = _filter_points(_load_overlay_points(overlay_point_paths), exclude_labels=exclude_labels)
    paper_points = _filter_points(paper_points, exclude_labels=exclude_labels)
    if paper_points is not None and len(paper_points):
        overlay = pd.concat([paper_points[["label", "RC1", "RC2"]], overlay], ignore_index=True)
    overlay = overlay.drop_duplicates(subset=["label"], keep="last")

    fig, ax = plt.subplots(figsize=(14, 10))
    ax.set_facecolor("white")

    for category in CULTURE_ZONE_DRAW_ORDER:
        category_points = human_df.loc[human_df["Category"] == category, ["RC1_final", "RC2_final"]].to_numpy()
        if len(category_points) < 3:
            continue
        polygon = smooth_region_polygon(category_points)
        ax.fill(
            polygon[:, 0],
            polygon[:, 1],
            facecolor=CATEGORY_COLORS[category],
            edgecolor="black",
            linewidth=2.2,
            alpha=0.88,
            zorder=0,
        )

    ax.scatter(human_df["RC1_final"], human_df["RC2_final"], s=32, color="black", zorder=3)
    for _, row in human_df.iterrows():
        ax.text(row["RC1_final"] + 0.03, row["RC2_final"] + 0.03, row["country"], fontsize=8.3, color="black", zorder=4)

    for category, (x_pos, y_pos) in CATEGORY_LABEL_POSITIONS.items():
        label = category.replace("-", "-\n") if category == "African-Islamic" else category.replace(" & ", " &\n").replace(" ", "\n", 1) if category in ("West & South Asia", "Orthodox Europe", "Catholic Europe", "Protestant Europe", "English-Speaking") else category
        ax.text(
            x_pos,
            y_pos,
            label,
            fontsize=17,
            fontstyle="italic",
            fontweight="semibold",
            ha="center",
            va="center",
            color="black",
            zorder=2,
        )

    if len(overlay):
        ax.scatter(
            overlay["RC1"],
            overlay["RC2"],
            s=260,
            marker="^",
            color="#c98ae6",
            edgecolors="black",
            linewidths=1.4,
            zorder=6,
        )
        for _, row in overlay.iterrows():
            display_label = _format_model_label(row["label"])
            nudge_x, nudge_y = MODEL_LABEL_NUDGES.get(display_label, MODEL_LABEL_NUDGES.get(row["label"], (0.08, 0.08)))
            ax.text(
                row["RC1"] + nudge_x,
                row["RC2"] + nudge_y,
                display_label,
                fontsize=12.5,
                color="black",
                bbox=_model_label_bbox(),
                zorder=7,
            )

    x_minimum = min(-2.5, float(human_df["RC1_final"].min()) - 0.2)
    x_maximum = max(3.5, float(human_df["RC1_final"].max()) + 0.2)
    y_minimum = min(-2.5, float(human_df["RC2_final"].min()) - 0.2)
    y_maximum = max(2.0, float(human_df["RC2_final"].max()) + 0.2)
    if len(overlay):
        x_minimum = min(x_minimum, float(overlay["RC1"].min()) - 0.3)
        x_maximum = max(x_maximum, float(overlay["RC1"].max()) + 0.3)
        y_minimum = min(y_minimum, float(overlay["RC2"].min()) - 0.3)
        y_maximum = max(y_maximum, float(overlay["RC2"].max()) + 0.3)

    ax.set_xlim(x_minimum, x_maximum)
    ax.set_ylim(y_minimum, y_maximum)
    ax.set_xlabel("Survival vs. Self-Expression Values", fontsize=18, fontstyle="italic", fontweight="semibold")
    ax.set_ylabel("Traditional vs. Secular Values", fontsize=18, fontstyle="italic", fontweight="semibold")
    if title:
        ax.set_title(title, fontsize=18)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.grid(False)
    fig.tight_layout()

    if output_path is not None:
        output_path = Path(output_path)
        ensure_dir(output_path.parent)
        fig.savefig(str(output_path), dpi=300, bbox_inches="tight")

    return fig, ax


def plot_culture_map_with_template_shapes(
    human_df,
    template_image_path,
    overlay_point_paths=None,
    paper_points=None,
    output_path=None,
    exclude_labels=None,
):
    overlay_point_paths = overlay_point_paths or []
    overlay = _filter_points(_load_overlay_points(overlay_point_paths), exclude_labels=exclude_labels)
    paper_points = _filter_points(paper_points, exclude_labels=exclude_labels)
    if paper_points is not None and len(paper_points):
        overlay = pd.concat([paper_points[["label", "RC1", "RC2"]], overlay], ignore_index=True)
    overlay = overlay.drop_duplicates(subset=["label"], keep="last")
    polygons = extract_template_polygons(template_image_path)
    category_points = {
        category: human_df.loc[human_df["Category"] == category, ["RC1_final", "RC2_final"]].to_numpy()
        for category in CULTURE_ZONE_DRAW_ORDER
    }
    for category in CULTURE_ZONE_DRAW_ORDER:
        polygon = polygons[category]
        params = _category_expansion_params(category)
        polygon = _radial_expand_polygon(polygon, factor=params["factor"], pad=params["pad"])
        polygon = _envelope_polygon_from_shape_and_points(
            polygon,
            category_points[category],
            margin=params["margin"],
            samples=720,
            bandwidth=params["sigma"],
        )
        polygon = _resample_closed_curve(polygon, samples=500)
        polygon = _smooth_closed_curve(polygon, window=11, passes=3)
        polygons[category] = polygon

    fig, ax = plt.subplots(figsize=(14, 10))
    ax.set_facecolor("white")

    for category in CULTURE_ZONE_DRAW_ORDER:
        polygon = polygons[category]
        ax.fill(
            polygon[:, 0],
            polygon[:, 1],
            facecolor=CATEGORY_COLORS[category],
            edgecolor="none",
            alpha=0.92,
            zorder=0,
        )
    for category in CULTURE_ZONE_DRAW_ORDER:
        polygon = polygons[category]
        ax.plot(
            np.r_[polygon[:, 0], polygon[0, 0]],
            np.r_[polygon[:, 1], polygon[0, 1]],
            color="black",
            linewidth=2.35,
            zorder=1,
            solid_joinstyle="round",
            solid_capstyle="round",
            antialiased=True,
        )

    ax.scatter(human_df["RC1_final"], human_df["RC2_final"], s=36, color="black", zorder=3)
    for _, row in human_df.iterrows():
        ax.text(
            row["RC1_final"] + 0.03,
            row["RC2_final"] + 0.03,
            row["country"],
            fontsize=8.4,
            fontstyle="italic" if row["country"] in MUSLIM_COUNTRIES else "normal",
            color="black",
            zorder=4,
        )

    for category, (x_pos, y_pos) in CATEGORY_LABEL_POSITIONS.items():
        label = category
        if category == "African-Islamic":
            label = "African-\nIslamic"
        elif category == "Latin America":
            label = "Latin\nAmerica"
        elif category == "Orthodox Europe":
            label = "Orthodox\nEurope"
        elif category == "West & South Asia":
            label = "West&South\nAsia"
        elif category == "Catholic Europe":
            label = "Catholic\nEurope"
        elif category == "English-Speaking":
            label = "English-Speaking"
        elif category == "Protestant Europe":
            label = "Protestant\nEurope"

        text = ax.text(
            x_pos,
            y_pos,
            label,
            fontsize=CATEGORY_LABEL_FONT_SIZES[category],
            fontstyle="italic",
            fontweight="semibold",
            ha="center",
            va="center",
            color="black",
            zorder=2,
        )
        text.set_path_effects([patheffects.withStroke(linewidth=2.0, foreground=(1, 1, 1, 0.20))])

    if len(overlay):
        ax.scatter(
            overlay["RC1"],
            overlay["RC2"],
            s=360,
            marker="^",
            color="#d5a3ea",
            edgecolors="#4f425b",
            linewidths=1.4,
            zorder=6,
        )
        for _, row in overlay.iterrows():
            display_label = _format_model_label(row["label"])
            nudge_x, nudge_y = MODEL_LABEL_NUDGES.get(display_label, MODEL_LABEL_NUDGES.get(row["label"], (0.08, 0.08)))
            ax.text(
                row["RC1"] + nudge_x,
                row["RC2"] + nudge_y,
                display_label,
                fontsize=MODEL_LABEL_FONT_SIZES.get(display_label, MODEL_LABEL_FONT_SIZES.get(row["label"], 18)),
                color="black",
                bbox=_model_label_bbox(),
                zorder=7,
            )

    x_minimum = -2.5
    x_maximum = 3.5
    y_minimum = -2.5
    y_maximum = 2.0
    if len(overlay):
        y_maximum = max(y_maximum, float(overlay["RC2"].max()) + 0.25)

    ax.set_xlim(x_minimum, x_maximum)
    ax.set_ylim(y_minimum, y_maximum)
    ax.set_xticks(np.arange(-2.5, 3.51, 0.5))
    ax.set_yticks(np.arange(-2.5, y_maximum + 0.001, 0.5))
    ax.xaxis.set_major_formatter(FormatStrFormatter("%.2f"))
    ax.yaxis.set_major_formatter(FormatStrFormatter("%.2f"))
    ax.set_xlabel("Survival vs. Self-Expression Values", fontsize=18, fontstyle="italic", fontweight="semibold")
    ax.set_ylabel("Traditional vs. Secular Values", fontsize=18, fontstyle="italic", fontweight="semibold")
    ax.text(
        0.01,
        0.03,
        "Muslim countries in italic",
        transform=ax.transAxes,
        fontsize=10.5,
        bbox={"facecolor": "white", "edgecolor": "#999999", "boxstyle": "square,pad=0.25"},
        zorder=10,
    )
    ax.text(
        0.985,
        0.06,
        "Source: World Values Survey &\nEuropean Values Study\n(2005-2022)",
        transform=ax.transAxes,
        ha="right",
        va="bottom",
        fontsize=10.5,
        color="black",
        zorder=10,
    )
    ax.text(
        0.985,
        0.007,
        "www.worldvaluessurvey.org\nhttps://europeanvaluesstudy.eu/",
        transform=ax.transAxes,
        ha="right",
        va="bottom",
        fontsize=10.5,
        color="#2663d8",
        zorder=10,
    )
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.spines["left"].set_color("#c9c9c9")
    ax.spines["bottom"].set_color("#c9c9c9")
    ax.spines["left"].set_linewidth(1.0)
    ax.spines["bottom"].set_linewidth(1.0)
    ax.tick_params(axis="both", which="major", labelsize=10.5, color="#c9c9c9", width=1.0, length=3)
    ax.grid(False)
    fig.tight_layout()

    if output_path is not None:
        output_path = Path(output_path)
        ensure_dir(output_path.parent)
        fig.savefig(str(output_path), dpi=300, bbox_inches="tight")
    return fig, ax


def plot_template_map(
    template_image_path,
    overlay_point_paths=None,
    paper_points=None,
    output_path=None,
    hide_title=False,
    hide_source_block=False,
    add_axis_end_labels=False,
    add_zero_guides=False,
    exclude_labels=None,
):
    overlay_point_paths = overlay_point_paths or []
    overlay = _filter_points(_load_overlay_points(overlay_point_paths), exclude_labels=exclude_labels)
    paper_points = _filter_points(paper_points, exclude_labels=exclude_labels)
    if paper_points is not None and len(paper_points):
        overlay = pd.concat([paper_points[["label", "RC1", "RC2"]], overlay], ignore_index=True)
    overlay = overlay.drop_duplicates(subset=["label"], keep="last")
    overlay = _template_data_to_pixels(overlay)

    image = np.asarray(Image.open(template_image_path).convert("RGBA"))
    title_crop = 0
    if hide_title:
        title_crop = 100
        image = image[title_crop:, :, :].copy()
        if len(overlay):
            overlay = overlay.copy()
            overlay["y_px"] = overlay["y_px"] - title_crop
    if hide_source_block:
        image = image.copy()
        source_y0 = max(0, 885 - title_crop)
        source_y1 = max(0, 1050 - title_crop)
        image[source_y0:source_y1, 1250:1602, :3] = 255
        image[source_y0:source_y1, 1250:1602, 3] = 255
    height, width = image.shape[0], image.shape[1]

    top_pad = 0
    if len(overlay):
        min_overlay_y = float(overlay["y_px"].min())
        if min_overlay_y < 24:
            top_pad = int(np.ceil(24 - min_overlay_y))

    if top_pad > 0:
        padded = np.full((height + top_pad, width, 4), 255, dtype=np.uint8)
        padded[top_pad:, :, :] = image
        image = padded
        overlay = overlay.copy()
        overlay["y_px"] = overlay["y_px"] + top_pad
        height = image.shape[0]

    fig_width = 14
    fig_height = fig_width * (height / float(width))
    fig, ax = plt.subplots(figsize=(fig_width, fig_height))
    ax.imshow(image, origin="upper")
    ax.set_xlim(0, width)
    ax.set_ylim(height, 0)
    ax.axis("off")

    if len(overlay):
        label_placements = _place_template_overlay_labels(overlay, width=width, height=height)
        ax.scatter(
            overlay["x_px"],
            overlay["y_px"],
            s=360,
            marker="^",
            color="#d5a3ea",
            edgecolors="black",
            linewidths=1.6,
            zorder=6,
        )
        for _, row in overlay.iterrows():
            placement = label_placements[row.name]

            line_start_x = float(row["x_px"])
            line_start_y = float(row["y_px"]) + 18.0
            line_end_x = np.clip(line_start_x, placement["rect"][0] + 10.0, placement["rect"][2] - 10.0)
            line_end_y = placement["rect"][1] - 4.0
            if line_end_y > line_start_y + 2.0:
                ax.plot(
                    [line_start_x, line_end_x],
                    [line_start_y, line_end_y],
                    color="#666666",
                    linewidth=1.0,
                    linestyle=(0, (1, 2)),
                    zorder=6.4,
                )

            ax.text(
                placement["text_x"],
                placement["text_y"],
                placement["display_label"],
                fontsize=placement["font_size"],
                color="black",
                ha=placement["ha"],
                va=placement["va"],
                bbox=_model_label_bbox(),
                zorder=7,
            )

    if add_zero_guides:
        zero_frame = _template_data_to_pixels(pd.DataFrame([{"RC1": 0.0, "RC2": 0.0}]))
        x_zero = float(zero_frame["x_px"].iloc[0])
        y_zero = float(zero_frame["y_px"].iloc[0]) - title_crop + top_pad
        x_left = float(TEMPLATE_2023_SPEC["axis_left_px"])
        x_right = float(TEMPLATE_2023_SPEC["axis_right_px"])
        y_top = float(TEMPLATE_2023_SPEC["axis_top_px"]) - title_crop + top_pad
        y_bottom = float(TEMPLATE_2023_SPEC["axis_bottom_px"]) - title_crop + top_pad
        guide_style = {"color": "#4a4a4a", "linewidth": 1.8, "alpha": 0.85, "linestyle": (0, (1, 3)), "zorder": 5}
        ax.plot([x_zero, x_zero], [y_top, y_bottom], **guide_style)
        ax.plot([x_left, x_right], [y_zero, y_zero], **guide_style)

    if add_axis_end_labels:
        ax.text(
            205,
            top_pad + 165,
            "Strong\nSecular\nValues",
            ha="left",
            va="top",
            fontsize=22,
            fontweight="bold",
            color="black",
            zorder=8,
        )
        ax.text(
            1455,
            top_pad + 955,
            "Strong\nSelf-Expression\nValues",
            ha="right",
            va="bottom",
            fontsize=22,
            fontweight="bold",
            color="black",
            zorder=8,
        )

    fig.tight_layout(pad=0)
    if output_path is not None:
        output_path = Path(output_path)
        ensure_dir(output_path.parent)
        fig.savefig(str(output_path), dpi=300, bbox_inches="tight", pad_inches=0)
    return fig, ax
