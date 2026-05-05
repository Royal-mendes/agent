#!/usr/bin/env python3
import argparse
from pathlib import Path

import cv2
import numpy as np
import rosbag
import sensor_msgs.point_cloud2 as pc2


def read_value_map_frames(bag_path):
    frames = []
    with rosbag.Bag(bag_path) as bag:
        for _, msg, t in bag.read_messages(topics=["/grid_map/value_map"]):
            pts = []
            for p in pc2.read_points(
                msg, field_names=("x", "y", "intensity"), skip_nans=True
            ):
                x, y, intensity = p
                pts.append((float(x), float(y), float(intensity)))
            if pts:
                frames.append((t.to_sec(), pts))
    return frames


def render_heatmaps(frames, out_path, fps=6, size=720):
    if not frames:
        raise RuntimeError("no /grid_map/value_map messages with points")

    all_pts = [p for _, pts in frames for p in pts]
    xs = np.array([p[0] for p in all_pts], dtype=np.float32)
    ys = np.array([p[1] for p in all_pts], dtype=np.float32)
    margin = 1.0
    xmin, xmax = float(xs.min() - margin), float(xs.max() + margin)
    ymin, ymax = float(ys.min() - margin), float(ys.max() + margin)
    span = max(xmax - xmin, ymax - ymin, 1e-3)
    cx, cy = (xmin + xmax) / 2, (ymin + ymax) / 2
    xmin, xmax = cx - span / 2, cx + span / 2
    ymin, ymax = cy - span / 2, cy + span / 2

    writer = cv2.VideoWriter(
        str(out_path), cv2.VideoWriter_fourcc(*"mp4v"), fps, (size, size)
    )
    if not writer.isOpened():
        raise RuntimeError(f"failed to open video writer: {out_path}")

    accum = np.zeros((size, size), dtype=np.float32)
    start_t = frames[0][0]
    for t, pts in frames:
        for x, y, val in pts:
            px = int((x - xmin) / (xmax - xmin) * (size - 1))
            py = int((ymax - y) / (ymax - ymin) * (size - 1))
            if 0 <= px < size and 0 <= py < size:
                accum[py, px] = max(accum[py, px], float(val))

        img = np.clip(accum, 0.0, 0.35) / 0.35
        img8 = (img * 255).astype(np.uint8)
        img8 = cv2.dilate(img8, np.ones((5, 5), np.uint8), iterations=1)
        color = cv2.applyColorMap(img8, cv2.COLORMAP_INFERNO)
        mask = img8 > 0
        bg = np.full_like(color, 245)
        bg[mask] = color[mask]

        cv2.rectangle(bg, (0, 0), (size - 1, 36), (255, 255, 255), -1)
        cv2.putText(
            bg,
            f"/grid_map/value_map  t={t - start_t:.1f}s",
            (14, 25),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.65,
            (20, 20, 20),
            2,
            cv2.LINE_AA,
        )
        cv2.putText(
            bg,
            "low",
            (14, size - 18),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.55,
            (80, 80, 80),
            1,
            cv2.LINE_AA,
        )
        cv2.putText(
            bg,
            "high",
            (size - 58, size - 18),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.55,
            (80, 80, 80),
            1,
            cv2.LINE_AA,
        )
        for i in range(140):
            v = int(i / 139 * 255)
            c = cv2.applyColorMap(
                np.array([[v]], dtype=np.uint8), cv2.COLORMAP_INFERNO
            )[0, 0].tolist()
            x0 = 55 + i
            cv2.line(bg, (x0, size - 24), (x0, size - 12), c, 1)
        writer.write(bg)
    writer.release()


def hstack_videos(left_path, right_path, out_path):
    left = cv2.VideoCapture(str(left_path))
    right = cv2.VideoCapture(str(right_path))
    if not left.isOpened():
        raise RuntimeError(f"failed to open rgb video: {left_path}")
    if not right.isOpened():
        raise RuntimeError(f"failed to open semantic video: {right_path}")

    fps = left.get(cv2.CAP_PROP_FPS) or 6
    left_frames = int(left.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
    right_frames = int(right.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
    w = int(left.get(cv2.CAP_PROP_FRAME_WIDTH))
    h = int(left.get(cv2.CAP_PROP_FRAME_HEIGHT))
    writer = cv2.VideoWriter(
        str(out_path), cv2.VideoWriter_fourcc(*"mp4v"), fps, (w * 2, h)
    )
    if not writer.isOpened():
        raise RuntimeError(f"failed to open combined writer: {out_path}")

    last_right = None
    for i in range(max(left_frames, 1)):
        ok_l, lf = left.read()
        if not ok_l:
            break
        if right_frames > 0:
            target_r = min(
                right_frames - 1,
                int(i / max(left_frames - 1, 1) * max(right_frames - 1, 0)),
            )
            right.set(cv2.CAP_PROP_POS_FRAMES, target_r)
        ok_r, rf = right.read()
        if ok_r:
            last_right = rf
        elif last_right is not None:
            rf = last_right
        else:
            rf = np.full((h, w, 3), 245, dtype=np.uint8)

        rf = cv2.resize(rf, (w, h), interpolation=cv2.INTER_AREA)
        cv2.rectangle(lf, (0, 0), (w - 1, 34), (255, 255, 255), -1)
        cv2.putText(
            lf,
            "Habitat evaluation video",
            (12, 24),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.65,
            (20, 20, 20),
            2,
            cv2.LINE_AA,
        )
        cv2.rectangle(rf, (0, 0), (w - 1, 34), (255, 255, 255), -1)
        cv2.putText(
            rf,
            "Semantic score map",
            (12, 24),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.65,
            (20, 20, 20),
            2,
            cv2.LINE_AA,
        )
        writer.write(np.hstack([lf, rf]))

    writer.release()
    left.release()
    right.release()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--bag", required=True)
    parser.add_argument("--rgb-video", required=True)
    parser.add_argument("--out-dir", required=True)
    args = parser.parse_args()

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    heatmap = out_dir / "ep_0025_couch_value_map.mp4"
    combined = out_dir / "ep_0025_couch_rgb_plus_value_map.mp4"
    frames = read_value_map_frames(args.bag)
    render_heatmaps(frames, heatmap)
    hstack_videos(args.rgb_video, heatmap, combined)
    print("messages", len(frames))
    print("heatmap", heatmap)
    print("combined", combined)


if __name__ == "__main__":
    main()
