#!/usr/bin/env python3
import argparse
from collections import OrderedDict
from pathlib import Path

import cv2
import numpy as np
import rosbag
import sensor_msgs.point_cloud2 as pc2


VALUE_TOPIC = "/grid_map/value_map"
FREE_TOPIC = "/grid_map/free"
UNKNOWN_TOPIC = "/grid_map/unknown"
OCC_TOPIC = "/grid_map/occupied_inflate"
FRONTIER_TOPIC = "/planning_vis/frontier"
VIEWPOINT_TOPIC = "/planning_vis/viewpoints"
TOPO_TOPIC = "/planning_vis/topo_path"
ROBOT_TOPIC = "/robot"
RESULT_TOPIC = "/ros/expl_result"
STATE_TOPIC = "/ros/expl_state"
ACTION_TOPIC = "/habitat/plan_action"


def point_key(x, y, scale=100):
    return (int(round(float(x) * scale)), int(round(float(y) * scale)))


def read_pc_xy(msg, with_intensity=False):
    pts = []
    fields = ("x", "y", "intensity") if with_intensity else ("x", "y")
    for p in pc2.read_points(msg, field_names=fields, skip_nans=True):
        if with_intensity:
            x, y, intensity = p
            pts.append((float(x), float(y), float(intensity)))
        else:
            x, y = p
            pts.append((float(x), float(y)))
    return pts


def marker_points(msg):
    pts = [(float(p.x), float(p.y)) for p in msg.points]
    if not pts and hasattr(msg, "pose"):
        pts = [(float(msg.pose.position.x), float(msg.pose.position.y))]
    return pts


def read_events(bag_path):
    topics = [
        VALUE_TOPIC,
        FREE_TOPIC,
        UNKNOWN_TOPIC,
        OCC_TOPIC,
        FRONTIER_TOPIC,
        VIEWPOINT_TOPIC,
        TOPO_TOPIC,
        ROBOT_TOPIC,
        RESULT_TOPIC,
        STATE_TOPIC,
        ACTION_TOPIC,
    ]
    events = []
    all_xy = []
    with rosbag.Bag(bag_path) as bag:
        for topic, msg, t in bag.read_messages(topics=topics):
            ts = t.to_sec()
            if topic == VALUE_TOPIC:
                pts = read_pc_xy(msg, with_intensity=True)
                events.append({"t": ts, "topic": topic, "points": pts})
                all_xy.extend((x, y) for x, y, _ in pts)
            elif topic in (FREE_TOPIC, UNKNOWN_TOPIC, OCC_TOPIC):
                pts = read_pc_xy(msg)
                events.append({"t": ts, "topic": topic, "points": pts})
                all_xy.extend(pts)
            elif topic in (FRONTIER_TOPIC, VIEWPOINT_TOPIC, TOPO_TOPIC, ROBOT_TOPIC):
                pts = marker_points(msg)
                events.append(
                    {
                        "t": ts,
                        "topic": topic,
                        "ns": msg.ns,
                        "id": msg.id,
                        "action": msg.action,
                        "type": msg.type,
                        "points": pts,
                    }
                )
                all_xy.extend(pts)
            elif topic in (RESULT_TOPIC, STATE_TOPIC, ACTION_TOPIC):
                events.append({"t": ts, "topic": topic, "value": int(msg.data)})
    events.sort(key=lambda e: e["t"])
    return events, all_xy


def compute_bounds(all_xy, margin=1.0):
    if not all_xy:
        return -5.0, 5.0, -5.0, 5.0
    xs = np.array([p[0] for p in all_xy], dtype=np.float32)
    ys = np.array([p[1] for p in all_xy], dtype=np.float32)
    xmin, xmax = float(xs.min() - margin), float(xs.max() + margin)
    ymin, ymax = float(ys.min() - margin), float(ys.max() + margin)
    span = max(xmax - xmin, ymax - ymin, 1e-3)
    cx, cy = (xmin + xmax) / 2.0, (ymin + ymax) / 2.0
    return cx - span / 2.0, cx + span / 2.0, cy - span / 2.0, cy + span / 2.0


def project(x, y, bounds, size):
    xmin, xmax, ymin, ymax = bounds
    px = int((float(x) - xmin) / (xmax - xmin) * (size - 1))
    py = int((ymax - float(y)) / (ymax - ymin) * (size - 1))
    return px, py


def draw_points(img, pts, bounds, color, radius=2):
    h, w = img.shape[:2]
    for x, y in pts:
        px, py = project(x, y, bounds, min(w, h))
        if 0 <= px < w and 0 <= py < h:
            cv2.circle(img, (px, py), radius, color, -1, cv2.LINE_AA)


def draw_marker(img, marker, bounds):
    topic = marker["topic"]
    ns = marker.get("ns", "")
    pts = marker.get("points", [])
    if not pts:
        return
    if topic == FRONTIER_TOPIC:
        color = (0, 0, 255) if ns == "frontier" else (80, 80, 80)
        draw_points(img, pts, bounds, color, radius=3 if ns == "frontier" else 2)
        return
    if topic in (VIEWPOINT_TOPIC, TOPO_TOPIC):
        if ns == "next_path":
            color = (0, 60, 255)
        elif ns == "tsp_tour":
            color = (40, 170, 40)
        else:
            color = (120, 80, 20)
        for i in range(0, len(pts) - 1, 2):
            p1 = project(pts[i][0], pts[i][1], bounds, img.shape[0])
            p2 = project(pts[i + 1][0], pts[i + 1][1], bounds, img.shape[0])
            cv2.line(img, p1, p2, color, 2, cv2.LINE_AA)
        return
    if topic == ROBOT_TOPIC:
        draw_points(img, pts[:1], bounds, (255, 80, 0), radius=5)


def draw_header(img, title, t, t0, status=None):
    cv2.rectangle(img, (0, 0), (img.shape[1] - 1, 42), (255, 255, 255), -1)
    text = f"{title}  t={t - t0:.1f}s"
    if status:
        text += f"  {status}"
    cv2.putText(
        img,
        text,
        (14, 28),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.68,
        (20, 20, 20),
        2,
        cv2.LINE_AA,
    )


def draw_legend(img, entries):
    x, y = 14, img.shape[0] - 18 - 24 * (len(entries) - 1)
    for label, color in entries:
        cv2.rectangle(img, (x, y - 13), (x + 16, y + 3), color, -1)
        cv2.putText(
            img,
            label,
            (x + 24, y + 2),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.48,
            (35, 35, 35),
            1,
            cv2.LINE_AA,
        )
        y += 24


def render_semantic(events, bounds, out_path, fps=6, size=900):
    value_events = [e for e in events if e["topic"] == VALUE_TOPIC and e["points"]]
    if not value_events:
        raise RuntimeError("no /grid_map/value_map messages with points")
    writer = cv2.VideoWriter(
        str(out_path), cv2.VideoWriter_fourcc(*"mp4v"), fps, (size, size)
    )
    if not writer.isOpened():
        raise RuntimeError(f"failed to open writer: {out_path}")

    accum = np.zeros((size, size), dtype=np.float32)
    t0 = value_events[0]["t"]
    for event in value_events:
        for x, y, val in event["points"]:
            px, py = project(x, y, bounds, size)
            if 0 <= px < size and 0 <= py < size:
                accum[py, px] = max(accum[py, px], float(val))

        img = np.clip(accum, 0.0, 0.35) / 0.35
        img8 = (img * 255).astype(np.uint8)
        img8 = cv2.dilate(img8, np.ones((5, 5), np.uint8), iterations=1)
        color = cv2.applyColorMap(img8, cv2.COLORMAP_INFERNO)
        bg = np.full_like(color, 245)
        mask = img8 > 0
        bg[mask] = color[mask]
        draw_header(bg, "Semantic score map (/grid_map/value_map)", event["t"], t0)
        for i in range(180):
            v = int(i / 179 * 255)
            c = cv2.applyColorMap(np.array([[v]], dtype=np.uint8), cv2.COLORMAP_INFERNO)[
                0, 0
            ].tolist()
            cv2.line(bg, (80 + i, size - 30), (80 + i, size - 14), c, 1)
        cv2.putText(
            bg,
            "low",
            (14, size - 17),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.5,
            (70, 70, 70),
            1,
            cv2.LINE_AA,
        )
        cv2.putText(
            bg,
            "high",
            (270, size - 17),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.5,
            (70, 70, 70),
            1,
            cv2.LINE_AA,
        )
        writer.write(bg)
    writer.release()
    return len(value_events)


def render_frontier_diagnostics(events, bounds, out_path, fps=6, size=900):
    writer = cv2.VideoWriter(
        str(out_path), cv2.VideoWriter_fourcc(*"mp4v"), fps, (size, size)
    )
    if not writer.isOpened():
        raise RuntimeError(f"failed to open writer: {out_path}")

    free = OrderedDict()
    unknown = OrderedDict()
    occupied = OrderedDict()
    markers = {}
    status = {"action": None, "expl_state": None, "expl_result": None}
    t0 = events[0]["t"] if events else 0.0
    final_t = events[-1]["t"] if events else t0
    next_render_t = t0
    render_interval = 1.0 / max(float(fps), 1e-3)
    frame_count = 0

    for event in events:
        topic = event["topic"]
        if topic == FREE_TOPIC:
            for x, y in event["points"]:
                key = point_key(x, y)
                unknown.pop(key, None)
                occupied.pop(key, None)
                free[key] = (x, y)
        elif topic == UNKNOWN_TOPIC:
            for x, y in event["points"]:
                key = point_key(x, y)
                free.pop(key, None)
                occupied.pop(key, None)
                unknown[key] = (x, y)
        elif topic == OCC_TOPIC:
            for x, y in event["points"]:
                key = point_key(x, y)
                free.pop(key, None)
                unknown.pop(key, None)
                occupied[key] = (x, y)
        elif topic in (FRONTIER_TOPIC, VIEWPOINT_TOPIC, TOPO_TOPIC, ROBOT_TOPIC):
            mkey = (topic, event.get("ns", ""), event.get("id", 0))
            if event.get("action") in (2, 3):
                markers.pop(mkey, None)
            elif event.get("points"):
                markers[mkey] = event
        elif topic == ACTION_TOPIC:
            status["action"] = event["value"]
        elif topic == STATE_TOPIC:
            status["expl_state"] = event["value"]
        elif topic == RESULT_TOPIC:
            status["expl_result"] = event["value"]
        else:
            continue

        should_render = event["t"] + 1e-6 >= next_render_t or event["t"] >= final_t - 1e-6
        if not should_render:
            continue
        while next_render_t <= event["t"] + 1e-6:
            next_render_t += render_interval

        img = np.full((size, size, 3), 248, dtype=np.uint8)
        draw_points(img, unknown.values(), bounds, (218, 218, 218), radius=1)
        draw_points(img, free.values(), bounds, (230, 248, 230), radius=1)
        draw_points(img, occupied.values(), bounds, (30, 30, 30), radius=2)
        for marker in markers.values():
            draw_marker(img, marker, bounds)
        status_text = (
            f"action={status['action']} expl_state={status['expl_state']} "
            f"expl_result={status['expl_result']}"
        )
        draw_header(img, "Frontier / occupancy diagnostic map", event["t"], t0, status_text)
        draw_legend(
            img,
            [
                ("unknown", (218, 218, 218)),
                ("free", (230, 248, 230)),
                ("occupied_inflate", (30, 30, 30)),
                ("frontier", (0, 0, 255)),
                ("dormant_frontier", (80, 80, 80)),
                ("next_path", (0, 60, 255)),
                ("tsp_tour", (40, 170, 40)),
            ],
        )
        writer.write(img)
        frame_count += 1

    writer.release()
    return frame_count


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--bag", required=True)
    parser.add_argument("--out-dir", required=True)
    parser.add_argument("--size", type=int, default=900)
    args = parser.parse_args()

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    events, all_xy = read_events(args.bag)
    bounds = compute_bounds(all_xy)
    semantic_out = out_dir / "ep_0025_couch_semantic_score_map.mp4"
    diagnostic_out = out_dir / "ep_0025_couch_frontier_diagnostic_map.mp4"
    semantic_count = render_semantic(events, bounds, semantic_out, size=args.size)
    diagnostic_count = render_frontier_diagnostics(events, bounds, diagnostic_out, size=args.size)
    print(f"events {len(events)}")
    print(f"semantic_messages {semantic_count}")
    print(f"diagnostic_frames {diagnostic_count}")
    print(f"semantic_video {semantic_out}")
    print(f"diagnostic_video {diagnostic_out}")


if __name__ == "__main__":
    main()
