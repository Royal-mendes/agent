#!/usr/bin/env python3
import json
import math
import heapq
from collections import OrderedDict

import rosbag
import sensor_msgs.point_cloud2 as pc2


BAG_PATH = "/root/autodl-tmp/ApexNav/agent-Apexnav/repro/map_diagnostics_ep25_20260424_221004/apexnav_ep25_diagnostics.bag"
FREE = "/grid_map/free"
UNKNOWN = "/grid_map/unknown"
OCC = "/grid_map/occupied_inflate"
FRONTIER = "/planning_vis/frontier"
ROBOT = "/robot"
RESULT = "/ros/expl_result"
ACTION = "/habitat/plan_action"
STATE = "/ros/expl_state"
SCALE = 20  # 5 cm map resolution


def key(x, y):
    return (int(round(float(x) * SCALE)), int(round(float(y) * SCALE)))


def pc_xy(msg):
    return [
        (float(x), float(y))
        for x, y in pc2.read_points(msg, field_names=("x", "y"), skip_nans=True)
    ]


def marker_pts(msg, allow_pose=False):
    pts = [(float(p.x), float(p.y)) for p in msg.points]
    if allow_pose and not pts and hasattr(msg, "pose"):
        pts = [(float(msg.pose.position.x), float(msg.pose.position.y))]
    return pts


def nearest_dist(pt, store):
    if not store:
        return None
    x, y = pt
    best = 1e9
    bestp = None
    for p in store.values():
        d = math.hypot(p[0] - x, p[1] - y)
        if d < best:
            best = d
            bestp = p
    return best, bestp


def main():
    free = OrderedDict()
    unknown = OrderedDict()
    occ = OrderedDict()
    markers = {}
    robot = None
    status = []
    actions = []

    with rosbag.Bag(BAG_PATH) as bag:
        for topic, msg, t in bag.read_messages():
            ts = t.to_sec()
            if topic == FREE:
                for x, y in pc_xy(msg):
                    k = key(x, y)
                    unknown.pop(k, None)
                    occ.pop(k, None)
                    free[k] = (x, y)
            elif topic == UNKNOWN:
                for x, y in pc_xy(msg):
                    k = key(x, y)
                    free.pop(k, None)
                    occ.pop(k, None)
                    unknown[k] = (x, y)
            elif topic == OCC:
                for x, y in pc_xy(msg):
                    k = key(x, y)
                    free.pop(k, None)
                    unknown.pop(k, None)
                    occ[k] = (x, y)
            elif topic == FRONTIER:
                mkey = (msg.ns, msg.id)
                pts = marker_pts(msg, allow_pose=False)
                if msg.action in (2, 3) or not pts:
                    markers.pop(mkey, None)
                else:
                    markers[mkey] = (ts, msg.ns, msg.id, pts)
            elif topic == ROBOT:
                pts = marker_pts(msg, allow_pose=True)
                if pts:
                    robot = (ts, pts[0])
            elif topic in (RESULT, STATE, ACTION):
                status.append((ts, topic, int(msg.data)))
                if topic == ACTION:
                    actions.append((ts, int(msg.data)))
                if topic == RESULT:
                    break

    print("status_tail", status[-10:])
    print("actions_tail", actions[-10:])
    print("robot", robot)
    print(
        "counts",
        {
            "free": len(free),
            "unknown": len(unknown),
            "occ_inflate": len(occ),
            "markers": len(markers),
        },
    )

    frontiers = []
    dormant = []
    for (_, _), (ts, ns, mid, pts) in markers.items():
        if ns not in ("frontier", "dormant_frontier"):
            continue
        cx = sum(p[0] for p in pts) / len(pts)
        cy = sum(p[1] for p in pts) / len(pts)
        rec = {"id": mid, "ns": ns, "npts": len(pts), "center": (cx, cy), "ts": ts, "pts": pts}
        (frontiers if ns == "frontier" else dormant).append(rec)
    frontiers.sort(key=lambda r: r["id"])
    dormant.sort(key=lambda r: r["id"])
    print("active_frontiers", len(frontiers), "dormant", len(dormant))

    rx, ry = robot[1] if robot else (None, None)

    def state_at(pt):
        k = key(*pt)
        if k in occ:
            return "occ_inflate"
        if k in unknown:
            return "unknown"
        if k in free:
            return "free"
        return "unobserved"

    def free_neighbors_near(pt, rad=0.35):
        x, y = pt
        out = []
        for p in free.values():
            d = math.hypot(p[0] - x, p[1] - y)
            if d <= rad:
                out.append((d, p))
        out.sort(key=lambda z: z[0])
        return out[:5]

    free_keys = set(free.keys())
    occ_keys = set(occ.keys())
    unknown_keys = set(unknown.keys())
    pass_keys = free_keys - occ_keys - unknown_keys
    map_pts = list(free.values()) + list(unknown.values()) + list(occ.values())
    xs = [p[0] for p in map_pts]
    ys = [p[1] for p in map_pts]
    xmin, xmax = min(xs) - 1.0, max(xs) + 1.0
    ymin, ymax = min(ys) - 1.0, max(ys) + 1.0
    steps = [(0.25 * math.cos(i * math.pi / 6), 0.25 * math.sin(i * math.pi / 6)) for i in range(12)]

    def safe_xy(x, y):
        return key(x, y) in pass_keys

    def astar_approx(start, end, max_expand=200000):
        sx, sy = start
        ex, ey = end

        def idx(x, y):
            return (int(math.floor(x / 0.1)), int(math.floor(y / 0.1)))

        start_idx = idx(sx, sy)
        pq = [(math.hypot(ex - sx, ey - sy), 0.0, sx, sy, start_idx)]
        seen = {start_idx: 0.0}
        expands = 0
        nearest = (math.hypot(ex - sx, ey - sy), sx, sy)
        while pq and expands < max_expand:
            _, g, x, y, ik = heapq.heappop(pq)
            expands += 1
            d = math.hypot(ex - x, ey - y)
            if d < nearest[0]:
                nearest = (d, x, y)
            reach_index = abs(idx(x, y)[0] - idx(ex, ey)[0]) <= 1 and abs(idx(x, y)[1] - idx(ex, ey)[1]) <= 1
            if reach_index or d < 0.25:
                return True, expands, nearest
            for dx, dy in steps:
                nx, ny = x + dx, y + dy
                if nx < xmin or nx > xmax or ny < ymin or ny > ymax:
                    continue
                if math.hypot(nx - sx, ny - sy) > 0.25:
                    if not safe_xy(nx, ny):
                        continue
                    length = math.hypot(dx, dy)
                    ok = True
                    nseg = max(1, int(length / 0.025))
                    for j in range(1, nseg):
                        tx = x + dx * j / nseg
                        ty = y + dy * j / nseg
                        if not safe_xy(tx, ty):
                            ok = False
                            break
                    if not ok:
                        continue
                nik = idx(nx, ny)
                ng = g + 0.25
                if ng < seen.get(nik, 1e18):
                    seen[nik] = ng
                    heapq.heappush(pq, (ng + math.hypot(ex - nx, ey - ny), ng, nx, ny, nik))
        return False, expands, nearest

    print("frontier_detail")
    for rec in frontiers[:40]:
        c = rec["center"]
        d_robot = math.hypot(c[0] - rx, c[1] - ry) if robot else None
        nd_occ = nearest_dist(c, occ)
        nd_free = nearest_dist(c, free)
        nd_unk = nearest_dist(c, unknown)
        nbr = free_neighbors_near(c, 0.35)
        ok, exp, near = astar_approx((rx, ry), c) if robot else (None, None, None)
        print(
            json.dumps(
                {
                    "id": rec["id"],
                    "npts": rec["npts"],
                    "center": tuple(round(v, 3) for v in c),
                    "state_center": state_at(c),
                    "dist_robot": round(d_robot, 3),
                    "nearest_free": round(nd_free[0], 3) if nd_free else None,
                    "nearest_occ_inflate": round(nd_occ[0], 3) if nd_occ else None,
                    "nearest_unknown": round(nd_unk[0], 3) if nd_unk else None,
                    "free_neighbors_35cm": [
                        (round(d, 3), (round(p[0], 2), round(p[1], 2))) for d, p in nbr[:3]
                    ],
                    "astar_approx_ok": ok,
                    "astar_expands": exp,
                    "astar_nearest_to_goal": round(near[0], 3) if near else None,
                },
                ensure_ascii=False,
            )
        )

    print("dormant_detail")
    for rec in dormant[:20]:
        c = rec["center"]
        d_robot = math.hypot(c[0] - rx, c[1] - ry) if robot else None
        nd_occ = nearest_dist(c, occ)
        print(
            json.dumps(
                {
                    "id": rec["id"],
                    "npts": rec["npts"],
                    "center": tuple(round(v, 3) for v in c),
                    "state_center": state_at(c),
                    "dist_robot": round(d_robot, 3),
                    "nearest_occ_inflate": round(nd_occ[0], 3) if nd_occ else None,
                },
                ensure_ascii=False,
            )
        )

    print("start_neighborhood")
    start = (rx, ry)
    print(
        json.dumps(
            {
                "start_state": state_at(start),
                "nearest_free": round(nearest_dist(start, free)[0], 3),
                "nearest_occ_inflate": round(nearest_dist(start, occ)[0], 3),
                "nearest_unknown": round(nearest_dist(start, unknown)[0], 3),
            }
        )
    )
    for radius_name, factor in (("ring_0p25", 1), ("ring_0p50", 2)):
        rows = []
        for i, (dx, dy) in enumerate(steps):
            pt = (rx + factor * dx, ry + factor * dy)
            rows.append(
                {
                    "dir": i,
                    "pt": (round(pt[0], 3), round(pt[1], 3)),
                    "state": state_at(pt),
                    "safe": key(*pt) in pass_keys,
                    "nearest_occ_inflate": round(nearest_dist(pt, occ)[0], 3),
                    "nearest_unknown": round(nearest_dist(pt, unknown)[0], 3),
                }
            )
        print(radius_name, json.dumps(rows, ensure_ascii=False))

    print("passable_components")
    neighbours = [(1, 0), (-1, 0), (0, 1), (0, -1), (1, 1), (1, -1), (-1, 1), (-1, -1)]
    seen = set()
    comps = []
    for k in list(pass_keys):
        if k in seen:
            continue
        queue = [k]
        seen.add(k)
        comp = []
        for cur in queue:
            comp.append(cur)
            for dx, dy in neighbours:
                nb = (cur[0] + dx, cur[1] + dy)
                if nb in pass_keys and nb not in seen:
                    seen.add(nb)
                    queue.append(nb)
        comps.append(comp)
    comps.sort(key=len, reverse=True)
    print(json.dumps({"num_components": len(comps), "top_sizes": [len(c) for c in comps[:10]]}))

    def comp_nearest(pt):
        best = (1e9, None, None, None)
        for ci, comp in enumerate(comps):
            for k in comp:
                p = (k[0] / SCALE, k[1] / SCALE)
                d = math.hypot(p[0] - pt[0], p[1] - pt[1])
                if d < best[0]:
                    best = (d, ci, p, len(comp))
        return {
            "dist": round(best[0], 3),
            "component": best[1],
            "point": (round(best[2][0], 2), round(best[2][1], 2)) if best[2] else None,
            "size": best[3],
        }

    comp_rows = {"start": comp_nearest(start)}
    for rec in frontiers:
        comp_rows[f"frontier_{rec['id']}"] = comp_nearest(rec["center"])
    print(json.dumps(comp_rows, ensure_ascii=False))


if __name__ == "__main__":
    main()
