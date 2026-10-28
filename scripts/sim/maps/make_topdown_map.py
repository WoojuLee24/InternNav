"""Top-down background map (png + json meta) from a reconstructed scene mesh (.obj, binary .ply, or a .usdz
with mesh.ply inside), for web_viz.py.

Floor triangles (|z - floor_z| < 0.1) are light green, obstacles (floor_z + 0.25 .. + 1.6) dark grey.
Usage: python3 scripts/sim/maps/make_topdown_map.py <mesh.obj> scripts/sim/maps/etri_indoor --floor_z -0.13
"""
import argparse
import json
import zipfile

import cv2
import numpy as np


def load_ply(b):
    h = b.index(b"end_header\n") + 11
    head = b[:h].decode().split("\n")
    assert "format binary_little_endian 1.0" in head and "property list uchar int vertex_indices" in head
    nv = int(next(l for l in head if l.startswith("element vertex")).split()[2])
    nf = int(next(l for l in head if l.startswith("element face")).split()[2])
    V = np.frombuffer(b, "<f4", nv * 3, h).reshape(-1, 3).astype(np.float64)
    F = np.frombuffer(b, np.dtype([("n", "u1"), ("i", "<i4", 3)]), nf, h + nv * 12)
    assert (F["n"] == 3).all()
    return V, F["i"]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("obj")
    ap.add_argument("out_prefix")
    ap.add_argument("--floor_z", type=float, default=-0.13, help="map z of the floor (ETRI indoor: -0.13)")
    ap.add_argument("--res", type=float, default=0.05)
    ap.add_argument("--bbox", type=float, nargs=4, metavar=("XMIN", "XMAX", "YMIN", "YMAX"))
    a = ap.parse_args()
    if a.obj.endswith(".usdz"):
        V, F = load_ply(zipfile.ZipFile(a.obj).read("mesh.ply"))
    elif a.obj.endswith(".ply"):
        V, F = load_ply(open(a.obj, "rb").read())
    else:
        V, F = [], []
        for line in open(a.obj):
            if line.startswith("v "):
                V.append([float(x) for x in line.split()[1:4]])
            elif line.startswith("f "):
                F.append([int(x.split("/")[0]) - 1 for x in line.split()[1:4]])
        V, F = np.array(V), np.array(F)
    c = V[F].mean(1)
    x0, x1, y0, y1 = a.bbox or (V[:, 0].min(), V[:, 0].max(), V[:, 1].min(), V[:, 1].max())
    W = int((x1 - x0) / a.res) + 1
    H = int((y1 - y0) / a.res) + 1
    img = np.full((H, W, 3), 255, np.uint8)
    floor = np.abs(c[:, 2] - a.floor_z) < 0.1
    obst = (c[:, 2] > a.floor_z + 0.25) & (c[:, 2] < a.floor_z + 1.6)
    for mask, col in ((floor, (200, 230, 200)), (obst, (60, 60, 60))):
        for tri in V[F[mask]]:
            cv2.fillConvexPoly(img, ((tri[:, :2] - [x0, y0]) / a.res).astype(np.int32), col)
    img = np.ascontiguousarray(img[::-1])  # image row 0 = max y
    cv2.imwrite(a.out_prefix + ".png", img)
    meta = {"x0": float(x0), "y0": float(y0), "res": a.res, "width": W, "height": H,
            "note": "pixel (u, v) = ((x - x0) / res, height - 1 - (y - y0) / res)"}
    json.dump(meta, open(a.out_prefix + ".json", "w"), indent=1)
    print(meta)


if __name__ == "__main__":
    main()
