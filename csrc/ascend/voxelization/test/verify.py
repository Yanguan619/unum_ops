import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
DATA = os.path.join(HERE, "data")


def main():
    m = int(open(os.path.join(DATA, "output/num_voxels.txt")).read())
    gold_m = int(open(os.path.join(DATA, "ref/num_voxels.txt")).read())

    gold_vox = np.fromfile(os.path.join(DATA, "ref/voxels.bin"), dtype=np.float32).reshape(gold_m, 32, 4)
    gold_coord = np.fromfile(os.path.join(DATA, "ref/coords.bin"), dtype=np.int32).reshape(gold_m, 3)
    gold_npts = np.fromfile(os.path.join(DATA, "ref/num_points.bin"), dtype=np.int32).reshape(gold_m)

    vox = np.fromfile(os.path.join(DATA, "output/voxels.bin"), dtype=np.float32).reshape(m, 32, 4)
    coord = np.fromfile(os.path.join(DATA, "output/coords.bin"), dtype=np.int32).reshape(m, 3)
    npts = np.fromfile(os.path.join(DATA, "output/num_points.bin"), dtype=np.int32).reshape(m)

    print("golden M=%d  kernel M=%d" % (gold_m, m))
    ok = True
    if m != gold_m:
        ok = False
        print("MISMATCH: M %d != %d" % (m, gold_m))
    if ok and not np.array_equal(coord, gold_coord):
        d = np.argmax(np.any(coord != gold_coord, axis=1))
        print("MISMATCH: coords differ at %d: %s vs %s" % (d, coord[d], gold_coord[d]))
        ok = False
    if ok and not np.array_equal(npts, gold_npts):
        print("MISMATCH: num_points differ")
        ok = False
    if ok:
        diff = 0
        for v in range(m):
            g = gold_vox[v][: int(gold_npts[v])]
            k = vox[v][: int(npts[v])]
            if int(npts[v]) != int(gold_npts[v]):
                diff += 1
                continue
            gs = np.sort(g, axis=0)
            ks = np.sort(k, axis=0)
            if not np.allclose(gs, ks, atol=0.0):
                diff += 1
        print("voxel point-set mismatch (row-level) count = %d / %d" % (diff, m))
        ok = ok and (diff == 0)
    print("RESULT: %s" % ("PASS" if ok else "FAIL"))
    sys.exit(0 if ok else 1)


if __name__ == "__main__":
    main()
