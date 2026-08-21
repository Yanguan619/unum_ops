import os
import sys

import warnings

warnings.filterwarnings("ignore")

ROOT = "/data/workspace/OpenPCDet"
HERE = os.path.dirname(os.path.abspath(__file__))
OUT = os.path.join(HERE, "data")
FRAME = os.environ.get("VOXEL_FRAME", "000008")


def main():
    sys.path.insert(0, ROOT)
    sys.path.insert(0, os.path.join(ROOT, "tools"))
    sys.path.insert(0, "/data/workspace/unum_ops/src/unum_ops")

    from pcdet.config import cfg, cfg_from_yaml_file
    from pcdet.datasets.kitti.kitti_dataset2 import KittiDataset
    from pcdet.utils import common_utils
    from unum_ops.spconv.utils import VoxelGeneratorV2

    cfg_from_yaml_file(os.path.join(ROOT, "data/config.yaml"), cfg)
    logger = common_utils.create_logger()
    ds = KittiDataset(
        dataset_cfg=cfg.DATA_CONFIG, class_names=cfg.CLASS_NAMES, training=False, logger=logger
    )

    points = np_fromfile(os.path.join(ROOT, "data/kitti/training/velodyne/%s.bin" % FRAME)).reshape(-1, 4)
    if cfg.DATA_CONFIG.FOV_POINTS_ONLY:
        try:
            calib = ds.get_calib(FRAME)
            img_shape = ds.get_image_shape(FRAME)
            fov_flag = ds.get_fov_flag(points, img_shape, calib)
            points = points[fov_flag]
        except Exception:
            pass

    gen = VoxelGeneratorV2(
        voxel_size=[0.16, 0.16, 4.0],
        point_cloud_range=[0, -39.68, -3, 69.12, 39.68, 1],
        max_num_points=32,
        max_voxels=40000,
    )
    out = gen.generate(points)
    gold_vox = out["voxels"]
    gold_coord = out["coordinates"]
    gold_npts = out["num_points_per_voxel"]

    os.makedirs(os.path.join(OUT, "input"), exist_ok=True)
    os.makedirs(os.path.join(OUT, "ref"), exist_ok=True)
    os.makedirs(os.path.join(OUT, "output"), exist_ok=True)

    points.tofile(os.path.join(OUT, "input/points.bin"))
    gold_vox.tofile(os.path.join(OUT, "ref/voxels.bin"))
    gold_coord.tofile(os.path.join(OUT, "ref/coords.bin"))
    gold_npts.tofile(os.path.join(OUT, "ref/num_points.bin"))
    with open(os.path.join(OUT, "ref/num_voxels.txt"), "w") as f:
        f.write(str(gold_coord.shape[0]))

    with open(os.path.join(OUT, "input/num_points.txt"), "w") as f:
        f.write(str(points.shape[0]))
    print("frame=%s points=%d ref_voxels=%d" % (FRAME, points.shape[0], gold_coord.shape[0]))


def np_fromfile(p):
    import numpy as np

    return np.fromfile(p, dtype=np.float32)


if __name__ == "__main__":
    main()
