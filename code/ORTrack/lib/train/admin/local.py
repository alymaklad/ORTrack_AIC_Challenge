import os
from pathlib import Path


class EnvironmentSettings:
    def __init__(self):
        workspace = Path(os.environ.get("ORTRACK_WORKSPACE", Path.cwd())).resolve()
        data_root = Path(os.environ.get("AIC_DATA_ROOT", workspace / "data")).resolve()

        self.workspace_dir = str(workspace)
        self.tensorboard_dir = str(workspace / "tensorboard")
        self.pretrained_networks = str(workspace / "pretrained_networks")
        self.lasot_dir = str(data_root / "lasot")
        self.got10k_dir = str(data_root / "got10k" / "train")
        self.got10k_val_dir = str(data_root / "got10k" / "val")
        self.lasot_lmdb_dir = str(data_root / "lasot_lmdb")
        self.got10k_lmdb_dir = str(data_root / "got10k_lmdb")
        self.trackingnet_dir = str(data_root / "trackingnet")
        self.trackingnet_lmdb_dir = str(data_root / "trackingnet_lmdb")
        self.coco_dir = str(data_root / "coco")
        self.coco_lmdb_dir = str(data_root / "coco_lmdb")
        self.lvis_dir = ""
        self.sbd_dir = ""
        self.imagenet_dir = str(data_root / "vid")
        self.imagenet_lmdb_dir = str(data_root / "vid_lmdb")
        self.imagenetdet_dir = ""
        self.ecssd_dir = ""
        self.hkuis_dir = ""
        self.msra10k_dir = ""
        self.davis_dir = ""
        self.youtubevos_dir = ""
