import os
from pathlib import Path

from lib.test.evaluation.environment import EnvSettings


def local_env_settings():
    settings = EnvSettings()

    workspace = Path(os.environ.get("ORTRACK_WORKSPACE", Path.cwd())).resolve()
    data_root = Path(os.environ.get("AIC_DATA_ROOT", workspace / "data")).resolve()
    output_root = workspace / "output"

    settings.prj_dir = str(workspace)
    settings.save_dir = str(output_root)
    settings.results_path = str(output_root / "test" / "tracking_results")
    settings.segmentation_path = str(output_root / "test" / "segmentation_results")
    settings.network_path = str(output_root / "test" / "networks")
    settings.result_plot_path = str(output_root / "test" / "result_plots")
    settings.otb_path = ""
    settings.nfs_path = ""
    settings.uav_path = str(data_root / "uav")
    settings.tpl_path = ""
    settings.vot_path = ""
    settings.got10k_path = ""
    settings.lasot_path = ""
    settings.trackingnet_path = ""
    settings.davis_dir = ""
    settings.youtubevos_dir = ""
    settings.got_packed_results_path = ""
    settings.got_reports_path = ""
    settings.tn_packed_results_path = ""
    settings.dtb70_path = str(data_root / "dtb70")
    settings.uavdt_path = str(data_root / "uavdt")
    settings.visdrone2018_path = str(data_root / "visdrone2018")
    settings.uav123_path = str(data_root / "uav123")
    settings.biodrone_path = str(data_root / "biodrone")

    return settings
