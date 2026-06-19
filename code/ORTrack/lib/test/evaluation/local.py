from lib.test.evaluation.environment import EnvSettings


def local_env_settings():
    settings = EnvSettings()

    settings.prj_dir = r"C:\AIC\ORTrack"
    settings.save_dir = r"C:\AIC\ORTrack\output"
    settings.results_path = r"C:\AIC\ORTrack\output\test\tracking_results"
    settings.segmentation_path = r"C:\AIC\ORTrack\output\test\segmentation_results"
    settings.network_path = r"C:\AIC\ORTrack\output\test\networks"
    settings.result_plot_path = r"C:\AIC\ORTrack\output\test\result_plots"
    settings.otb_path = ""
    settings.nfs_path = ""
    settings.uav_path = r"C:\AIC\Data\uav"
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
    settings.dtb70_path = r"C:\AIC\Data\dtb70"
    settings.uavdt_path = r"C:\AIC\Data\uavdt"
    settings.visdrone2018_path = r"C:\AIC\Data\visdrone2018"
    settings.uav123_path = r"C:\AIC\Data\uav123"
    settings.biodrone_path = r"C:\AIC\Data\biodrone"

    return settings
