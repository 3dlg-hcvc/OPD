import os

base_path = "/local-scratch/localhome/hja40/Desktop/Research/proj-motionnet/2DMotion/full_experiments"
script_name = "/local-scratch/localhome/hja40/Desktop/Research/proj-motionnet/2DMotion/run_test.sh"
data_path = "/localhome/hja40/Desktop/Research/proj-motionnet/Dataset/MotionDataset_h5_6.11"

if __name__ == "__main__":
    base_script = []

    set_name = os.listdir(base_path)
    for name in set_name:
        if name == "done":
            continue
        experiments = os.listdir(f"{base_path}/{name}")
        for experiment in experiments:
            if "cc_" in experiment:
                config_file = "configs/bmcc.yaml"
            elif "ocv0_" in experiment:
                config_file = "configs/bmoc_v0.yaml"
            elif "oc_" in experiment:
                config_file = "configs/bmoc.yaml"
            else:
                import pdb
                pdb.set_trace()

            if "rgbd_" in experiment:
                input_format = "RGBD"
            elif "rgb_" in experiment:
                input_format = "RGB"
            elif "depth_" in experiment:
                input_format = "depth"
            else:
                import pdb
                pdb.set_trace()

            base_script.append(f"conda run -n 2dmotion python evaluate_on_log.py --config-file {config_file} --data-path {data_path} --model_attr_path scripts/data/data_statistics/urdf-attr.json --output-dir {base_path}/{name}/{experiment} --input-format {input_format} --opts MODEL.WEIGHTS {base_path}/{name}/{experiment}/model_final.pth DATASETS.TEST \"('MotionNet_test',)\" > {base_path}/{name}/{experiment}/eval.log")

    file = open(script_name, "w")
    file.write("\n".join(base_script))
    file.close()

