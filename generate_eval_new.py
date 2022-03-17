import os
import glob

base_path = "/local-scratch/localhome/hja40/Desktop/Research/proj-motionnet/2DMotion/full_experiments"
script_name = "/local-scratch/localhome/hja40/Desktop/Research/proj-motionnet/2DMotion/run_test.sh"
data_path = "/localhome/hja40/Desktop/Research/proj-motionnet/Dataset/MotionDataset_h5_ancsh"
model_attr_path = '/local-scratch/localhome/hja40/Desktop/Research/proj-motionnet/2DMotion/scripts/data/data_statistics/urdf-attr.json'

double_level = True
inference_file = True
test = True

extra = ""

if __name__ == "__main__":
    base_script = []

    set_name = os.listdir(base_path)
    if double_level == True:
        for name in set_name:
            if name == "done":
                continue
            experiments = os.listdir(f"{base_path}/{name}")
            for experiment in experiments:
                # model_name = glob.glob(f"{base_path}/{name}/{experiment}/*.pth")[0].split('/')[-1]
    
                # base_script.append(f"mv {base_path}/{name}/{experiment}/{model_name} {base_path}/{name}/{experiment}/model_best.pth")
                if "cc_" in experiment:
                    config_file = "configs/bmcc.yaml"
                elif "ocv0_" in experiment:
                    config_file = "configs/bmoc_v0.yaml"
                elif "oc_" in experiment:
                    config_file = "configs/bmoc.yaml"
                elif "ocv0s_" in experiment:
                    config_file = "configs/bmoc_v0s.yaml"
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

                extra_set = ""
                if inference_file:
                    if test:
                        extra_set = "--opts DATASETS.TEST \"('MotionNet_test',)\""
                    base_script.append(f"conda run -n 2dmotion python evaluate_on_log.py --config-file {config_file} --data-path {data_path} --model_attr_path {model_attr_path} --output-dir {base_path}/{name}/{experiment} --input-format {input_format} --inference-file {base_path}/{name}/{experiment}/inference/instances_predictions.pth {extra} {extra_set} > {base_path}/{name}/{experiment}/eval.log")
                else:
                    if test:
                        extra_set = "DATASETS.TEST \"('MotionNet_test',)\""
                    base_script.append(f"conda run -n 2dmotion python evaluate_on_log.py --config-file {config_file} --data-path {data_path} --model_attr_path {model_attr_path} --output-dir {base_path}/{name}/{experiment} --input-format {input_format} {extra} --opts MODEL.WEIGHTS {base_path}/{name}/{experiment}/model_best.pth {extra_set} > {base_path}/{name}/{experiment}/eval.log")

    else:
        for experiment in set_name:
            if experiment == "done":
                continue
            if "cc_" in experiment:
                config_file = "configs/bmcc.yaml"
            elif "ocv0_" in experiment:
                config_file = "configs/bmoc_v0.yaml"
            elif "oc_" in experiment:
                config_file = "configs/bmoc.yaml"
            elif "ocv0s_" in experiment:
                config_file = "configs/bmoc_v0s.yaml"
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
            
            extra_set = ""
            if inference_file:
                if test:
                    extra_set = "--opts DATASETS.TEST \"('MotionNet_test',)\""
                base_script.append(f"conda run -n 2dmotion python evaluate_on_log.py --config-file {config_file} --data-path {data_path} --model_attr_path scripts/data/data_statistics/urdf-attr.json --output-dir {base_path}/{experiment} --input-format {input_format} --inference-file {base_path}/{experiment}/inference/instances_predictions.pth {extra} {extra_set} > {base_path}/{experiment}/eval.log")
            else:
                if test:
                    extra_set = "DATASETS.TEST \"('MotionNet_test',)\""
                base_script.append(f"conda run -n 2dmotion python evaluate_on_log.py --config-file {config_file} --data-path {data_path} --model_attr_path {model_attr_path} --output-dir {base_path}/{experiment} --input-format {input_format} {extra} --opts MODEL.WEIGHTS {base_path}/{experiment}/model_best.pth {extra_set} > {base_path}/{experiment}/eval.log")

    file = open(script_name, "w")
    file.write("\n".join(base_script))
    file.close()

