import os 

models = ["cc", "ocv0"]
inputs = ["RGB", "depth", "RGBD"]
buckets = ["80-100", "60-80", "40-60", "20-40", "0-20"]
scripts_path = "/local-scratch/localhome/hja40/Desktop/Research/proj-motionnet/2DMotion/scripts/analysis/eval_bucket.sh"

def existDir(dir):
    if not os.path.exists(dir):
        os.makedirs(dir)

if __name__ == "__main__":
    script = []
    for model in models:
        for current_input in inputs:
            for bucket in buckets:
                if model == "cc":
                    config = "bmcc"
                elif model == "ocv0":
                    config = "bmoc_v0"
                existDir(f"/local-scratch/localhome/hja40/Desktop/Research/proj-motionnet/2DMotion/scripts/analysis/image_bucket_results/{model}_{current_input.lower()}/{bucket}")
                script.append(f"python evaluate_on_log.py --config-file configs/{config}.yaml --data-path /localhome/hja40/Desktop/Research/proj-motionnet/Dataset/MotionDataset_h5_6.11 --model_attr_path scripts/data/data_statistics/urdf-attr.json --output-dir /local-scratch/localhome/hja40/Desktop/Research/proj-motionnet/2DMotion/scripts/analysis/image_bucket_results/{model}_{current_input.lower()}/{bucket} --input-format {current_input} --filter-file /local-scratch/localhome/hja40/Desktop/Research/proj-motionnet/2DMotion/scripts/analysis/image_bucket_results/{model}_{current_input.lower()}/eval_image_{model}_{current_input.lower()}_{bucket}.json --opts MODEL.WEIGHTS /local-scratch/localhome/hja40/Desktop/Research/proj-motionnet/2DMotion/experiments/finetuning/{model}_{current_input.lower()}/model_best.pth > /local-scratch/localhome/hja40/Desktop/Research/proj-motionnet/2DMotion/scripts/analysis/image_bucket_results/{model}_{current_input.lower()}/{bucket}/result_bucket.log")
            script.append("\n")

    file = open(scripts_path, "w")
    file.write("\n".join(script))
    file.close()