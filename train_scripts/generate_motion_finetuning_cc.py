import copy
import os

# This scripts is mainly for generating training scripts for fintuning CC and fintuning OC


def existDir(dir):
    if not os.path.exists(dir):
        os.makedirs(dir)

ABLATION = ["gtbbx", "gtcat"]
ablation_string = "_".join(ABLATION)
ablation_command_list = [f"--{i}" for i in ABLATION]
ablation_command = " ".join(ablation_command_list)

BASE_LR = 0.0005
MAX_ITER = 60000
NUM_RANDOM = 5
# Default Flip prob is 0.5
FLIP_PROB = 0.5
# MOTION_WEIGHTS = [[i, j, k] for i in [1, 8, 15] for j in [1, 8, 15] for k in [1, 8, 15]]
MOTION_WEIGHTS = [[1, 8, 8]]
# BEST_RGB_MODEL = "/scratch/hanxiao/proj-motionnet/train_output/done/only_det_0.005_30000/det_cc_rgb_1/model_0019999.pth"
# BEST_DEPTH_MODEL = "/scratch/hanxiao/proj-motionnet/train_output/done/only_det_0.005_30000/det_cc_depth_1/model_0024999.pth"
# BEST_RGBD_MODEL = "/scratch/hanxiao/proj-motionnet/train_output/done/only_det_0.005_30000/det_cc_rgbd_4/model_0019999.pth"
BEST_RGB_MODEL = "/scratch/hanxiao/proj-motionnet/train_output/done/motion_finetuning_cc_0.001_60000_0.5/finetune_cc_rgb_4___1_8_8/model_final.pth"
BEST_DEPTH_MODEL = "/scratch/hanxiao/proj-motionnet/train_output/done/motion_finetuning_cc_0.001_60000_0.5/finetune_cc_depth_1___1_8_8/model_final.pth"
BEST_RGBD_MODEL = "/scratch/hanxiao/proj-motionnet/train_output/done/motion_finetuning_cc_0.001_60000_0.5/finetune_cc_rgbd_4___1_8_8/model_final.pth"
EMAIL_ADDR = "shawn_jiang@sfu.ca"
OUTPUT_DIR = "/home/hanxiao/scratch/proj-motionnet/train_output"
name = f"motion_finetuning_cc_{BASE_LR}_{MAX_ITER}_{FLIP_PROB}_{ablation_string}"

# Generate the motion_finetuning script
base_script = []
base_script.append("#!/bin/bash")
base_script.append(
    ". /home/hanxiao/projects/rrg-msavva/hanxiao/proj-motionnet/2DMotion/train_scripts/configure.sh"
)
base_script.append("")
base_script.append(f"mkdir $OUTPUT_DIR/{name}")
base_script.append("")
# settings = ['finetune_cc_rgb', 'finetune_cc_depth', 'finetune_cc_rgbd', 'finetune_oc_rgb', 'finetune_oc_depth', 'finetune_oc_rgbd', 'finetune_pm_rgb', 'finetune_pm_depth', 'finetune_pm_rgbd']
settings = ['finetune_cc_rgb', 'finetune_cc_depth', 'finetune_cc_rgbd']
# settings = ["finetune_cc_rgb"]

for setting in settings:
    for motion_weight in MOTION_WEIGHTS:
        for i in range(1, NUM_RANDOM + 1, 1):
            run_name = f"{setting}_{i}___{motion_weight[0]}_{motion_weight[1]}_{motion_weight[2]}"
            base_script.append(f"mkdir $OUTPUT_DIR/{name}/{run_name}")
            base_script.append(
                f"sbatch --exclude=cdr26,cdr27,cdr28,cdr29,cdr30,cdr31,cdr32,cdr33,cdr34,cdr35,cdr40,cdr104,cdr111,cdr905,cdr922,cdr199 $PROJ_DIR/train_scripts/{name}/train_{run_name}.sh"
            )
        base_script.append("")

OUTPUT_PATH = "/localhome/hja40/Desktop/Research/proj-motionnet/2DMotion/train_scripts"
existDir(f"{OUTPUT_PATH}/{name}")
file = open(f"{OUTPUT_PATH}/{name}.sh", "w")
file.write("\n".join(base_script))
file.close()

# Generate the detailed scripts for motion_finetuning
OUTPUT_PATH = (
    f"/localhome/hja40/Desktop/Research/proj-motionnet/2DMotion/train_scripts/{name}"
)

for setting in settings:
    model_name = setting.split("_")[1]
    input_name = setting.split("_")[2]

    extra = ""
    # Assign the config based on the model type
    if model_name == "cc":
        config = "$PROJ_DIR/configs/bmcc.yaml"
    elif model_name == "oc":
        config = "$PROJ_DIR/configs/bmoc.yaml"
    elif model_name == "pm":
        config = "$PROJ_DIR/configs/pm.yaml"
        # Specify the pose iteration and motion iteration
        extra = f"MODEL.POSE_ITER 0 MODEL.MOTION_ITER {int(MAX_ITER*0.3)}"
    # Assign the input format
    if input_name == "rgb":
        input_format = "RGB"
        model_path = BEST_RGB_MODEL
    elif input_name == "depth":
        input_format = "depth"
        model_path = BEST_DEPTH_MODEL
    elif input_name == "rgbd":
        input_format = "RGBD"
        model_path = BEST_RGBD_MODEL

    other_option = f"--opts MODEL.WEIGHTS {model_path} SOLVER.BASE_LR {BASE_LR} SOLVER.MAX_ITER {MAX_ITER} SOLVER.STEPS '({int(MAX_ITER*0.6)}, {int(MAX_ITER*0.8)})' SOLVER.CHECKPOINT_PERIOD 5000"

    base_script = []
    base_script.append("#!/bin/bash")
    # base_script.append("#SBATCH --nodes=1")
    base_script.append("#SBATCH --account=rrg-msavva")
    base_script.append("#SBATCH --gres=gpu:p100:1         # Number of GPUs (per node)")
    base_script.append("#SBATCH --mem=32000               # memory (per node)")
    base_script.append("#SBATCH --time=1-23:00            # time (DD-HH:MM)")
    base_script.append("#SBATCH --cpus-per-task=6         # Number of CPUs (per task)")
    base_script.append("#SBATCH --mail-type=FAIL")
    base_script.append(f"#SBATCH --mail-user={EMAIL_ADDR}")

    # Create five scripts for different random seed

    for motion_weight in MOTION_WEIGHTS:
        for i in range(1, NUM_RANDOM + 1, 1):
            run_name = f"{setting}_{i}___{motion_weight[0]}_{motion_weight[1]}_{motion_weight[2]}"
            file_name = f"train_{run_name}"

            script = copy.deepcopy(base_script)
            script.append(f"#SBATCH --output={OUTPUT_DIR}/{name}/{run_name}/%x_%j.out")
            script.append(f"#SBATCH --job-name={run_name}")

            script.append("echo 'Start'")

            # script.append("")
            # script.append(
            #     "for mem in $(srun nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits); do"
            # )
            # script.append(' echo "GPU mem used: $mem"')
            # script.append(" if [[ $mem != 0 ]]; then")
            # script.append(
            #     '  echo "ERROR: GPU memory not empty. Resubmitting job in 1 min..."'
            # )
            # script.append("  sleep 60")
            # script.append("  sbatch $0")
            # script.append("  exit 1")
            # script.append(" fi")
            # script.append("done")

            # script.append("")
            # script.append(
            #     "for err in $(srun nvidia-smi --query-gpu=ecc.errors.uncorrected.volatile.device_memory --format=csv,noheader,nounits); do"
            # )
            # script.append(' echo "Ecc uncorr Error: $err"')
            # script.append(" if [[ $err != 0 ]]; then")
            # script.append(
            #     '  echo "ERROR: GPU error not zero. Resubmitting job in 1 min..."'
            # )
            # script.append("  sleep 60")
            # script.append("  sbatch $0")
            # script.append("  exit 1")
            # script.append(" fi")
            # script.append("done")

            script.append("")
            script.append("echo 'ENV Start'")
            script.append("")
            script.append("module load StdEnv/2020  intel/2020.1.217")
            script.append("module load python/3.7")
            script.append("module load cuda/11.0")
            script.append("module load cudnn/8.0.3")
            script.append("")
            script.append(
                ". /home/hanxiao/projects/rrg-msavva/hanxiao/proj-motionnet/2DMotion/train_scripts/configure.sh"
            )
            script.append("source $MNET_VENV/bin/activate")
            script.append("")
            script.append("echo 'Job Start'")
            script.append(
                f"python $PROJ_DIR/train.py --config-file {config} {ablation_command} --output-dir $OUTPUT_DIR/{name}/{run_name} --data-path $DATASET_DIR/MotionDataset_h5_6.11 --input-format {input_format} --model_attr_path $PROJ_DIR/scripts/data/data_statistics/urdf-attr.json --flip_prob {FLIP_PROB} --motion_weights {motion_weight[0]} {motion_weight[1]} {motion_weight[2]} {other_option} INPUT.RNG_SEED {i} {extra}"
            )

            file = open(f"{OUTPUT_PATH}/{file_name}.sh", "w")
            file.write("\n".join(script))
            file.close()
