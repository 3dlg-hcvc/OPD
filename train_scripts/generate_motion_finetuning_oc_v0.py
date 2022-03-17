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

BASE_LR = 0.0001
MAX_ITER = 60000
SOLVER_1 = int(MAX_ITER * 0.6)
SOLVER_2 = int(MAX_ITER * 0.8)
# SOLVER_1 = 45000
# SOLVER_2 = 50000
# Default EXTRINSIC WEIGHT should be 1
EXTRINSIC_WEIGHT = 15
# Default DET EXTRINIC ITER should be 0
DET_EXTRINSIC_ITER = 0
# Default Flip prob is 0.5
FLIP_PROB = 0.5
# MOTION_WEIGHTS = [[i, j, k] for i in [1, 8, 15] for j in [1, 8, 15] for k in [1, 8, 15]]
MOTION_WEIGHTS = [[1, 8, 8]]
NUM_RANDOM = 5
# BEST_RGB_MODEL = "/scratch/hanxiao/proj-motionnet/train_output/done/only_det_0.005_30000/det_cc_rgb_1/model_0019999.pth"
# BEST_DEPTH_MODEL = "/scratch/hanxiao/proj-motionnet/train_output/done/only_det_0.005_30000/det_cc_depth_1/model_0024999.pth"
# BEST_RGBD_MODEL = "/scratch/hanxiao/proj-motionnet/train_output/done/only_det_0.005_30000/det_cc_rgbd_4/model_0019999.pth"
BEST_RGB_MODEL = "/scratch/hanxiao/proj-motionnet/train_output/done/motion_finetuning_ocv0_ew_dei_0.001_60000_36000_48000_0.5_15_0/finetune_ocv0_rgb_5___1_8_8/model_final.pth"
BEST_DEPTH_MODEL = "/scratch/hanxiao/proj-motionnet/train_output/done/motion_finetuning_ocv0_ew_dei_0.001_60000_36000_48000_0.5_15_0/finetune_ocv0_depth_4___1_8_8/model_final.pth"
BEST_RGBD_MODEL = "/scratch/hanxiao/proj-motionnet/train_output/done/motion_finetuning_ocv0_ew_dei_0.001_60000_36000_48000_0.5_15_0/finetune_ocv0_rgbd_2___1_8_8/model_final.pth"
EMAIL_ADDR = "shawn_jiang@sfu.ca"
OUTPUT_DIR = "/home/hanxiao/scratch/proj-motionnet/train_output"
name = f"motion_finetuning_ocv0_ew_dei_{BASE_LR}_{MAX_ITER}_{SOLVER_1}_{SOLVER_2}_{FLIP_PROB}_{EXTRINSIC_WEIGHT}_{DET_EXTRINSIC_ITER}_{ablation_string}"
settings = ['finetune_ocv0_rgb', 'finetune_ocv0_depth', 'finetune_ocv0_rgbd']
# settings = ["finetune_ocv0_rgb"]

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
    extra_v0 = ""
    # Assign the config based on the model type
    if model_name == "cc":
        config = "$PROJ_DIR/configs/bmcc.yaml"
    elif model_name == "oc":
        config = "$PROJ_DIR/configs/bmoc.yaml"
    elif model_name == "pm":
        config = "$PROJ_DIR/configs/pm.yaml"
        # Specify the pose iteration and motion iteration
        extra = f"MODEL.POSE_ITER 0 MODEL.MOTION_ITER {int(MAX_ITER*0.3)}"
    elif model_name == "ocv0":
        config = "$PROJ_DIR/configs/bmoc_v0.yaml"
        extra_v0 = f"--extrinsic_weight {EXTRINSIC_WEIGHT} --det_extrinsic_iter {DET_EXTRINSIC_ITER}"
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

    other_option = f"--opts MODEL.WEIGHTS {model_path} SOLVER.BASE_LR {BASE_LR} SOLVER.MAX_ITER {MAX_ITER} SOLVER.STEPS '({SOLVER_1}, {SOLVER_2})' SOLVER.CHECKPOINT_PERIOD 5000"

    base_script = []
    base_script.append("#!/bin/bash")
    # base_script.append("#SBATCH --nodes=1")
    base_script.append("#SBATCH --account=rrg-msavva")
    base_script.append("#SBATCH --gres=gpu:p100:1         # Number of GPUs (per node)")
    base_script.append("#SBATCH --mem=64000               # memory (per node)")
    base_script.append("#SBATCH --time=2-15:00            # time (DD-HH:MM)")
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

            # # Watch the GPU memory
            # script.append(
            #     f"watch -n 30 'nvidia-smi|tee -a $OUTPUT_DIR/{name}/{run_name}/gpu_log.txt' &"
            # )
            # script.append("MYKILL=$!")
            # script.append(
            #     f"python $PROJ_DIR/train.py --config-file {config} --output-dir $OUTPUT_DIR/{name}/{run_name} --data-path $DATASET_DIR/MotionDataset_h5_6.11 --input-format {input_format} --model_attr_path $PROJ_DIR/scripts/data/data_statistics/urdf-attr.json --flip_prob {FLIP_PROB} --motion_weights {motion_weight[0]} {motion_weight[1]} {motion_weight[2]} {extra_v0} {other_option} INPUT.RNG_SEED {i} {extra} &"
            # )
            # script.append("MYWAIT=$!")
            # script.append("wait $MYWAIT")
            # script.append("kill $MYKILL")
            script.append(f"python $PROJ_DIR/train.py --config-file {config} {ablation_command} --output-dir $OUTPUT_DIR/{name}/{run_name} --data-path $DATASET_DIR/MotionDataset_h5_6.11 --input-format {input_format} --model_attr_path $PROJ_DIR/scripts/data/data_statistics/urdf-attr.json --flip_prob {FLIP_PROB} --motion_weights {motion_weight[0]} {motion_weight[1]} {motion_weight[2]} {extra_v0} {other_option} INPUT.RNG_SEED {i} {extra}")

            file = open(f"{OUTPUT_PATH}/{file_name}.sh", "w")
            file.write("\n".join(script))
            file.close()
