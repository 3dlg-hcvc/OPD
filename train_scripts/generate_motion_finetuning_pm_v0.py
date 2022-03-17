import copy
import os

# This code is to generate train scripts for fintuning PM

def existDir(dir):
    if not os.path.exists(dir):
        os.makedirs(dir)

BASE_LR = 0.001
MAX_ITER = 60000
# SOLVER_1 = int(MAX_ITER*0.6)
# SOLVER_2 = int(MAX_ITER*0.8)
SOLVER_1 = 20000
SOLVER_2 = 40000
NUM_RANDOM = 2
BEST_RGB_MODEL = "/scratch/hanxiao/proj-motionnet/train_output/done/only_det_0.005_30000/det_cc_rgb_1/model_0019999.pth"
BEST_DEPTH_MODEL = "/scratch/hanxiao/proj-motionnet/train_output/done/only_det_0.005_30000/det_cc_depth_1/model_0024999.pth"
BEST_RGBD_MODEL = "/scratch/hanxiao/proj-motionnet/train_output/done/only_det_0.005_30000/det_cc_rgbd_4/model_0019999.pth"
EMAIL_ADDR = "shawn_jiang@sfu.ca"
OUTPUT_DIR = "/home/hanxiao/scratch/proj-motionnet/train_output"
POSE_ITER = 0
MOTION_ITER = 20000
# POSE_RT_WEIGHTS = [i for i in [1, 8, 15]]
POSE_RT_WEIGHTS = [[1, 15]]
motion_weight = [1, 8, 8]
# settings = ['finetune_pmv0_rgb', 'finetune_pmv0_depth', 'finetune_pmv0_rgbd']
settings = ['finetune_pmv0_rgb']
name = f'motion_finetuning_pm_v0_normalize_{BASE_LR}_{MAX_ITER}_{SOLVER_1}_{SOLVER_2}_{POSE_ITER}_{MOTION_ITER}_{motion_weight[0]}_{motion_weight[1]}_{motion_weight[2]}'

# Generate the motion_finetuning script
base_script = []
base_script.append("#!/bin/bash")
base_script.append(". /home/hanxiao/projects/rrg-msavva/hanxiao/proj-motionnet/2DMotion/train_scripts/configure.sh")
base_script.append("")
base_script.append(f"mkdir $OUTPUT_DIR/{name}")
base_script.append("")

for setting in settings:
    for pose_weight in POSE_RT_WEIGHTS:
        for i in range(1, NUM_RANDOM+1, 1):
            run_name = f"{setting}_{i}___{pose_weight[0]}_{pose_weight[1]}"
            base_script.append(f"mkdir $OUTPUT_DIR/{name}/{run_name}")
            base_script.append(f"sbatch --exclude=cdr26,cdr27,cdr28,cdr29,cdr30,cdr31,cdr32,cdr33,cdr34,cdr35,cdr40,cdr104,cdr111,cdr905,cdr922,cdr199 $PROJ_DIR/train_scripts/{name}/train_{run_name}.sh")
        base_script.append("")

OUTPUT_PATH = "/localhome/hja40/Desktop/Research/proj-motionnet/2DMotion/train_scripts"
existDir(f"{OUTPUT_PATH}/{name}")
file = open(f"{OUTPUT_PATH}/{name}.sh", "w")
file.write("\n".join(base_script))
file.close()

# Generate the detailed scripts for motion_finetuning
OUTPUT_PATH = f"/localhome/hja40/Desktop/Research/proj-motionnet/2DMotion/train_scripts/{name}"

for setting in settings:
    model_name = setting.split('_')[1]
    input_name = setting.split('_')[2]

    extra = ""
    # Assign the config based on the model type
    if model_name == 'cc':
        config = "$PROJ_DIR/configs/bmcc.yaml"
    elif model_name == 'oc':
        config = "$PROJ_DIR/configs/bmoc.yaml"
    elif model_name == 'pm':
        config = "$PROJ_DIR/configs/pm.yaml"
        # Specify the pose iteration and motion iteration
        extra = f"MODEL.POSE_ITER {POSE_ITER} MODEL.MOTION_ITER {MOTION_ITER}"
    elif model_name == "pmv0":
        config = "$PROJ_DIR/configs/pm_v0.yaml"
        extra = f"MODEL.POSE_ITER {POSE_ITER} MODEL.MOTION_ITER {MOTION_ITER}"
    # Assign the input format
    if input_name == 'rgb':
        input_format = "RGB"
        model_path = BEST_RGB_MODEL
    elif input_name == 'depth':
        input_format = "depth"
        model_path = BEST_DEPTH_MODEL
    elif input_name == 'rgbd':
        input_format = "RGBD"
        model_path = BEST_RGBD_MODEL

    other_option = f"--opts MODEL.WEIGHTS {model_path} SOLVER.BASE_LR {BASE_LR} SOLVER.MAX_ITER {MAX_ITER} SOLVER.STEPS '({SOLVER_1}, {SOLVER_2})' SOLVER.CHECKPOINT_PERIOD 5000"

    base_script = []
    base_script.append("#!/bin/bash")
    # base_script.append("#SBATCH --nodes=1")
    base_script.append("#SBATCH --account=rrg-msavva")
    base_script.append("#SBATCH --gres=gpu:p100:1         # Number of GPUs (per node)")
    base_script.append("#SBATCH --mem=64000               # memory (per node)")
    base_script.append("#SBATCH --time=2-23:00            # time (DD-HH:MM)")
    base_script.append("#SBATCH --cpus-per-task=6         # Number of CPUs (per task)")
    base_script.append("#SBATCH --mail-type=FAIL")
    base_script.append(f"#SBATCH --mail-user={EMAIL_ADDR}")

    # Create five scripts for different random seed
    for pose_weight in POSE_RT_WEIGHTS:
        for i in range(1, NUM_RANDOM+1, 1):
            run_name = f"{setting}_{i}___{pose_weight[0]}_{pose_weight[1]}"
            file_name = f"train_{run_name}"

            script = copy.deepcopy(base_script)
            script.append(f"#SBATCH --output={OUTPUT_DIR}/{name}/{run_name}/%x_%j.out")
            script.append(f"#SBATCH --job-name={run_name}")

            # script.append("")
            # script.append("for mem in $(srun nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits); do")
            # script.append(' echo "GPU mem used: $mem"')
            # script.append(" if [[ $mem != 0 ]]; then")
            # script.append('  echo "ERROR: GPU memory not empty. Resubmitting job in 1 min..."')
            # script.append("  sleep 60")
            # script.append("  sbatch $0")
            # script.append("  exit 1")
            # script.append(" fi")
            # script.append("done")

            # script.append("")
            # script.append("for err in $(srun nvidia-smi --query-gpu=ecc.errors.uncorrected.volatile.device_memory --format=csv,noheader,nounits); do")
            # script.append(' echo "Ecc uncorr Error: $err"')
            # script.append(" if [[ $err != 0 ]]; then")
            # script.append('  echo "ERROR: GPU error not zero. Resubmitting job in 1 min..."')
            # script.append("  sleep 60")
            # script.append("  sbatch $0")
            # script.append("  exit 1")
            # script.append(" fi")
            # script.append("done")

            script.append("")
            script.append("module load StdEnv/2020  intel/2020.1.217")
            script.append("module load python/3.7")
            script.append("module load cuda/11.0")
            script.append("module load cudnn/8.0.3")
            script.append("")
            script.append(". /home/hanxiao/projects/rrg-msavva/hanxiao/proj-motionnet/2DMotion/train_scripts/configure.sh")
            script.append("source $MNET_VENV/bin/activate")

            # Watch the GPU memory
            # script.append(f"watch -n 30 'nvidia-smi|tee -a $OUTPUT_DIR/{name}/{run_name}/gpu_log.txt' &")
            # script.append("MYKILL=$!")
            # script.append(f"python $PROJ_DIR/train.py --config-file {config} --output-dir $OUTPUT_DIR/{name}/{run_name} --data-path $DATASET_DIR/MotionDataset_h5_6.11 --input-format {input_format} --model_attr_path $PROJ_DIR/scripts/data/data_statistics/urdf-attr.json {other_option} INPUT.RNG_SEED {i} {extra} &")
            # script.append("MYWAIT=$!")
            # script.append("wait $MYWAIT")
            # script.append("kill $MYKILL")

            script.append(f"python $PROJ_DIR/train.py --config-file {config} --output-dir $OUTPUT_DIR/{name}/{run_name} --data-path $DATASET_DIR/MotionDataset_h5_6.11 --input-format {input_format} --model_attr_path $PROJ_DIR/scripts/data/data_statistics/urdf-attr.json --motion_weights {motion_weight[0]} {motion_weight[1]} {motion_weight[2]} --pose_rt_weight {pose_weight[0]} {pose_weight[1]} {other_option} INPUT.RNG_SEED {i} {extra}")

            file = open(f"{OUTPUT_PATH}/{file_name}.sh", "w")
            file.write("\n".join(script))
            file.close()