import copy
import os

def existDir(dir):
    if not os.path.exists(dir):
        os.makedirs(dir)


BASE_LR = 0.0005
MAX_ITER = 30000
NUM_RANDOM = 1
EMAIL_ADDR = "shawn_jiang@sfu.ca"
OUTPUT_DIR = "/home/hanxiao/scratch/proj-motionnet/train_output"
name = f'only_det_ancsh_{BASE_LR}_{MAX_ITER}'

# Generate the only_det script
base_script = []
base_script.append("#!/bin/bash")
base_script.append(". /home/hanxiao/projects/rrg-msavva/hanxiao/proj-motionnet/2DMotion/train_scripts/configure.sh")
base_script.append("")
base_script.append(f"mkdir $OUTPUT_DIR/{name}")
base_script.append("")
settings = ['det_cc_rgb', 'det_cc_rgbd', 'det_cc_depth']

for setting in settings:
    for i in range(1, NUM_RANDOM+1, 1):
        base_script.append(f"mkdir $OUTPUT_DIR/{name}/{setting}_{i}")
        base_script.append(f"sbatch --exclude=cdr26,cdr27,cdr28,cdr29,cdr30,cdr31,cdr32,cdr33,cdr34,cdr35,cdr40,cdr104,cdr111,cdr905,cdr922,cdr199 $PROJ_DIR/train_scripts/{name}/train_{setting}_{i}.sh")
    base_script.append("")

OUTPUT_PATH = "/localhome/hja40/Desktop/Research/proj-motionnet/2DMotion/train_scripts"
existDir(f"{OUTPUT_PATH}/{name}")
file = open(f"{OUTPUT_PATH}/{name}.sh", "w")
file.write("\n".join(base_script))
file.close()

# Generate the detailed scripts for only_det
OUTPUT_PATH = f"/localhome/hja40/Desktop/Research/proj-motionnet/2DMotion/train_scripts/{name}"

for setting in settings:
    model_name = setting.split('_')[1]
    input_name = setting.split('_')[2]
    # Assign the config based on the model type
    if model_name == 'cc':
        config = "$PROJ_DIR/configs/bmcc.yaml"
    elif model_name == 'oc':
        config = "$PROJ_DIR/configs/bmoc.yaml"
    elif model_name == 'pm':
        config = "$PROJ_DIR/configs/pm.yaml"
    # Assign the input format
    if input_name == 'rgb':
        input_format = "RGB"
    elif input_name == 'depth':
        input_format = "depth"
    elif input_name == 'rgbd':
        input_format = "RGBD"
    
    other_option = f"--only_det --opts SOLVER.BASE_LR {BASE_LR} SOLVER.MAX_ITER {MAX_ITER} SOLVER.STEPS '({int(MAX_ITER*0.6)}, {int(MAX_ITER*0.8)})' SOLVER.CHECKPOINT_PERIOD 5000"

    base_script = []
    base_script.append("#!/bin/bash")
    # base_script.append("#SBATCH --nodes=1")
    base_script.append("#SBATCH --account=rrg-msavva")
    base_script.append("#SBATCH --gres=gpu:p100:1         # Number of GPUs (per node)")
    base_script.append("#SBATCH --mem=32000               # memory (per node)")
    base_script.append("#SBATCH --time=0-23:00            # time (DD-HH:MM)")
    base_script.append("#SBATCH --cpus-per-task=6         # Number of CPUs (per task)")
    base_script.append("#SBATCH --mail-type=FAIL")
    base_script.append(f"#SBATCH --mail-user={EMAIL_ADDR}")

    # Create five scripts for different random seed
    for i in range(1, NUM_RANDOM+1, 1):
        file_name = f"train_{setting}_{i}"

        script = copy.deepcopy(base_script)
        script.append(f"#SBATCH --output={OUTPUT_DIR}/{name}/{setting}_{i}/%x_%j.out")
        script.append(f"#SBATCH --job-name={setting}_{i}")

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

        script.append("")
        script.append("module load StdEnv/2020  intel/2020.1.217")
        script.append("module load python/3.7")
        script.append("module load cuda/11.0")
        script.append("module load cudnn/8.0.3")
        script.append("")
        script.append(". /home/hanxiao/projects/rrg-msavva/hanxiao/proj-motionnet/2DMotion/train_scripts/configure.sh")
        script.append("source $MNET_VENV/bin/activate")
        
        ## Watch the GPU memory
        # script.append(f"watch -n 30 'nvidia-smi|tee -a $OUTPUT_DIR/{name}/{setting}_{i}/gpu_log.txt' &")
        # script.append("MYKILL=$!")
        # script.append(f"python $PROJ_DIR/train.py --config-file {config} --output-dir $OUTPUT_DIR/{name}/{setting}_{i} --data-path $DATASET_DIR/MotionDataset_h5_6.11 --input-format {input_format} --model_attr_path $PROJ_DIR/scripts/data/data_statistics/urdf-attr.json {other_option} INPUT.RNG_SEED {i} &")
        # script.append("MYWAIT=$!")
        # script.append("wait $MYWAIT")
        # script.append("kill $MYKILL")

        script.append(f"python $PROJ_DIR/train.py --config-file {config} --output-dir $OUTPUT_DIR/{name}/{setting}_{i} --data-path $DATASET_DIR/MotionDataset_h5_ancsh --input-format {input_format} --model_attr_path $PROJ_DIR/scripts/data/data_statistics/urdf-attr.json {other_option} INPUT.RNG_SEED {i}")

        file = open(f"{OUTPUT_PATH}/{file_name}.sh", "w")
        file.write("\n".join(script))
        file.close()

