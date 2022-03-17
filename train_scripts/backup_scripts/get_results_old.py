# This script is to extract the useful evaluation results from the log
import os
import glob
import argparse


def get_folder_list(path):
    if not os.path.exists(path):
        raise OSError("Path {} not exist!".format(path))

    dir_list = []
    for dirname in os.listdir(path):
        if os.path.isdir(os.path.join(path, dirname)):
            dir_list.append(os.path.join(path, dirname))
    return dir_list


def extractInfo(dir_path):
    out_files = glob.glob(f"{dir_path}/*.out")
    if not len(out_files) == 1:
        print(f"Error: multiple logs -> {dir_path}")
        return
    out_file = open(out_files[0])

    # Used to judge if the train log is complete
    valid = False
    index = 0

    # Used to locate the final evaluation
    box_line = [-1, -1]
    seg_line = [-1, -1]
    # Used to store the evaluation results
    mAP_pdet = [-1, -1]
    mAP_mtype = [-1, -1, -1, -1]
    mAP_motion = [-1, -1]
    mAP_axis = [-1, -1, -1, -1]
    mAP_axis_t = [-1, -1, -1, -1]
    mAP_axis_r = [-1, -1, -1, -1]
    mAP_orig_r = [-1, -1, -1, -1]
    error_axis = [-1, -1, -1, -1]
    error_axis_t = [-1, -1, -1, -1]
    error_axis_r = [-1, -1, -1, -1]
    error_orig_r = [-1, -1, -1, -1]
    # evaluation results
    for line in out_file:
        index += 1
        if "seconds" in line:
            valid = True
        if "Evaluate annotation type *bbox*" in line:
            box_line[0] = box_line[1]
            box_line[1] = index
        if "Evaluate annotation type *segm*" in line:
            seg_line[0] = seg_line[1]
            seg_line[1] = index
        if (
            "Average Precision (Detection) (mAP_Detection) @[ IoU=0.50      | area=   all | maxDets=100 ]"
            in line
        ):
            mAP_pdet[0] = mAP_pdet[1]
            mAP_pdet[1] = line.split("=")[-1][1:-1]
        if (
            "Average Precision (Motion Type Threshold) (mAP_Type) @[ IoU=0.50      | area=   all | maxDets=100 ]"
            in line
        ):
            for i in range(len(mAP_mtype) - 1):
                mAP_mtype[i] = mAP_mtype[i + 1]
            mAP_mtype[3] = line.split("=")[-1][1:-1]
        if (
            "Average Precision (All motion) Threshold) (mAP_Motion) @[ IoU=0.50      | area=   all | maxDets=100 ]"
            in line
        ):
            mAP_motion[0] = mAP_motion[1]
            mAP_motion[1] = line.split("=")[-1][1:-1]
        if (
            "Average Precision (Motion Axis Direction Threshold) (mAP_ADir) @[ IoU=0.50      | area=   all | maxDets=100 ]"
            in line
        ):
            for i in range(len(mAP_axis) - 1):
                mAP_axis[i] = mAP_axis[i + 1]
            mAP_axis[3] = line.split("=")[-1][1:-1]
        if (
            "Average Precision (Motion Axis Direction Threshold translation) (mAP_ADir@t) @[ IoU=0.50      | area=   all | maxDets=100 ]"
            in line
        ):
            for i in range(len(mAP_axis_t) - 1):
                mAP_axis_t[i] = mAP_axis_t[i + 1]
            mAP_axis_t[3] = line.split("=")[-1][1:-1]
        if (
            "Average Precision (Motion Axis Direction Threshold rotation) (mAP_ADir@r) @[ IoU=0.50      | area=   all | maxDets=100 ]"
            in line
        ):
            for i in range(len(mAP_axis_r) - 1):
                mAP_axis_r[i] = mAP_axis_r[i + 1]
            mAP_axis_r[3] = line.split("=")[-1][1:-1]
        if (
            "Average Precision (Motion Origin Threshold) (mAP_Orig) @[ IoU=0.50      | area=   all | maxDets=100 ]"
            in line
        ):
            for i in range(len(mAP_orig_r) - 1):
                mAP_orig_r[i] = mAP_orig_r[i + 1]
            mAP_orig_r[3] = line.split("=")[-1][1:-1]
        if (
            "Error (Motion Axis Direction) (ERR_ADir) @[ IoU=0.50      | area=   all | maxDets=100 ]"
            in line
        ):
            for i in range(len(error_axis) - 1):
                error_axis[i] = error_axis[i + 1]
            error_axis[3] = line.split("=")[-1][1:-1]
        if (
            "Average Translation Axis Score (ERR_ADir@t) @[ IoU=0.50      | area=   all | maxDets=100 ]"
            in line
        ):
            for i in range(len(error_axis_t) - 1):
                error_axis_t[i] = error_axis_t[i + 1]
            error_axis_t[3] = line.split("=")[-1][1:-1]
        if (
            "Average Rotation Axis Score (ERR_ADir@r) @[ IoU=0.50      | area=   all | maxDets=100 ]"
            in line
        ):
            for i in range(len(error_axis_r) - 1):
                error_axis_r[i] = error_axis_r[i + 1]
            error_axis_r[3] = line.split("=")[-1][1:-1]
        if (
            "Error (Motion Origin) (ERR_Orig) @[ IoU=0.50      | area=   all | maxDets=100 ]"
            in line
        ):
            for i in range(len(error_orig_r) - 1):
                error_orig_r[i] = error_orig_r[i + 1]
            error_orig_r[3] = line.split("=")[-1][1:-1]

    if valid == False:
        print(f"Error: the training log is not complete -> {dir_path}")
        return

    if -1 in box_line or -1 in seg_line:
        print(f"Error: indirect index -> {dir_path}")
        return

    final_mAP_p_pdet = round(float(mAP_pdet[0]) * 100, 1)
    final_mAP_p_mtype = round(float(mAP_mtype[0]) * 100, 1)
    final_mAP_p_motion = round(float(mAP_motion[0]) * 100, 1)

    final_mAP_m_mtype = round(float(mAP_mtype[1]) * 100, 1)
    final_mAP_m_axis = round(float(mAP_axis[1]) * 100, 1)
    final_mAP_m_axis_t = round(float(mAP_axis_t[1]) * 100, 1)
    final_mAP_m_axis_r = round(float(mAP_axis_r[1]) * 100, 1)
    final_mAP_m_orig_r = round(float(mAP_orig_r[1]) * 100, 1)

    final_error_axis = error_axis[1]
    final_error_axis_t = error_axis_t[1]
    final_error_axis_r = error_axis_r[1]
    final_error_orig_r = error_orig_r[1]

    results = f"{final_mAP_p_pdet},{final_mAP_p_mtype},{final_mAP_p_motion},{final_mAP_m_mtype},{final_mAP_m_axis},{final_mAP_m_axis_t},{final_mAP_m_axis_r},{final_mAP_m_orig_r},{final_error_axis},{final_error_axis_t},{final_error_axis_r},{final_error_orig_r}"
    
    with open(f"{dir_path}/result.txt", "w") as f:
        f.write(results)
    

def processDir(dir_path):
    dirs = get_folder_list(dir_path)
    if len(dirs) > 1 or (len(dirs) == 1 and "inference" not in dirs[0]):
        for dir in dirs:
            processDir(dir)
    else:
        extractInfo(dir_path)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="extract final evaluation from the log"
    )
    parser.add_argument(
        "-base",
        "--base_path",
        action="store",
        required=False,
        default="/cs/3dlg-project/3dlg-hcvc/motionnet/extract_logs/extract/",
        help="Input base path",
    )
    parser.add_argument(
        "--exp_path", action="store", required=True, help="The path after the base path"
    )
    args = parser.parse_args()
    exp_dir = args.base_path + args.exp_path

    processDir(exp_dir)

