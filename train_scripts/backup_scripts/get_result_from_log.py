# This script is to extract the useful evaluation results from the log
import os
import glob
import argparse

def extractInfo(log_path):
    out_file = open(log_path)

    # Used to judge if the train log is complete
    valid = False
    index = 0

    # Used to locate the final evaluation
    box_line = [-1, -1]
    seg_line = [-1, -1]
    # Used to store the evaluation results
    mAP_pdet = [-1, -1]
    mAP_mtype = [-1, -1, -1, -1]
    mAP_motion = [-1, -1, -1, -1]
    mAP_MA = [-1, -1, -1, -1]
    mAP_axis = [-1, -1, -1, -1]
    mAP_axis_t = [-1, -1, -1, -1]
    mAP_axis_r = [-1, -1, -1, -1]
    mAP_orig_r = [-1, -1, -1, -1]
    # Error evaluation metric
    error_axis = [-1, -1, -1, -1]
    error_axis_t = [-1, -1, -1, -1]
    error_axis_r = [-1, -1, -1, -1]
    error_orig_r = [-1, -1, -1, -1]
    number_error_all = [-1, -1, -1, -1]
    number_error_trans = [-1, -1, -1, -1]
    number_error_rot = [-1, -1, -1, -1]
    # Total error evaluation metric
    total_error_axis = [-1, -1, -1, -1]
    total_error_axis_t = [-1, -1, -1, -1]
    total_error_axis_r = [-1, -1, -1, -1]
    total_error_orig_r = [-1, -1, -1, -1]
    total_number_error_all = [-1, -1, -1, -1]
    total_number_error_trans = [-1, -1, -1, -1]
    total_number_error_rot = [-1, -1, -1, -1]
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
            for i in range(len(mAP_motion) - 1):
                mAP_motion[i] = mAP_motion[i + 1]
            mAP_motion[3] = line.split("=")[-1][1:-1]
        if (
            "Average Precision (+MA) Threshold) (mAP_+MA) @[ IoU=0.50      | area=   all | maxDets=100 ]"
            in line
        ):
            for i in range(len(mAP_MA) - 1):
                mAP_MA[i] = mAP_MA[i + 1]
            mAP_MA[3] = line.split("=")[-1][1:-1]
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
        if (
            "NumInstances for error axis (axis) (nIns_error_axis) @[ IoU=0.50      | area=   all | maxDets=100 ]"
            in line
        ):
            for i in range(len(number_error_all) - 1):
                number_error_all[i] = number_error_all[i + 1]
            number_error_all[3] = line.split("=")[-1][1:-1]
        if (
            "NumInstances for error trans (axis_trans) (nIns_error_trans) @[ IoU=0.50      | area=   all | maxDets=100 ]"
            in line
        ):
            for i in range(len(number_error_trans) - 1):
                number_error_trans[i] = number_error_trans[i + 1]
            number_error_trans[3] = line.split("=")[-1][1:-1]
        if (
            "NumInstances for error rot (origin, axis_rot) (nIns_error_rot) @[ IoU=0.50      | area=   all | maxDets=100 ]"
            in line
        ):
            for i in range(len(number_error_rot) - 1):
                number_error_rot[i] = number_error_rot[i + 1]
            number_error_rot[3] = line.split("=")[-1][1:-1]

        if (
            "total error of axis (TotERR_Adir) @[ IoU=0.50      | area=   all | maxDets=100 ]"
            in line
        ):
            for i in range(len(total_error_axis) - 1):
                total_error_axis[i] = total_error_axis[i + 1]
            total_error_axis[3] = line.split("=")[-1][1:-1]
        if (
            "total error of axis (translation) (TotERR_Adir@t) @[ IoU=0.50      | area=   all | maxDets=100 ]"
            in line
        ):
            for i in range(len(total_error_axis_t) - 1):
                total_error_axis_t[i] = total_error_axis_t[i + 1]
            total_error_axis_t[3] = line.split("=")[-1][1:-1]
        if (
            "total error of axis (rotation) (TotERR_Adir@r) @[ IoU=0.50      | area=   all | maxDets=100 ]"
            in line
        ):
            for i in range(len(total_error_axis_r) - 1):
                total_error_axis_r[i] = total_error_axis_r[i + 1]
            total_error_axis_r[3] = line.split("=")[-1][1:-1]
        if (
            "total error of origin (TotERR_Orig) @[ IoU=0.50      | area=   all | maxDets=100 ]"
            in line
        ):
            for i in range(len(total_error_orig_r) - 1):
                total_error_orig_r[i] = total_error_orig_r[i + 1]
            total_error_orig_r[3] = line.split("=")[-1][1:-1]
        if (
            "total number of predictions (TotIns_pred) @[ IoU=0.50      | area=   all | maxDets=100 ]"
            in line
        ):
            for i in range(len(total_number_error_all) - 1):
                total_number_error_all[i] = total_number_error_all[i + 1]
            total_number_error_all[3] = line.split("=")[-1][1:-1]
        if (
            "total number of predictions (translation) (TotIns_pred@t) @[ IoU=0.50      | area=   all | maxDets=100 ]"
            in line
        ):
            for i in range(len(total_number_error_trans) - 1):
                total_number_error_trans[i] = total_number_error_trans[i + 1]
            total_number_error_trans[3] = line.split("=")[-1][1:-1]
        if (
            "total number of predictions (rotation) (TotIns_pred@r) @[ IoU=0.50      | area=   all | maxDets=100 ]"
            in line
        ):
            for i in range(len(total_number_error_rot) - 1):
                total_number_error_rot[i] = total_number_error_rot[i + 1]
            total_number_error_rot[3] = line.split("=")[-1][1:-1]

    # if valid == False:
    #     print(f"Error: the training log is not complete -> {dir_path}")
    #     return

    if -1 in box_line or -1 in seg_line:
        print(f"Error: indirect index -> {log_path}")
        return

    final_mAP_p_pdet = round(float(mAP_pdet[0]) * 100, 1)
    final_mAP_p_mtype = round(float(mAP_mtype[0]) * 100, 1)
    final_mAP_p_MA = round(float(mAP_MA[0]) * 100, 1)
    final_mAP_p_motion = round(float(mAP_motion[0]) * 100, 1)

    final_mAP_m_mtype = round(float(mAP_mtype[1]) * 100, 1)
    final_mAP_m_axis = round(float(mAP_axis[1]) * 100, 1)
    final_mAP_m_axis_t = round(float(mAP_axis_t[1]) * 100, 1)
    final_mAP_m_axis_r = round(float(mAP_axis_r[1]) * 100, 1)
    final_mAP_m_orig_r = round(float(mAP_orig_r[1]) * 100, 1)
    final_mAP_m_MA = round(float(mAP_MA[1]) * 100, 1)
    final_mAP_m_motion = round(float(mAP_motion[1]) * 100, 1)

    # Defined match in motion-averaged
    final_error_axis = round(float(error_axis[1]), 2)
    final_error_axis_t = round(float(error_axis_t[1]), 2)
    final_error_axis_r = round(float(error_axis_r[1]), 2)
    final_error_orig_r = round(float(error_orig_r[1]), 2)

    # match number
    final_number_error_all = int(float(number_error_all[1]))
    final_number_error_trans = int(float(number_error_trans[1]))
    final_number_error_rot = int(float(number_error_rot[1]))

    # Defined match in motion-averaged
    final_total_error_axis = round(float(total_error_axis[1]), 2)
    final_total_error_axis_t = round(float(total_error_axis_t[1]), 2)
    final_total_error_axis_r = round(float(total_error_axis_r[1]), 2)
    final_total_error_orig_r = round(float(total_error_orig_r[1]), 2)

    # match number
    final_total_number_error_all = int(float(total_number_error_all[1]))
    final_total_number_error_trans = int(float(total_number_error_trans[1]))
    final_total_number_error_rot = int(float(total_number_error_rot[1]))

    # results = f"{final_mAP_p_pdet} & {final_mAP_p_mtype} & {final_mAP_p_motion} & {final_mAP_m_mtype} & {final_mAP_m_axis} & {final_mAP_m_axis_t} & {final_mAP_m_axis_r} & {final_mAP_m_orig_r}"
    # log_results = f"{final_mAP_p_pdet},{final_mAP_p_mtype},{final_mAP_p_motion},{final_mAP_m_mtype},{final_mAP_m_axis},{final_mAP_m_axis_t},{final_mAP_m_axis_r},{final_mAP_m_orig_r},{final_error_axis},{final_error_axis_t},{final_error_axis_r},{final_error_orig_r}"
    
    results = f"{final_mAP_p_pdet} & {final_mAP_p_mtype} & {final_mAP_p_MA} & {final_mAP_p_motion} & {final_mAP_m_mtype} & {final_mAP_m_MA} & {final_mAP_m_motion}     & {final_error_axis} & {final_error_axis_t} & {final_error_axis_r} & {final_error_orig_r} & {final_number_error_all} & {final_number_error_trans} & {final_number_error_rot}    & {final_total_error_axis} & {final_total_error_axis_t} & {final_total_error_axis_r} & {final_total_error_orig_r} & {final_total_number_error_all} & {final_total_number_error_trans} & {final_total_number_error_rot}"
    # log_results = f"{final_mAP_p_pdet},{final_mAP_p_mtype},{final_mAP_p_motion},{final_mAP_m_mtype},{final_mAP_m_axis},{final_mAP_m_axis_t},{final_mAP_m_axis_r},{final_mAP_m_orig_r},{final_error_axis},{final_error_axis_t},{final_error_axis_r},{final_error_orig_r}"
    # log_results = f"& {final_error_axis} & {final_error_axis_t} & {final_error_axis_r} & {final_error_orig_r} & {final_number_error_all} & {final_number_error_trans} & {final_number_error_rot}"
    

    print(results)
    # print(log_results)
    


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="extract final evaluation from the log"
    )
    parser.add_argument(
        "--log", action="store", required=True, help="The path for the log"
    )
    args = parser.parse_args()

    extractInfo(args.log)

