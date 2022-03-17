import os
import pdb
import re
import argparse
import numpy as np

def sorted_alphanum(file_list):
    """sort the file list by arrange the numbers in filenames in increasing order
    :param file_list: a file list
    :return: sorted file list
    """
    if len(file_list) <= 1:
        return file_list
    convert = lambda text: int(text) if text.isdigit() else text
    alphanum_key = lambda key: [convert(c) for c in re.split('([0-9]+)', key)]
    return sorted(file_list, key=alphanum_key)

def get_folder_list(path):
    if not os.path.exists(path):
        raise OSError('Path {} not exist!'.format(path))

    dir_list = []
    for dirname in os.listdir(path):
        if os.path.isdir(os.path.join(path, dirname)):
            dir_list.append(os.path.join(path, dirname))
    dir_list = sorted_alphanum(dir_list)
    return dir_list

def get_file_list(path, ext=''):
    if not os.path.exists(path):
        raise OSError('Path {} not exist!'.format(path))

    file_list = []
    for filename in os.listdir(path):
        file_ext = os.path.splitext(filename)[1]
        if (ext in file_ext or not ext) and os.path.isfile(os.path.join(path, filename)):
            file_list.append(os.path.join(path, filename))
    file_list = sorted_alphanum(file_list)
    return file_list

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Log to Json')
    parser.add_argument('-base', '--base_path', dest='base_path', action='store',
                        required=False,
                        default='.',
                        help='Input base path')
    parser.add_argument('-ss', '--strategies', dest='strategies', action='store',
                        nargs="+",
                        default=['finetuning', 'end_to_end', 'pretrain_freeze'],
                        help='strategies')
    parser.add_argument('-ms', '--models', dest='models', action='store',
                        nargs="+",
                        default=[
                            'cc_rgb', 'cc_depth', 'cc_rgbd',
                            'oc_rgb', 'oc_depth', 'oc_rgbd',
                            'ocv0_rgb', 'ocv0_depth', 'ocv0_rgbd',
                        ],
                        help='models')
    parser.add_argument(
        "--test",
        action="store_true",
        help="If true, evaluate on the test set instead of val set",
    )
    parser.add_argument(
        "--real",
        action="store_true",
        help="If true, evaluate on the test set instead of val set",
    )
    parser.add_argument('--opts', dest='opts', action='store',
                        nargs="*",
                        help='opts')
    args = parser.parse_args()

    opts = args.opts
    if opts:
        for i in range(len(opts)):
            opts[i] = '--' + opts[i]
    
    eval_ext = "eval_output"
    if args.test:
        eval_ext = eval_ext + '_test'
    if args.real:
        eval_ext = eval_ext + '_real'
    if args.opts:
        eval_ext = eval_ext + '_' + '_'.join(args.opts)
    print(eval_ext)
    # first_key = "Average Precision (Detection) (mAP_Detection) @[ IoU=0.50:0.95 | area=   all | maxDets=100 ]"
    # last_key = "NumInstances for error trans (axis_trans) (nIns_error_trans) @[ IoU=0.95      | area=   all | maxDets=100 ]"
    first_key = "########### Results on the part category ###############"
    second_key = "########### Results on the motion type ###############"


    select_keys = ["@[ IoU=0.50      | area=   all | maxDets=100 ]", "@[ IoU=0.50:0.95 | area=   all | maxDets=100 ]", "@[ IoU=0.75      | area=   all | maxDets=100 ]"]

    # \partdet & \mtype & \axis & \axis (t) & \axis (r) & \orig (r) & \mtype & \axis & \axis (t) & \axis (r) & \orig (r)   \\

    # bbx_order = ["mAP_Detection 0", "mAP_Type", "mAP_Motion ", "mAP_Type ", "mAP_ADir ", "mAP_ADir@t ", "mAP_ADir@r ", "mAP_Orig ", \
    #         "ERR_ADir ", "ERR_ADir@t ", "ERR_ADir@r ", "ERR_Orig "]

    bbx_order = ["mAP_Detection 0", "mAP_Type ", "mAP_+MA ", "mAP_Motion ", "mAP_Type ", "mAP_+MA ", "mAP_Motion "]

    # inst_order = ['nIns_mAP_normal', 'nIns_mAP_trans', 'nIns_mAP_rot', 'nIns_precision_type', 'nIns_error_axis', 'nIns_error_trans', 'nIns_error_rot']
    # inst_repeat = [3, 1, 2, 1, 1, 1, 2]

    # segm_order = ["mAP_Detection 1", "mAP_Detection 0", "mAP_Detection 2"]

    with open(os.path.join(args.base_path, f'experiments/result_latex_{eval_ext}.txt'), 'w+') as fp:
        for strategy in args.strategies:
            fp.write('#####################################################################\n')
            fp.write(f'{strategy}\n')
            fp.write('#####################################################################\n')
            for model in args.models:
                model_base = os.path.join(args.base_path, f'experiments/{strategy}/{model}')
                log_dir = os.path.join(model_base, f'{eval_ext}')
                eval_dirs = get_folder_list(log_dir)
                if len(eval_dirs) == 0:
                    continue
                eval_dir = eval_dirs[-1]
                eval_file = os.path.join(eval_dir, 'eval.log')
                
                print(f"{strategy} {model}")

                if not os.path.isfile(eval_file):
                    print(f"No file {strategy} {model}")
                    continue

                with open(eval_file, "r") as f:
                    data = f.read().split('\n')
                
                # print(data[0])
                # import pdb
                # pdb.set_trace()
                if not model in data[0]:
                    print(f"Wrong model {strategy} {model}")
                    continue

                part_line = []
                motion_line = []
                end = len(data)
                for idx, line in enumerate(data):
                    if first_key in line:
                        part_line.append(idx)
                    if second_key in line:
                        motion_line.append(idx)
                    
                assert len(part_line) == len(motion_line)

                if len(part_line) == 0:
                    print(f"Invalid eval {strategy} {model}")
                    continue

                # bbox
                bbx_part_val_str = []
                # Get the attribute-value pair in the macro average over part category
                for i in range(part_line[0], motion_line[0]):
                    for j, select_key in enumerate(select_keys):
                        if "Average Precision (Detection)" not in data[i] and j > 0:
                            continue

                        if select_key in data[i]:
                            attr_name = re.findall("\([0-9A-Za-z_ +@]+\)", data[i])[-1]
                            eval_value = float(data[i].split('=')[-1])
                            if ("mAP" in data[i] and "nIns" not in data[i]) or "Precision" in data[i]:
                                eval_value = round(eval_value*100.0, 1)
                            elif "nIns" in data[i]:
                                eval_value = round(eval_value, 1)
                            bbx_part_val_str.append(attr_name[1:-1] + ' ' + str(j) + ' ' + str(eval_value))

                bbx_motion_val_str = []
                # Get the attribute-value pair in the macro average over motion type
                for i in range(motion_line[0], part_line[1]):
                    for j, select_key in enumerate(select_keys):
                        if "Average Precision (Detection)" not in data[i] and j > 0:
                            continue

                        if select_key in data[i]:
                            attr_name = re.findall("\([0-9A-Za-z_ +@]+\)", data[i])[-1]
                            eval_value = float(data[i].split('=')[-1])
                            if ("mAP" in data[i] and "nIns" not in data[i]) or "Precision" in data[i]:
                                eval_value = round(eval_value*100.0, 1)
                            elif "nIns" in data[i]:
                                eval_value = round(eval_value, 1)
                            bbx_motion_val_str.append(attr_name[1:-1] + ' ' + str(j) + ' ' + str(eval_value))
                
                # Get the first two attributes over the part category
                bbx_val_list = []
                for key in bbx_order[:4]:
                    for entry in bbx_part_val_str:
                        if key in entry:
                            bbx_val_list.append(entry.split(' ')[-1])

                # Get the left attributes over the motion types
                for key in bbx_order[4:]:
                    for entry in bbx_motion_val_str:
                        if key in entry:
                            bbx_val_list.append(entry.split(' ')[-1])
                
                # inst_val_list = []
                # for idx, key in enumerate(inst_order):
                #     for entry in bbx_val_str:
                #         if key in entry:
                #             for i in range(inst_repeat[idx]):
                #                 inst_val_list.append(entry.split(' ')[-1])
                
                # # segms
                # segm_val_str = []
                # for i in range(starts[1], ends[1]+1):
                #     for j, select_key in enumerate(select_keys):
                #         if "Average Precision (Detection)" in data[i] and select_key in data[i]:
                #             attr_name = re.findall("\([0-9A-Za-z_ @]+\)", data[i])[-1]
                #             eval_value = float(data[i].split('=')[-1])
                #             if "mAP" in data[i] or "Precision" in data[i]:
                #                 eval_value = round(eval_value*100.0, 1)
                #             segm_val_str.append(attr_name[1:-1] + ' ' + str(j) + ' ' + str(eval_value))
                
                # bbx_detect_list = []
                # for key in segm_order:
                #     for entry in bbx_val_str:
                #         if key in entry:
                #             bbx_detect_list.append(entry.split(' ')[-1])

                # segm_detect_list = []
                # for key in segm_order:
                #     for entry in segm_val_str:
                #         if key in entry:
                #             segm_detect_list.append(entry.split(' ')[-1])

                fp.write('\n')
                fp.write('###################################\n')
                fp.write(f'#  {model}  #\n')
                
                # fig results
                for key in bbx_order:
                    fp.write(key+' ')
                fp.write('\n')

                fp.write(f'# Evaluation bbx results #\n')

                for val in bbx_val_list:
                    fp.write(val + ' & ')
                fp.write('\n')

                # fp.write(f'# Evaluation num instances results #\n')

                # for val in inst_val_list:
                #     fp.write(val + ' & ')
                # fp.write('\n')

                # fp.write(f'# Evaluation bbx segm results #\n')

                # for key in segm_order:
                #     fp.write(key+' ')
                # fp.write('\n')

                # for val in bbx_detect_list:
                #     fp.write(val + ' & ')

                # for val in segm_detect_list:
                #     fp.write(val + ' & ')
                # fp.write('\n')

                fp.write('###################################\n')
            



