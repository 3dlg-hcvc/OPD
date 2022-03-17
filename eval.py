import os
import subprocess as subp
import re
import sys
import traceback
import datetime
import shutil
from timeit import default_timer as timer
import argparse
import logging
import copy

FORMAT = '%(asctime)-15s [%(levelname)s] %(message)s'
formatter = logging.Formatter(FORMAT)
logging.basicConfig(format=FORMAT)
log = logging.getLogger('motinnet eval')
log.setLevel(logging.INFO)

PRINT = True

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

def add_handler(filename):
    fh = logging.FileHandler(filename)
    fh.setLevel(logging.INFO)
    fh.setFormatter(formatter)
    log.addHandler(fh)
    return fh

def call(cmd, log, rundir='', env=None, desc=None, testMode=None):
    if not cmd:
        log.warning('No command given')
        return 0
    cwd = os.getcwd()
    res = -1
    prog = None
    try:
        start_time = timer()
        if rundir:
            os.chdir(rundir)
            log.info('Currently in ' + os.getcwd())
        log.info('Running ' + str(cmd))
        prog = subp.Popen(cmd, stdout=subp.PIPE, stderr=subp.PIPE, env=env)
        out, err = prog.communicate()
        if out:
            log.info(out.decode('utf-8'))
        if err:
            log.error('Errors reported running ' + str(cmd))
            log.error(err.decode('utf-8'))
        end_time = timer()
        delta_time = end_time - start_time
        desc_str = desc + ', ' if desc else ''
        desc_str = desc_str + 'cmd="' + str(cmd) + '"'
        log.info('Time=' + str(datetime.timedelta(seconds=delta_time)) + ' for ' + desc_str)
        res = prog.returncode
    except Exception as e:
        if prog:
            prog.kill()
            out, err = prog.communicate()
        log.error(traceback.format_exc())
    os.chdir(cwd)
    return res

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Motionnet Evaluation')
    parser.add_argument('-proj', '--project_path', dest='project_path', action='store',
                        required=False,
                        default='.',
                        help='Input project path')
    parser.add_argument('-base', '--base_path', dest='base_path', action='store',
                        required=False,
                        # default='/project/3dlg-hcvc/motionnet',
                        default='/local-scratch/localhome/hja40/Desktop/Research/proj-motionnet/2DMotion',
                        help='Input base path')
    parser.add_argument('-n', '--conda_name', dest='conda_name', action='store',
                        required=False,
                        default='2dmotion',
                        help='Conda env name')
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
        help="If true, evaluate on the MotionREAL dataset",
    )
    parser.add_argument(
        "--use_threshold",
        action="store_true",
        help="If true, use the motion threshold and use inference file",
    )
    parser.add_argument(
        "--motion_threshold",
        nargs=2,
        type=float,
        default=[10, 0.25],
        help="the threshold for axis and origin for calculating mAP",
    )
    parser.add_argument(
        "--inference_file",
        action="store_true",
        help="If true, use inference file",
    )
    parser.add_argument('--opts', dest='opts', action='store',
                        nargs="*",
                        help='opts')
    args = parser.parse_args()

    env = os.environ.copy()
	
    passed_opts = copy.deepcopy(args.opts)
    if passed_opts is None:
        passed_opts = []
    opts = args.opts
    if opts:
        for i in range(len(opts)):
            opts[i] = '--' + opts[i]
    else:
        opts = []

    dataset_opt = []
    if args.test:
        dataset_opt = ["DATASETS.TEST",  "(\"MotionNet_test\",)"]

    conda_name = args.conda_name
    # Finetune CC
    # cc rgb
    for strategy in args.strategies:
        for model in args.models:
            if model.split('_')[0] == 'cc':
                model_type = 'bmcc'
            elif model.split('_')[0] == 'oc':
                model_type = 'bmoc'
            elif model.split('_')[0] == 'ocv0':
                model_type = 'bmoc_v0'
            elif model.split('_')[0] == 'pm':
                model_type = 'pm'

            if model.split('_')[-1] == 'rgb':
                input_type = 'RGB'
            elif model.split('_')[-1] == 'depth':
                input_type = 'depth'
            elif model.split('_')[-1] == 'rgbd':
                input_type = 'RGBD'
            
            if args.opts:
                eval_ext = '_'.join(args.opts)
            else:
                eval_ext = None
            
            model_base = os.path.join(args.base_path, f'experiments/{strategy}/{model}')
            if args.use_threshold:
                if eval_ext == None:
                    inference_dir = os.path.join(model_base, f'eval_output')
                    eval_ext = f'_{args.motion_threshold[0]}_{args.motion_threshold[1]}'
                else:
                    inference_dir = os.path.join(model_base, f'eval_output_{eval_ext}')
                    eval_ext += f'_{args.motion_threshold[0]}_{args.motion_threshold[1]}'
                eval_dirs = get_folder_list(inference_dir)
                if len(eval_dirs) == 0:
                    continue
                eval_dir = eval_dirs[-1]
                inference_file = os.path.join(eval_dir, 'inference/instances_predictions.pth')

            if args.real:
                if eval_ext == None:
                    eval_ext = "real"
                else:
                    eval_ext = "real_" + eval_ext
                data_path = '/local-scratch/localhome/hja40/Desktop/Research/proj-motionnet/Dataset/MotionDataset_h5_real'
                model_attr_path = '/local-scratch/localhome/hja40/Desktop/Research/proj-motionnet/2DMotion/scripts/data/data_statistics/real-attr.json'
                extra = ["MODEL.PIXEL_MEAN", "[144.7425400388733, 131.67830996768458, 113.38040344244014, 975.0775146484375]", "MODEL.PIXEL_STD", "[20.100716763269578, 20.805474870130748, 23.863171739073888, 291.606201171875]"]
            else:
                data_path = '/local-scratch/localhome/hja40/Desktop/Research/proj-motionnet/Dataset/MotionDataset_h5_6.11'
                model_attr_path = '/local-scratch/localhome/hja40/Desktop/Research/proj-motionnet/2DMotion/scripts/data/data_statistics/urdf-attr.json'
                extra = []

            if args.test:
                if eval_ext == None:
                    eval_ext = "test"
                else:
                    eval_ext = "test_" + eval_ext

            if eval_ext:
                output_dir = os.path.join(model_base, f'eval_output_{eval_ext}/{datetime.datetime.now().isoformat()}')
            else:
                output_dir = os.path.join(model_base, f'eval_output/{datetime.datetime.now().isoformat()}')

            if PRINT:
                # directly use latest folder
                if eval_ext == None:
                    inference_dir = os.path.join(model_base, f'eval_output')
                else:
                    inference_dir = os.path.join(model_base, f'eval_output_{eval_ext}')
                eval_dirs = get_folder_list(inference_dir)
                if len(eval_dirs) == 0:
                    continue
                output_dir = eval_dirs[-1]

            if args.inference_file and not args.use_threshold:
                if eval_ext == None:
                    inference_dir = os.path.join(model_base, f'eval_output')
                else:
                    inference_dir = os.path.join(model_base, f'eval_output_{eval_ext}')
                eval_dirs = get_folder_list(inference_dir)
                if len(eval_dirs) == 0:
                    continue
                eval_dir = eval_dirs[-1]
                inference_file = os.path.join(eval_dir, 'inference/instances_predictions.pth')
            os.makedirs(output_dir, exist_ok=True)
            fh = add_handler(os.path.join(output_dir,'eval.log'))
            ret = 1
            try:
                extra_inference = []
                if args.use_threshold:
                    extra_inference = ["--motion_threshold", f"{args.motion_threshold[0]}", f"{args.motion_threshold[1]}", "--inference-file", f"{inference_file}"]
                if args.inference_file and not args.use_threshold:
                    extra_inference = ["--inference-file", f"{inference_file}"]
                if not PRINT:
                    ret = call(['conda', 'run', '-n', conda_name, 'python', 'evaluate_on_log.py'] +
                        opts + 
                        ['--config-file', os.path.join(args.project_path, f'configs/{model_type}.yaml'), 
                        '--data-path', data_path,
                        '--model_attr_path', model_attr_path,
                        '--input-format', input_type,
                        '--output-dir', output_dir,
                        '--opts', 'MODEL.WEIGHTS', os.path.join(model_base, 'model_best.pth'),
                        ] + dataset_opt + extra, log, env=env)
                else:
                    print(" ".join(['conda', 'run', '-n', conda_name, 'python', 'evaluate_on_log.py'] +
                        opts + extra_inference +
                        ['--config-file', os.path.join(args.project_path, f'configs/{model_type}.yaml'), 
                        '--data-path', data_path,
                        '--model_attr_path', model_attr_path,
                        '--input-format', input_type,
                        '--output-dir', output_dir,
                        '--opts', 'MODEL.WEIGHTS', os.path.join(model_base, 'model_best.pth'),
                        ] + dataset_opt + extra) + " > " + os.path.join(output_dir,'eval.log'))

            except Exception as e:
                log.error(traceback.format_exc())

            if ret == 0:
                log.info(f'Evaluation {model_base} succedded')
            else:
                log.error(f'ERROR:: Evaluation {model_base} failed')

            log.removeHandler(fh)
    
    extra_option = []
    if args.real:
        extra_option.append('--real')
    if args.test:
        extra_option.append('--test')

    if not PRINT:
        ret = call(['conda', 'run', '-n', conda_name, 'python', 'log2latex.py'] + extra_option + ['-ss'] + args.strategies +['-ms']\
                    + args.models + ['--opts'] + passed_opts, log, env=env)
