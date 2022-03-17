# 2DMotion
Relevant codes for 2DMotion

# Getting Started

Uses Detectron2 0.3 with PyTorch 1.7.1 and CUDA 11 on Ubuntu 20.04

```
conda create -n 2dmotion python=3.7 
conda activate 2dmotion  
pip install -r requirements.txt
```

# Configuration

Configuration files are in `configs/`

(Other parameters can refer to `configs/motionnet.yaml`)

MotionNet configuration
* MODEL.MOTIONNET
  * TYPE: BMOC | BMCC | PM
* INPUT.FORMAT: RGB | RGBD | depth

# Data

## Sapien Data Preprocess
`scripts/data/sapien_data_process`
* Raw Process `scripts/data/sapien_data_process/raw_process`
  * Get all parts from the sapien dataset 
    ```
    python sapien_process.py
    ```
  * Pick the part which we care
    ```
    python sapien_statistics.py
    ```
  * Validate the part name (One example for each part name)
    ```
    python gif_validation.py
    ```
* Valid Process
`scripts/data/sapien_data_process/valid_process`

  * Validate each part manually
    ```
    python valid_part_process.py
    ```
  * (Optional) Get statistic results of the validated parts
    ```
    python valid_part_statistics.py
    ```

## Render Images
(The rendering process makes use of the scene toolkit project https://github.com/3dlg-hcvc/scene-toolkit/tree/articulations-refactor-motionnet)

* Copy `scripts/data/sapien_data_process/valid_process/validModelPart.json` into `$STK/ssc/articulations/`

* Render the raw data we need based on the picked models and parts
    ```
    python render-2DMotion.py # Code is in $STK/ssc/articulations/
    ``` 

## Raw Data Process (For 2DMotion data only)
`scripts/data/raw_data_process/`

* Preprocess 
`scripts/data/raw_data_process/preprocess`
  * Process the raw annotations into the format we want
    ```
    python raw_preprocess_camera.py
    ``` 
  * Add random background from matterport3d into the RGB images in our dataset
    ```
    python TEST_add_background.py
    ```
  * Get the model id in the valid and test set (we use 7:1.5:1.5) 
    ```
    # todo: make these code together
    python test_id.py
    python re_test_id.py
    ```
  * Split the raw dataset to get the process dataset
    ```
    python split.py
    ```
  * (Optional) Calculate the dimension mean for each part 3D bounding box
    ```
    python HELP_avg.py
    ```
* Covert into COCO format
`scripts/data/raw_data_process/coco`
  * Convert the annotation files into COCO annotation format
    ```
    python convert_coco.py
    ```
  * Divide the files into the dataset format detectron2 needs
    ```
    python final_dataset.py
    ``` 
  * Convert the images into h5 format
    ```
    python convert_h5.py
    ```

## REAL Data Preprocess (For MotionREAL data only)
`scripts/data/real_scan_process/`

* Preprocess 
  * Get the name mapping to rename the scans into consistent format and get the diagonal of each scan
    ```
    python motion_real_diagonal.py
    ``` 
  * Check the valid scans, change the annotation into 2DMotion format, get the dataset split
    ```
    python motion_real_statistics.py
    ```
  * Split the raw dataset to get the process dataset
    ```
    python split.py
    ```
* Covert into COCO format
`scripts/data/raw_data_process/coco`
  * Convert the annotation files into COCO annotation format
    ```
    python convert_coco.py
    ```
  * Divide the files into the dataset format detectron2 needs
    ```
    python final_dataset.py
    ``` 
  * Convert the images into h5 format
    ```
    python convert_h5.py
    ```

## OneDoor Data Preprocess (For ANCSH OneDoor data only)
`scripts/data/ancsh_data/`

* Preprocess 
  * Move the objects that are in the OneDoor dataset to a new folder
    ```
    python generate_raw_ancsh.py
    ```
  * Split the raw dataset to get the process dataset
    ```
    python split.py
    ```
* Covert into COCO format
`scripts/data/raw_data_process/coco`
  * Convert the annotation files into COCO annotation format
    ```
    python convert_coco.py
    ```
  * Divide the files into the dataset format detectron2 needs
    ```
    python final_dataset.py
    ``` 
  * Convert the images into h5 format
    ```
    python convert_h5.py
    ```

## 2DMotion baseline network
All library files are in `motilib/`

* Model Training 

  Refer to `run.sh`
  * Compute Canada: the train scripts are in `train_scripts/` 

    Set up the enviroment using `train_scripts/setup.sh`

    Update the PATH in `train_scripts/configure.sh`

    * The scripts for only_det: shell scripts starting with `only_det_*.sh` (The folder with the same name include the detailed scripts)

      ```
      python generate_only_det.py
      ```

  * Lab Computer: Check the commands on above scripts

  * Paras Details:

	To train the default network, specify the data path (`/path/to/Dataset/MotionDataset_6.11'):
	`python train.py --data-path <datapath>` 

	To train the network with specific configuration, input format, output directory, and data path:
	`python train.py --config-file configs/bmcc.yaml --input-format RGB --output-dir train_output/<dirname> --data-path <datapath>` 

	To train different models, specify `configs/pm.yaml` or `configs/bmoc.yaml`.

	Additional configuration parameters can be specified on the command line with `--opts` followed by key-value pairs.

	Examples
	`--opts MODEL.WEIGHTS <path-to-weights> SOLVER.MAX_ITER 40000` 

	Some training configurations
	* TEST.EVAL_PERIOD: How often to evaluate during training.  Set to 0 to disable evaluation during training.
	* MODEL.WEIGHTS: model weights to start with 
	* SOLVER.MAX_ITER: Number of iterations to train for
	* SOLVER.BASE_LR: Base learning rate

* Model Inference and Evaluation
  
  Refer to `run.sh`
  ```
  python eval.py -ss finetuning -ms oc_rgb oc_depth oc_rgbd -n 2dmotion --opts most_frequent_pred origin_NOC
	python eval.py -ss finetuning -ms oc_rgb oc_depth oc_rgbd -n 2dmotion --opts most_frequent_pred random_NOC
	python eval.py -ss finetuning -n 2dmotion
	python eval.py -ss finetuning -n 2dmotion --opts gtbbx gtcat 
	python eval.py -ss finetuning -ms oc_rgb -n 2dmotion --opts most_frequent_gt origin_NOC
	python eval.py -ss finetuning -ms oc_rgb -n 2dmotion --opts most_frequent_gt random_NOC
	python eval.py -ss finetuning -ms oc_rgb oc_depth oc_rgbd -n 2dmotion --opts gtbbx gtcat gtextrinsic
	python eval.py -ss finetuning -ms oc_rgb oc_depth oc_rgbd -n 2dmotion --opts gtextrinsic 
	python eval.py -ss finetuning -ms pm_rgb pm_depth pm_rgbd -n 2dmotion --opts gtpose
  ```

* Visualization 

  Render the visualization and generate html for them
	```
	python render_all.py
	python vis_scripts/vis_html_gen.py
	```

* Run All (todo: update)

	Train -> Inference -> Evaluate -> Visualize
	```
	./run.sh
	```
