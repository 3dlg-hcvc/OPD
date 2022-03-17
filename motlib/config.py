from detectron2.config import CfgNode as CN

def add_motionnet_config(cfg: CN):
    _C = cfg
    _C.MODEL.MOTIONSTATE = False
    _C.MODEL.MOTIONNET = CN()
    _C.MODEL.MOTIONNET.TYPE = "BMCC"

    # TODO(AXC): clean up custom configuration for PM
    _C.INPUT.DIMENSION_MEAN = [0, 0, 0]
    _C.INPUT.ROTATION_BIN_NUM = 10
    _C.INPUT.COVER_VALUE = 0.2 
    _C.INPUT.RNG_SEED = 1 

    _C.MODEL.POSE_ITER = 0
    _C.MODEL.MOTION_ITER = 0

    _C.SOLVER.OPTIMIZER = "SGD"