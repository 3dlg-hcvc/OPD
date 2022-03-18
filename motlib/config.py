from detectron2.config import CfgNode as CN

def add_motionnet_config(cfg: CN):
    _C = cfg
    _C.MODEL.MOTIONSTATE = False
    _C.MODEL.MOTIONNET = CN()
    _C.MODEL.MOTIONNET.TYPE = "BMCC"

    _C.INPUT.RNG_SEED = 1 

    _C.SOLVER.OPTIMIZER = "SGD"