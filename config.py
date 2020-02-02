#!/usr/bin/env python

# ----------------------------------------------------------------
# Written by Haiyang Jiang
# ----------------------------------------------------------------

# file lists ================================================================
FILE_LIST = 'file_list'
VALID_LIST = 'valid_list'
TEST_LIST = 'test_list'
CUSOMIZED_LIST = 'customized_list'

# network.py ================================================================
DEBUG = False


# train.py ================================================================
EXP_NAME = 'down_sampled_he2he'
CHECKPOINT_DIR = './1_checkpoint/' + EXP_NAME + '/'
RESULT_DIR = './2_result/' + EXP_NAME + '/'
LOGS_DIR = RESULT_DIR
TRAIN_LOG_DIR = 'train'
VAL_LOG_DIR = 'val'
# training settings
ALL_FRAME = 200
SAVE_FRAMES = list(range(0, ALL_FRAME, 32))
CROP_FRAME = 16
CROP_HEIGHT = 128
CROP_WIDTH = 128
LOW_LIGHT_THRESHOLD = 0.03

SAVE_FREQ = 5
MAX_EPOCH = 60

FRAME_FREQ = 4
GROUP_NUM = 12

INIT_LR = 1e-4
DECAY_LR = 1e-5
DECAY_EPOCH = 30
NUMPY_RANDOM_SEED = 236
WL = 4.0
WH = 1.0

# test.py ================================================================
TEST_CROP_FRAME = 32
TEST_CROP_HEIGHT = 512
TEST_CROP_WIDTH= 512

MAX_FRAME = 800

OVERLAP = 0.01
OUT_MAX = 255.0

