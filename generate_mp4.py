#!/usr/bin/env python
# by boyuanwa 3043994708
# ----------------------------------------------------------------
# Written by Boyuan Wang
# Fall 2019
# ----------------------------------------------------------------
# This helper python aims to cnvert from npy file to mp4 file

import os, glob, time
import numpy as np
from skvideo.io import vread, vwrite

TEST_LIST = 'test_list'
TEST_RESULT_DIR = '0_data/gt_he/'

# get train IDs
with open(TEST_LIST) as f:
    text = f.readlines()
train_files = text
t0 = time.time()

ids = [line.strip().split(' ')[0] for line in train_files]

output_files = glob.glob(TEST_RESULT_DIR + '*.npy')

for file in output_files:
    file_name = os.path.basename(file)[:-4]
    # if file_name[:-4] not in ids:
    #     continue
    output = np.load(file)
    out_file = file_name + '.mp4'
    vwrite(TEST_RESULT_DIR + out_file, output)
    print("Finishing converting from npy to mp4 For:", file_name, "\n")

t1 = time.time()
print ('ALL FINISHED. ({:.3f}s)'.format(t1 - t0))