#!/usr/bin/env python
# by boyuanwa 3043994708
# ----------------------------------------------------------------
# Written by Boyuan Wang
# Fall 2019
# ----------------------------------------------------------------
# This helper python aims to flip the vedio reversed in height direction

import numpy as np
import os, glob, time
from skvideo.io import vread, vwrite

model = ''
# model = 'MULTI_LOSS1/' #done
# model = 'MULTI_LOSS2/' #done
# model = 'VGG/'
# model = 'BASELINE/'


directory = 'test_set_results/'
RESULT_DIR = '2_result/down_sampled_he2he/'

TEST_RESULT_DIR = model + RESULT_DIR + directory
t0 = time.time()
output_files = glob.glob(TEST_RESULT_DIR + '*.mp4')
gt_files = [os.path.basename(file)[:-4] for file in output_files]

for output_file in output_files:

    videogen = vread(output_file)
    print("Beginning flipping ",output_file)
    output = np.flip(videogen, axis=1)
    out_file = os.path.basename(output_file)[:-4] + '.mp4'
    vwrite(TEST_RESULT_DIR + out_file, output)
    print("Finishing flipping ",output_file,"\n")

t1 = time.time()
print ('ALL FINISHED. ({:.3f}s)'.format(t1 - t0))