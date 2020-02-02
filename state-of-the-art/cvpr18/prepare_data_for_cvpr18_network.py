import os, time

import numpy as np

os.chdir('../../')

PRINT_FREQ = 50


def main(fl):
    t0 = time.time()
    with open(fl) as f:
        text = f.readlines()

    if not os.path.isdir(DATASET_DIR + 'gt'):
        os.makedirs(DATASET_DIR + 'gt')
    if not os.path.isdir(DATASET_DIR + 'input'):
        os.makedirs(DATASET_DIR + 'input')
    for line in text:
        t1 = time.time()
        if line == 'Training:\n' or line == 'Testing:\n':
            continue
        train_id = line.strip().split(' ')[0]
        gt_path = line.strip().split(' ')[1]
        in_path = line.strip().split(' ')[2]
        filename = os.path.basename(gt_path)
        print gt_path
        gt = np.load(gt_path)
        input = np.load(in_path)
        assert gt.shape[0] == input.shape[0]
        t2 = time.time()
        for i in range(gt.shape[0]):
            np.save(DATASET_DIR + 'gt/' + train_id + '_{:03d}'.format(i), gt[i])
            np.save(DATASET_DIR + 'input/' + train_id + '_{:03d}'.format(i), input[i])
            if i % PRINT_FREQ == 0:
                print '\t frame {} {}s'.format(i, time.time() - t2)
                t2 = time.time()
        print gt_path, 'done {}s'.format(time.time() - t1)
    print 'total {}s'.format(time.time() - t0)


if __name__ == '__main__':
    DATASET_DIR = './state-of-the-art/cvpr18/down_sampled_frame_pairs/'
    main('file_list')
    DATASET_DIR = './state-of-the-art/cvpr18/down_sampled_frame_pairs/valid/'
    main('valid_list')
    DATASET_DIR = './state-of-the-art/cvpr18/down_sampled_frame_pairs/test/'
    main('test_list')
