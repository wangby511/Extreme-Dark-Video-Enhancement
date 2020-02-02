import os, glob, time

import tensorflow as tf
import numpy as np
from skvideo.io import vread, vwrite


directory = 'test_set_results/'

TEST_RESULT_DIR = 'result_downsampled_he2he/test_videos/'

MAX_VAL = 255

sess = tf.Session()
t_vid1 = tf.placeholder(tf.uint8, [None, None, None, None])
t_vid2 = tf.placeholder(tf.uint8, [None, None, None, None])
t_psnr = tf.reduce_mean(tf.image.psnr(t_vid1, t_vid2, MAX_VAL))
t_ssim = tf.reduce_mean(tf.image.ssim(t_vid1, t_vid2, MAX_VAL))


def get_psnr_ssim(sess, vid1, vid2):
    assert vid1.shape[0] == vid2.shape[0]
    psnr = 0
    ssim = 0
    N = 20
    for i in range(vid1.shape[0] / N):
        psnr += sess.run(t_psnr, feed_dict={t_vid1: vid1[i * N:(i + 1) * N], t_vid2: vid2[i * N:(i + 1) * N]})
        ssim += sess.run(t_ssim, feed_dict={t_vid1: vid1[i * N:(i + 1) * N], t_vid2: vid2[i * N:(i + 1) * N]})
    return psnr / vid1.shape[0] * N, ssim / vid1.shape[0] * N


def brightness(vid):
    R, G, B = vid[:, :, :, 0], vid[:, :, :, 1], vid[:, :, :, 2]
    return (0.2126 * R + 0.7152 * G + 0.0722 * B) # refer to https://en.wikipedia.org/wiki/Relative_luminance


def get_mse_mabd(vid1, vid2):
    b_vid1 = brightness(vid1)
    b_vid2 = brightness(vid2)
    mabd1 = np.diff(b_vid1).mean(axis=(1,2))
    mabd2 = np.diff(b_vid2).mean(axis=(1,2))
    return ((mabd1 - mabd2) ** 2).mean()


output_files = glob.glob(TEST_RESULT_DIR + '*')
gt_files = [os.path.basename(file)[:-4] for file in output_files]

if 'psnr_ssim_mabd' in os.listdir('.'):
    os.rename('psnr_ssim_mabd', 'psnr_ssim_mabd' + '_' + str(time.localtime().tm_mon).zfill(2) + str(time.localtime().tm_mday).zfill(2) + '-' + str(time.localtime().tm_hour).zfill(2) + str(time.localtime().tm_min).zfill(2))

with open('psnr_ssim_mabd', 'w') as f:
    pass

all_psnr = 0
all_ssim = 0
all_mabd = 0

for output_file in output_files:
    out_vid = vread(output_file)
    gt_file = os.path.basename(output_file).split('.')[0][:-4] + '.npy'
    gt_vid = np.load('../../0_data/gt_he/' + gt_file)
    t0 = time.time()
    psnr, ssim = get_psnr_ssim(sess, out_vid, gt_vid)
    t1 = time.time()
    mabd = get_mse_mabd(out_vid, gt_vid)
    t2 = time.time()
    print('Done.\t{}s\t{}s'.format(t1 - t0, t2 - t1))
    with open('psnr_ssim_mabd', 'a') as f:
        f.write(os.path.basename(output_file)[:-4] + ' ' + str(psnr) + ' ' + str(ssim) + ' ' + str(mabd) + '\n')
    all_psnr += psnr
    all_ssim += ssim
    all_mabd += mabd

with open('psnr_ssim_mabd', 'a') as f:
    f.write('\n' * 3 + 'overall_average ' + str(all_psnr / len(gt_files)) + ' ' + str(all_ssim / len(gt_files)) + ' ' + str(all_mabd / len(gt_files)) + '\n')
