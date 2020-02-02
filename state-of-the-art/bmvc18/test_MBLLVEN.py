import os, time, scipy.io
import tensorflow as tf
import tensorflow.contrib.slim as slim
import numpy as np
from skvideo.io import vwrite, vread

from train_MBLLVEN_raw import network

CROP_FRAME = 16
CROP_HEIGHT = 256
CROP_WIDTH= 256


CHECKPOINT_DIR = './result_MBLLVEN_raw_he2he/'

in_max = 255.0
gt_max = 255.0 # 65535.0

max_frame = 800


def equalize_histogram(image, number_bins=256):
    image_histogram, bins = np.histogram(image.flatten(), number_bins)
    cdf = image_histogram.cumsum()
    cdf = (number_bins - 1) * cdf / cdf[-1] # normalize
    
    image_equalized = np.interp(image.flatten(), bins[:-1], cdf)
    
    return image_equalized.reshape(image.shape)


def process_video(sess, in_image, out_image, in_file, raw, RESULT_DIR, out_file=None):
    input_patch = equalize_histogram(raw, 65536)
    # print '!' * 20, input_patch.shape, input_patch.dtype, input_patch.max(), input_patch.min(), input_patch.mean()

    # print '[DEBUG] input_patch.shape:', input_patch.shape

    i = 0
    j = 0
    k = 0
    overlap = 0.99
    output = np.zeros([input_patch.shape[0], input_patch.shape[1], input_patch.shape[2], 3], dtype='uint16')
    i_range, j_range, k_range = input_patch.shape[0:3]
    weights = np.zeros(output.shape, dtype='uint8')
    frame_patch = CROP_FRAME
    height_patch = CROP_HEIGHT
    width_patch = CROP_WIDTH
    scaling_factor = 65535.0
    done = False
    while i < i_range:
        if i + frame_patch > i_range:
            if done:
                break
            i = i_range - frame_patch
            done = True
        print '[INFO] processing frame', i
        j = 0
        done_j = False
        while j < j_range:
            if j + height_patch > j_range:
                if done_j:
                    break
                j = j_range - height_patch
                done_j = True
            k = 0
            done_k = False
            while k < k_range:
                if k + width_patch > k_range:
                    if done_k:
                        break
                    k = k_range - width_patch
                    done_k = True
                temp = input_patch[i: i + frame_patch, j: j + height_patch, k: k + width_patch, :]
                network_input = np.float32(np.expand_dims(temp, axis=0))
                network_input = np.minimum(network_input / scaling_factor, 1.0)
                # print '[DEBUG] network_input.shape:', network_input.shape
                network_output = sess.run(out_image, feed_dict={in_image: network_input})
                # print '[DEBUG] network_output.shape:', network_output.shape

                if i + frame_patch > i_range:
                    temp = network_output[0, :i_range - i, :, :, :]
                else:
                    temp = network_output[0, :, :, :, :]
                network_output = np.minimum(np.maximum(temp, 0), 1)
                output[i: i + frame_patch, j: (j + height_patch), k: (k + width_patch), :] += (network_output * in_max).astype('uint16')
                weights[i: i + frame_patch, j: (j + height_patch), k: (k + width_patch), :] += 1
                k += int(width_patch * overlap)
            j += int(height_patch * overlap)
        i += int(frame_patch * overlap)

    output = (output / weights.astype(float)).astype('uint8')

    if out_file is None:
        out_file = os.path.basename(in_file)[:-4] + '.mp4'
        # print '[DEBUG] out_file:', out_file
    print '[PROCESS] Processing done. Saving...',
    t0 = time.time()
    vwrite(RESULT_DIR + out_file, output)
    t1 = time.time() - t0
    print 'done. ({:.3f}s)'.format(t1)


def unpack(raw):
    # [F, H/2, W/2, 4] -> [F, H, W, 1]
    F, M, N, _ = raw.shape
    new_raw = np.zeros([F, M * 2, N * 2, 1], dtype='uint16')
    new_raw[:, ::2, ::2, 0] = raw[:, :, :, 0]      # G
    new_raw[:, ::2, 1::2, 0] = raw[:, :, :, 1]     # B
    new_raw[:, 1::2, ::2, 0] = raw[:, :, :, 2]     # R
    new_raw[:, 1::2, 1::2, 0] = raw[:, :, :, 3]    # G
    return new_raw


def main(file_list, RESULT_DIR):
    with open(file_list) as f:
        text = f.readlines()

    train_ids = [line.strip().split(' ')[0] for line in text]
    in_paths = [line.strip().split(' ')[2] for line in text]
    # gt_paths = [line.strip().split(' ')[1] for line in text]
    
    sess = tf.Session()
    in_image = tf.placeholder(tf.float32, [None, CROP_FRAME, CROP_HEIGHT, CROP_WIDTH, 1])
    gt_image = tf.placeholder(tf.float32, [None, CROP_FRAME, CROP_HEIGHT, CROP_WIDTH, 3])
    out_image = network(in_image)
    # print out_image.shape

    saver = tf.train.Saver()
    sess.run(tf.global_variables_initializer())
    ckpt = tf.train.get_checkpoint_state(CHECKPOINT_DIR)
    if ckpt:
        print('loaded ' + ckpt.model_checkpoint_path)
        saver.restore(sess, ckpt.model_checkpoint_path)
    if not os.path.isdir(RESULT_DIR):
        os.makedirs(RESULT_DIR)

    # files = os.listdir(INPUT_DIR)
    # print files
    for i, file0 in enumerate(in_paths):
        t0 = time.time()
        # raw = vread(file0)
        raw = unpack(np.load(file0))
        if raw.shape[0] > max_frame:
            print 'Video with shape', raw.shape, 'is too large. Splitted.'
            count = 0
            begin_frame = 0
            while begin_frame < raw.shape[0]:
                t1 = time.time()
                print 'processing segment %d ...' % (count + 1),
                new_filename = '.'.join(file0.split('.')[:-1] + [str(count)] + file0.split('.')[-1::])
                process_video(sess, in_image, out_image, new_filename, raw[begin_frame: begin_frame + max_frame, :, :, :], RESULT_DIR, train_ids[i] + '.mp4')
                count += 1
                begin_frame += max_frame
                print '\t{}s'.format(time.time() - t1)
        else:
            process_video(sess, in_image, out_image, file0, raw, RESULT_DIR, train_ids[i] + '.mp4')
        print train_ids[i], '\t{}s'.format(time.time() - t0)


if __name__ == '__main__':
    # print '=' * 40
    # print 'testing on trainset'
    # t0 = time.time()
        
    # FILE_LIST = 'file_list'
    # RESULT_DIR = 'result_MBLLVEN_raw/final/'
    # main(FILE_LIST, RESULT_DIR)
    # print 'total time: {}s'.format(time.time() - t0)
    
    # print '=' * 40
    # print 'testing on validset'
    # t0 = time.time()
        
    # FILE_LIST = 'valid_list'
    # RESULT_DIR = 'result_MBLLVEN_raw/valid/'
    # main(FILE_LIST, RESULT_DIR)
    # print 'total time: {}s'.format(time.time() - t0)
    
    print '=' * 40
    print 'testing on testset'
    t0 = time.time()
        
    FILE_LIST = 'test_list'
    RESULT_DIR = './result_MBLLVEN_raw_he2he/test/'
    main(FILE_LIST, RESULT_DIR)
    print 'total time: {}s'.format(time.time() - t0)

