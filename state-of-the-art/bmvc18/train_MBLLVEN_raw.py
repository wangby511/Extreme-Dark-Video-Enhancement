from __future__ import division
import os, time, scipy.io
import tensorflow as tf
import tensorflow.contrib.slim as slim
import numpy as np
import glob
import cv2
from skvideo.io import vwrite

import vgg19


WEIGHT_LOW = 4
WEIGHT_HIGH = 1

INPUT_DIR = ''
GT_DIR = ''
CHECKPOINT_DIR = './result_MBLLVEN_raw_he2he/'
RESULT_DIR = './result_MBLLVEN_raw_he2he/'

FILE_LIST = 'file_list'
VALID_LIST = 'valid_list'

# get train IDs, input files, groundtruth files
with open(FILE_LIST) as f:
    text = f.readlines()

train_ids = [line.strip().split(' ')[0] for line in text]
gt_files = [line.strip().split(' ')[1] for line in text]
in_files = [line.strip().split(' ')[2] for line in text]


with open(VALID_LIST) as f:
    text = f.readlines()
validate_files = text

valid_ids = [line.strip().split(' ')[0] for line in validate_files]
valid_gt_files = [line.strip().split(' ')[1] for line in validate_files]
valid_in_files = [line.strip().split(' ')[2] for line in validate_files]


SAVE_FREQ = 5
ALL_FRAME = 200
SAVE_FRAMES = list(range(0, ALL_FRAME, 32))
MAX_EPOCH = 60

FRAME_FREQ = 8
GROUP_NUM = 10

INIT_LR = 1e-4


context_loss_weight = 1e-6
DECAY_RATE = 0.99
INITIAL_LR = 1e-4 / DECAY_RATE

CROP_FRAME = 16
CROP_HEIGHT = 224
CROP_WIDTH = 224

LOAD_TRAIN_FUNC = np.load
LOAD_GT_FUNC = np.load


DEBUG = 0
if DEBUG == 1:
    SAVE_FREQ = 2
    train_ids = train_ids[0:4]
    # print train_ids
    MAX_EPOCH = 50
    valid_in_files = valid_in_files[0:3]


def lrelu(x):
    return tf.maximum(x * 0.2, x)
    # return tf.maximum(0.0, x)


def em_branch(input, prefix='em_branch_'):
    # input should be of shape [batch_size, frame_count, height, width, 16]
    conv = slim.conv3d(input, 8, [3, 3, 3], rate=1, activation_fn=lrelu, scope=prefix + 'g_conv1', padding='SAME')

    padding_method = 'VALID'
    conv1 = slim.conv3d(conv, 16, [5, 5, 5], rate=1, activation_fn=lrelu, scope=prefix + 's_conv1', padding=padding_method)
    conv2 = slim.conv3d(conv1, 16, [5, 5, 5], rate=1, activation_fn=lrelu, scope=prefix + 's_conv2', padding=padding_method)
    conv3 = slim.conv3d(conv2, 16, [5, 5, 5], rate=1, activation_fn=lrelu, scope=prefix + 's_conv3', padding=padding_method)
    #
    # shape_image = tf.placeholder(tf.float32, [BATCH_SIZE, CROP_FRAME - 8, CROP_HEIGHT - 8, CROP_WIDTH - 8, 16])
    #
    # pool_size = 1
    # deconv_filter1 = tf.Variable(tf.truncated_normal([pool_size, pool_size, pool_size, 16, 16], stddev=0.02))
    # deconv1 = tf.nn.conv3d_transpose(conv3, deconv_filter1, tf.shape(shape_image), strides=[1, pool_size, pool_size, pool_size, 1])
    # deconv1 = lrelu(deconv1)
    #
    # # print deconv1.shape
    # # print 'conv1.shape[:-1] + (8,):', tuple(conv1.shape[:-1]) + (8,)
    #
    # shape_image = tf.placeholder(tf.float32, [BATCH_SIZE, CROP_FRAME - 4, CROP_HEIGHT - 4, CROP_WIDTH - 4, 8])
    # pool_size = 1
    # deconv_filter2 = tf.Variable(tf.truncated_normal([pool_size, pool_size, pool_size, 8, 16], stddev=0.02))
    # deconv2 = tf.nn.conv3d_transpose(deconv1, deconv_filter2, tf.shape(shape_image), strides=[1, pool_size, pool_size, pool_size, 1])
    # deconv2 = lrelu(deconv2)
    #
    # # print deconv2.shape
    # shape_image = tf.placeholder(tf.float32, [BATCH_SIZE, CROP_FRAME, CROP_HEIGHT, CROP_WIDTH, 3])
    # pool_size = 1
    # deconv_filter3 = tf.Variable(tf.truncated_normal([pool_size, pool_size, pool_size, 3, 8], stddev=0.02))
    # deconv3 = tf.nn.conv3d_transpose(deconv2, deconv_filter3, tf.shape(shape_image), strides=[1, pool_size, pool_size, pool_size, 1])
    # deconv3 = lrelu(deconv3)

    # print deconv3.shape
    deconv1 = slim.conv3d_transpose(conv3, 16, [5, 5, 5], activation_fn=lrelu, scope=prefix + 's_deconv1', padding=padding_method)
    deconv2 = slim.conv3d_transpose(deconv1, 8, [5, 5, 5], activation_fn=lrelu, scope=prefix + 's_deconv2', padding=padding_method)
    deconv3 = slim.conv3d_transpose(deconv2, 3, [5, 5, 5], activation_fn=lrelu, scope=prefix + 's_deconv3', padding=padding_method)


    if DEBUG == 1:
        print 'conv.shape:', conv.shape
        print 'conv1.shape:', conv1.shape
        print 'conv2.shape:', conv2.shape
        print 'conv3.shape:', conv3.shape
        print 'deconv1.shape:', deconv1.shape
        print 'deconv2.shape:', deconv2.shape
        print 'deconv3.shape:', deconv3.shape
    return deconv3


def network(input):
    # input should be of shape [batch_size, frame_count, height, width, 3]

    fem_conv1 = slim.conv3d(input, 16, [3, 3, 3], rate=1, activation_fn=lrelu, scope='fem_conv1')
    # em1 = em_branch(fem_conv1, prefix='em_branch1_')
    em1 = em_branch(fem_conv1[:, :, :, :, :3], prefix='em_branch1_')

    # fem_conv2 = slim.conv3d(fem_conv1, 16, [3, 3, 3], rate=1, activation_fn=lrelu, scope='fem_conv2')
    fem_conv2 = slim.conv3d(fem_conv1[:, :, :, :, 3:], 16, [3, 3, 3], rate=1, activation_fn=lrelu, scope='fem_conv2')
    # em2 = em_branch(fem_conv2, prefix='em_branch2_')
    em2 = em_branch(fem_conv2[:, :, :, :, :3], prefix='em_branch2_')

    # fem_conv3 = slim.conv3d(fem_conv2, 16, [3, 3, 3], rate=1, activation_fn=lrelu, scope='fem_conv3')
    fem_conv3 = slim.conv3d(fem_conv2[:, :, :, :, 3:], 16, [3, 3, 3], rate=1, activation_fn=lrelu, scope='fem_conv3')
    # em3 = em_branch(fem_conv3, prefix='em_branch3_')
    em3 = em_branch(fem_conv3[:, :, :, :, :3], prefix='em_branch3_')

    # fem_conv4 = slim.conv3d(fem_conv3, 16, [3, 3, 3], rate=1, activation_fn=lrelu, scope='fem_conv4')
    fem_conv4 = slim.conv3d(fem_conv3[:, :, :, :, 3:], 16, [3, 3, 3], rate=1, activation_fn=lrelu, scope='fem_conv4')
    # em4 = em_branch(fem_conv4, prefix='em_branch4_')
    em4 = em_branch(fem_conv4[:, :, :, :, :3], prefix='em_branch4_')

    # fem_conv5 = slim.conv3d(fem_conv4, 16, [3, 3, 3], rate=1, activation_fn=lrelu, scope='fem_conv5')
    fem_conv5 = slim.conv3d(fem_conv4[:, :, :, :, 3:], 16, [3, 3, 3], rate=1, activation_fn=lrelu, scope='fem_conv5')
    # em5 = em_branch(fem_conv5, prefix='em_branch5_')
    em5 = em_branch(fem_conv5[:, :, :, :, :3], prefix='em_branch5_')
    
    before_fusion = tf.concat([input, em1, em2, em3, em4, em5], axis=-1)
    # before_fusion = tf.concat(em_s, axis=-1)
    fm = slim.conv3d(before_fusion, 3, [1, 1, 1], rate=1, activation_fn=None, scope='fusion_module')

    return fm



def _10_branch():
    # fem_conv6 = slim.conv3d(fem_conv5, 16, [3, 3, 3], rate=1, activation_fn=lrelu, scope='fem_conv6')
    fem_conv6 = slim.conv3d(fem_conv5[:, :, :, :, 3:], 16, [3, 3, 3], rate=1, activation_fn=lrelu, scope='fem_conv6')
    # em6 = em_branch(fem_conv6, prefix='em_branch6_')
    em6 = em_branch(fem_conv6[:, :, :, :, :3], prefix='em_branch6_')

    # fem_conv7 = slim.conv3d(fem_conv6, 16, [3, 3, 3], rate=1, activation_fn=lrelu, scope='fem_conv7')
    fem_conv7 = slim.conv3d(fem_conv6[:, :, :, :, 3:], 16, [3, 3, 3], rate=1, activation_fn=lrelu, scope='fem_conv7')
    # em7 = em_branch(fem_conv7, prefix='em_branch7_')
    em7 = em_branch(fem_conv7[:, :, :, :, :3], prefix='em_branch7_')

    # fem_conv8 = slim.conv3d(fem_conv7, 16, [3, 3, 3], rate=1, activation_fn=lrelu, scope='fem_conv8')
    fem_conv8 = slim.conv3d(fem_conv7[:, :, :, :, 3:], 16, [3, 3, 3], rate=1, activation_fn=lrelu, scope='fem_conv8')
    # em8 = em_branch(fem_conv8, prefix='em_branch8_')
    em8 = em_branch(fem_conv8[:, :, :, :, :3], prefix='em_branch8_')

    # fem_conv9 = slim.conv3d(fem_conv8, 16, [3, 3, 3], rate=1, activation_fn=lrelu, scope='fem_conv9')
    fem_conv9 = slim.conv3d(fem_conv8[:, :, :, :, 3:], 16, [3, 3, 3], rate=1, activation_fn=lrelu, scope='fem_conv9')
    # em9 = em_branch(fem_conv9, prefix='em_branch9_')
    em9 = em_branch(fem_conv9[:, :, :, :, :3], prefix='em_branch9_')

    # fem_conv10 = slim.conv3d(fem_conv9, 16, [3, 3, 3], rate=1, activation_fn=lrelu, scope='fem_conv10')
    fem_conv10 = slim.conv3d(fem_conv9[:, :, :, :, 3:], 16, [3, 3, 3], rate=1, activation_fn=lrelu, scope='fem_conv10')
    # em10 = em_branch(fem_conv10, prefix='em_branch10_')
    em10 = em_branch(fem_conv10[:, :, :, :, :3], prefix='em_branch10_')

    before_fusion = tf.concat([input, em1, em2, em3, em4, em5, em6, em7, em8, em9, em10], axis=-1)
    # before_fusion = tf.concat(em_s, axis=-1)
    fm = slim.conv3d(before_fusion, 3, [1, 1, 1], rate=1, activation_fn=None, scope='fusion_module')

    return fm


def unpack(raw):
    # [F, H/2, W/2, 4] -> [F, H, W, 1]
    F, M, N, _ = raw.shape
    new_raw = np.zeros([F, M * 2, N * 2, 1], dtype='uint16')
    new_raw[:, ::2, ::2, 0] = raw[:, :, :, 0]      # G
    new_raw[:, ::2, 1::2, 0] = raw[:, :, :, 1]     # B
    new_raw[:, 1::2, ::2, 0] = raw[:, :, :, 2]     # R
    new_raw[:, 1::2, 1::2, 0] = raw[:, :, :, 3]    # G
    return new_raw


def demosaic(in_vid, converter=cv2.COLOR_BayerGB2BGR):
    # [1, F, H, W, 1] -> [1, F, H, W, 3]
    bayer_input = in_vid[0, :, :, :, 0]
    bayer_input = (bayer_input * 65535).astype('uint16')
    rgb_input = np.zeros([1, bayer_input.shape[0], bayer_input.shape[1], bayer_input.shape[2], 3])
    for j in range(bayer_input.shape[0]):
        rgb_input[0, j] = cv2.cvtColor(bayer_input[j], converter)
    return rgb_input / 65535.0


def crop(raw, demosaiced, gt_raw, H, W, start_frame=0):
    # inputs must be in a form of [batch_num, frame_num, height, width, channel_num]
    tt = start_frame
    xx = np.random.randint(0, W - CROP_WIDTH)
    yy = np.random.randint(0, H - CROP_HEIGHT)

    input_patch = raw[:, tt:tt + CROP_FRAME, yy:yy + CROP_HEIGHT, xx:xx + CROP_WIDTH, :]
    demosaiced = demosaiced[:, tt:tt + CROP_FRAME, yy:yy + CROP_HEIGHT, xx:xx + CROP_WIDTH, :]
    gt_patch = gt_raw[:, tt:tt + CROP_FRAME, yy:yy + CROP_HEIGHT, xx:xx + CROP_WIDTH, :]
    return input_patch, demosaiced, gt_patch


def flip(input_patch, demosaiced, gt_patch):
    # inputs must be in a form of [batch_num, frame_num, height, width, channel_num]
    if np.random.randint(2, size=1)[0] == 1:  # random flip
        input_patch = np.flip(input_patch, axis=1)
        demosaiced = np.flip(demosaiced, axis=1)
        gt_patch = np.flip(gt_patch, axis=1)
    if np.random.randint(2, size=1)[0] == 1:
        input_patch = np.flip(input_patch, axis=2)
        demosaiced = np.flip(demosaiced, axis=2)
        gt_patch = np.flip(gt_patch, axis=2)
    if np.random.randint(2, size=1)[0] == 1:
        input_patch = np.flip(input_patch, axis=3)
        demosaiced = np.flip(demosaiced, axis=3)
        gt_patch = np.flip(gt_patch, axis=3)
    if np.random.randint(2, size=1)[0] == 1:  # random transpose
        input_patch = np.transpose(input_patch, (0, 1, 3, 2, 4))
        demosaiced = np.transpose(demosaiced, (0, 1, 3, 2, 4))
        gt_patch = np.transpose(gt_patch, (0, 1, 3, 2, 4))
    return input_patch, demosaiced, gt_patch


def validate(in_path, gt_path, sess, G_loss, out_image, in_image, gt_image, mask_low, mask_high, num_low, num_high):
    read_in = LOAD_TRAIN_FUNC(in_path)
    raw = np.expand_dims(unpack(read_in) / 65535.0, axis=0)
    demosaiced = (demosaic(raw) * 65535.0).astype('uint16')

    gt_raw = np.expand_dims(np.float32(LOAD_GT_FUNC(gt_path) / 255.0), axis=0)
    B, F, H, W, C = raw.shape

    input_patch, demosaiced, gt_patch = crop(raw, demosaiced, gt_raw, H, W, np.random.randint(ALL_FRAME - CROP_FRAME))

    input_patch, demosaiced, gt_patch = flip(input_patch, demosaiced, gt_patch)
    input_patch = np.minimum(input_patch, 1.0)
    mask_l, mask_h, num_l, num_h = get_low_light_area(demosaiced)
    loss, output = sess.run([G_loss, out_image], feed_dict={in_image: input_patch, gt_image: gt_patch, mask_low: mask_l, mask_high: mask_h, num_low: num_l, num_high: num_h})
    return loss



def get_low_light_area(input_video):
    # input should be of shape [batch_size, frame_count, height, width, 3]
    R, G, B = input_video[:, :, :, :, 0], input_video[:, :, :, :, 1], input_video[:, :, :, :, 2]
    luminance = 0.2126 * R + 0.7152 * G + 0.0722 * B  # refer to https://en.wikipedia.org/wiki/Relative_luminance

    # eps = 1e-5
    low_light_portion = 0.4
    mask_l = []
    mask_h = []
    num_l = []
    num_h = []
    for i in range(input_video.shape[0]):
        video = luminance[i, :, :, :]
        eps = 2.0 / video.size
        high = video.max()
        low = video.min()
        thresh = (low + high) / 2
        portion_l = len(video[np.where(video < thresh)]) / float(video.size)
        while abs(portion_l - low_light_portion) > eps and (high - low) > eps:
            if portion_l > low_light_portion:
                high = thresh
            else:
                low = thresh
            thresh = (low + high) / 2
            portion_l = len(video[np.where(video < thresh)]) / float(video.size)
        mask_l.append(np.expand_dims(video < thresh, axis=0).astype('float32'))
        num_l.append(len(video[np.where(video < thresh)]))
        mask_h.append(np.expand_dims(video >= thresh, axis=0).astype('float32'))
        num_h.append(len(video[np.where(video >= thresh)]))
    mask_l = np.concatenate(mask_l)
    mask_h = np.concatenate(mask_h)
    num_l = np.array(num_l, dtype='float32')
    num_h = np.array(num_h, dtype='float32')

    return mask_l, mask_h, num_l, num_h


def get_writer(base_log_dir):
    log_dir = base_log_dir
    duplicate = 0
    while os.path.isdir(log_dir):
        log_dir = base_log_dir + str(duplicate)
        duplicate += 1
    writer = tf.summary.FileWriter(log_dir)# , graph=tf.get_default_graph())
    return writer


def main():
    sess = tf.Session()
    in_image = tf.placeholder(tf.float32, [None, CROP_FRAME, CROP_HEIGHT, CROP_WIDTH, 1])
    gt_image = tf.placeholder(tf.float32, [None, CROP_FRAME, CROP_HEIGHT, CROP_WIDTH, 3])
    out_image = network(in_image)
    print out_image.shape

    # return

    # loss function
    # G_loss = tf.reduce_mean(tf.abs(out_image - gt_image)) + tf.reduce_mean(tf.image.ssim(out_image, gt_image, 1.0))
    structure_loss = tf.reduce_mean(tf.image.ssim_multiscale(out_image, gt_image, 1.0)) + tf.reduce_mean(tf.image.ssim(out_image, gt_image, 1.0))

    vgg_gt = vgg19.Vgg19()
    with tf.name_scope("content_vgg_gt"):
        vgg_gt.build(gt_image[0,:,:,:,:])
    fm1 = vgg_gt.conv3_4

    vgg_out = vgg19.Vgg19()
    with tf.name_scope("content_vgg_out"):
        vgg_out.build(out_image[0,:,:,:,:])
    fm2 = vgg_out.conv3_4
    context_loss = tf.reduce_mean(tf.norm(fm1 - fm2))

    mask_low = tf.placeholder(tf.float32, [None, None, None, None])
    mask_high = tf.placeholder(tf.float32, [None, None, None, None])
    num_low = tf.placeholder(tf.float32, [None])
    num_high = tf.placeholder(tf.float32, [None])
    diff = tf.norm(out_image - gt_image, axis = -1)
    region_loss = tf.reduce_mean(WEIGHT_LOW * tf.reduce_sum(diff * mask_low, [1, 2, 3]) / num_low + WEIGHT_HIGH * tf.reduce_sum(diff * mask_high, [1, 2, 3]) / num_high)

    G_loss = 2 - structure_loss + context_loss_weight * context_loss + region_loss
    
    v_loss = tf.placeholder(tf.float32)

    t_vars = tf.trainable_variables()
    lr = tf.placeholder(tf.float32)
    G_opt = tf.train.AdamOptimizer(learning_rate=lr).minimize(G_loss)
    
    loss_scalar = tf.summary.scalar('loss', v_loss)
    learning_rate_scalar = tf.summary.scalar('learning_rate', lr)
    
    base_log_dir = './logs/train_loss'
    writer_train_loss = get_writer(base_log_dir)
    base_log_dir = './logs/structure_loss'
    writer_structure_loss = get_writer(base_log_dir)
    base_log_dir = './logs/context_loss'
    writer_context_loss = get_writer(base_log_dir)
    base_log_dir = './logs/region_loss'
    writer_region_loss = get_writer(base_log_dir)
    base_log_dir = './logs/val'
    writer_val = get_writer(base_log_dir)
    base_log_dir = './logs/lr'
    writer_lr = get_writer(base_log_dir)
    count = 0

    saver = tf.train.Saver()
    sess.run(tf.global_variables_initializer())
    ckpt = tf.train.get_checkpoint_state(CHECKPOINT_DIR)
    if ckpt:
        print('loaded ' + ckpt.model_checkpoint_path)
        saver.restore(sess, ckpt.model_checkpoint_path)

    # Raw data takes long time to load. Keep them in memory after loaded.
    gt_images = [None] * len(train_ids)
    input_images = [None] * len(train_ids)

    g_loss = np.zeros((len(train_ids), 1))

    lastepoch = 0
    if not os.path.isdir(RESULT_DIR):
        os.makedirs(RESULT_DIR)
    else:
        all_items = glob.glob(os.path.join(RESULT_DIR, '*'))
        all_folders = [os.path.basename(d) for d in all_items if os.path.isdir(d) and os.path.basename(d).isdigit()]
        for folder in all_folders:
            lastepoch = np.maximum(lastepoch, int(folder))

    learning_rate = INIT_LR

    np.random.seed(ord('c') + 137)
    losses = [0] * len(valid_in_files)
    for epoch in range(lastepoch + 1, MAX_EPOCH + 1):
        e_st = time.time()
        if epoch % SAVE_FREQ == 0:
            save_results = True
            if not os.path.isdir(RESULT_DIR + '%04d' % epoch):
                os.makedirs(RESULT_DIR + '%04d' % epoch)
        else:
            save_results = False
        cnt = 0
        bt = 0
        learning_rate *= DECAY_RATE
        print '[INFO] learning rate:', learning_rate

        N = len(train_ids)
        all_order = np.random.permutation(N)
        last_group = (N // GROUP_NUM) * GROUP_NUM
        split_order = np.split(all_order[:last_group], (N // GROUP_NUM))
        split_order.append(all_order[last_group:])
        for order in split_order:
            gt_images = [None] * len(train_ids)
            input_images = [None] * len(train_ids)
            demosaiced_images = [None] * len(train_ids)
            order_frame = [(one, y) for y in [t for t in np.random.permutation(ALL_FRAME - CROP_FRAME) if t % FRAME_FREQ == 0] for one in order]

            index = np.random.permutation(len(order_frame))
            for idx in index:
                # get the path from image id
                ind, start_frame = order_frame[idx]
                start_frame += np.random.randint(FRAME_FREQ)
                if start_frame + CROP_FRAME > ALL_FRAME:
                    start_frame = ALL_FRAME - CROP_FRAME
                
                train_id = train_ids[ind] + '_start_frame_' + str(start_frame)
                in_path = in_files[ind]

                gt_path = gt_files[ind]

                st = time.time()
                cnt += 1
                
                if input_images[ind] is None:
                    try:
                        read_in = LOAD_TRAIN_FUNC(in_path)
                        input_images[ind] = np.expand_dims(unpack(read_in) / 65535.0, axis=0)
                    except MemoryError as e:
                        print(e)
                        print(train_id, in_path)
                        print('!!!train')
                        continue
                raw = input_images[ind]
                if demosaiced_images[ind] is None:
                    try:
                        demosaiced_images[ind] = (demosaic(input_images[ind]) * 65535.0).astype('uint16')
                    except MemoryError as e:
                        print(e)
                        print(train_id, in_path)
                        print('!!!demosaic')
                        continue
                demosaiced = demosaiced_images[ind]
                # raw = np.expand_dims(raw / 65535.0, axis=0)

                if gt_images[ind] is None:
                    try:
                        gt_images[ind] = np.expand_dims(np.float32(LOAD_GT_FUNC(gt_path) / 255.0), axis=0)
                    except MemoryError as e:
                        print(e)
                        print(train_id, gt_path)
                        print('!!!gt')
                        continue
                gt_raw = gt_images[ind]
                # gt_raw = np.expand_dims(np.float32(gt_raw / 255.0), axis=0)
                B, F, H, W, C = raw.shape

                input_patch, demosaiced, gt_patch = crop(raw, demosaiced, gt_raw, H, W, start_frame)

                input_patch, demosaiced, gt_patch = flip(input_patch, demosaiced, gt_patch)
                mask_l, mask_h, num_l, num_h = get_low_light_area(demosaiced)
                input_patch = np.minimum(input_patch, 1.0)

                _, G_current, output, sl, cl, rl = sess.run([G_opt, G_loss, out_image, structure_loss, context_loss, region_loss],
                                        feed_dict={in_image: input_patch,
                                                   gt_image: gt_patch,
                                                   mask_low: mask_l,
                                                   mask_high: mask_h,
                                                   num_low: num_l,
                                                   num_high: num_h,
                                                   lr: learning_rate})
                output = np.minimum(np.maximum(output, 0), 1)
                g_loss[ind] = G_current
                summary_loss = sess.run(loss_scalar, feed_dict={v_loss: G_current})
                writer_train_loss.add_summary(summary_loss, count)
                summary_structure_loss = sess.run(loss_scalar, feed_dict={v_loss: (2 - sl)})
                writer_structure_loss.add_summary(summary_structure_loss, count)
                summary_context_loss = sess.run(loss_scalar, feed_dict={v_loss: (context_loss_weight * cl)})
                writer_context_loss.add_summary(summary_context_loss, count)
                summary_region_loss = sess.run(loss_scalar, feed_dict={v_loss: rl})
                writer_region_loss.add_summary(summary_region_loss, count)
                count += 1
                
                if save_results and start_frame in SAVE_FRAMES:
                    temp = np.concatenate((gt_patch[0, :, ::-1, :, :], output[0, :, ::-1, :, :]), axis=2)
                    try:
                        vwrite((RESULT_DIR + '%04d/%s_train.avi' % (epoch, train_id)), (temp * 255).astype('uint8'))
                    except OSError as e:
                        print('\t', e, 'Skip saving.')

                print("%d %d Loss=%.8f Time=%.3f (avg:%.3f)" % (epoch, cnt, np.mean(g_loss[np.where(g_loss)]), time.time() - st, (time.time() - e_st) / cnt)), train_id
                
        # validation after each epoch
        v_start = time.time()
        for i in range(len(valid_in_files)):
            in_path = valid_in_files[i]
            gt_path = valid_gt_files[i]
            loss = validate(in_path, gt_path, sess, G_loss, out_image, in_image, gt_image, mask_low, mask_high, num_low, num_high)
            if DEBUG:
                print loss
            losses[i] = loss
        summary_lr, summary_val = sess.run([learning_rate_scalar, loss_scalar], 
                           feed_dict={lr: learning_rate, v_loss: np.mean(losses)})
        writer_val.add_summary(summary_val, count)
        writer_lr.add_summary(summary_lr, count)
        
        print 'validation: Loss={:.8f} Time={:.3f}s'.format(np.mean(losses), time.time() - v_start)

        saver.save(sess, CHECKPOINT_DIR + 'model.ckpt')
        if save_results:
            saver.save(sess, RESULT_DIR + '%04d/' % epoch + 'model.ckpt')


if __name__ == '__main__':
    main()
