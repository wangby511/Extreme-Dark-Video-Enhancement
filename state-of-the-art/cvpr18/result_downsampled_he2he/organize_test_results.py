import glob, os
from skvideo.io import vwrite
import cv2
import numpy as np

DIR = 'test_videos'
if not os.path.isdir(DIR):
    os.mkdir(DIR)

files = glob.glob('./test/*')

videos = {}

for f in files:
    key = f.split('.')[1][:-3]
    if key in videos.keys():
        videos[key].append(f)
    else:
        videos[key] = [f]

for key in videos.keys():
    all_frames = videos[key]
    #print all_frames
    assert len(all_frames) == 200
    all_frames.sort()
    img = cv2.imread(all_frames[0])
    new_vid = np.zeros([200, img.shape[0], img.shape[1], 3], 'uint8')
    for i, f in enumerate(all_frames):
        new_vid[i] = cv2.imread(f)[:, :, (2, 1, 0)]
    vwrite(DIR + '/' + os.path.basename(all_frames[0].strip('_000.npy_out')) + '.mp4', new_vid)
