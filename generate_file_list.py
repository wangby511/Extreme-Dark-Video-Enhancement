import glob, os
import numpy as np

files = os.listdir('./0_data/train_he/')
files.sort()

assert len(files) == 179

np.random.seed(ord('c') + 137)
index = np.random.permutation(179)

with open('file_list', 'w') as f:
    for i in index[:125]:
        file = files[i]
        name = file[:-4]
        in_path = './0_data/train_he/' + file
        gt_path = './0_data/gt_he/' + file
        f.write(name + ' ' + gt_path + ' ' + in_path + '\n')

with open('valid_list', 'w') as f:
    for i in index[125:152]:
        file = files[i]
        name = file[:-4]
        in_path = './0_data/train_he/' + file
        gt_path = './0_data/gt_he/' + file
        f.write(name + ' ' + gt_path + ' ' + in_path + '\n')

with open('test_list', 'w') as f:
    for i in index[152:]:
        file = files[i]
        name = file[:-4]
        in_path = './0_data/train_he/' + file
        gt_path = './0_data/gt_he/' + file
        f.write(name + ' ' + gt_path + ' ' + in_path + '\n')
