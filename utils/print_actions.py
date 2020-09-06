import argparse
import json
import os
from collections import defaultdict
import numpy as np
import math


def print_action(file, key, rgb_score, start_frame):
    print("{2}-{3} RGB action {0} with probability {1}, start_frame {4}".format(
        np.argmax(rgb_score), np.max(rgb_score), file, key, start_frame))


def read_and_print(rgb_file, file):
    try:
        with open(rgb_file) as json_data:
            rgb_scores = json.load(json_data)
    except:
        print("Exception")
        return

    start_frame = 1
    for key in rgb_scores:
        print_action(file, key, rgb_scores[key]["rgb_scores"], start_frame)
        start_frame += window_stride


parser = argparse.ArgumentParser()
parser.add_argument('--rgb_dir', type=str)
parser.add_argument('--window_size', type=int)
parser.add_argument('--window_stride', type=int)
args = parser.parse_args()
rgb_dir = args.rgb_dir
window_size = args.window_size
window_stride = args.window_stride

# filelist = os.listdir(rgb_dir)
# for file in filelist:
#   rgb_file = os.path.join(rgb_dir, file)
#  read_and_print(rgb_file)
read_and_print(rgb_dir, file='')
