import argparse
import json
import os
from collections import defaultdict
import numpy as np
import math
from torch.functional import F
import torch


def softmax(scores):
    max = np.max(scores)
    stable_x = np.exp(scores - max)
    prob = stable_x / np.sum(stable_x)
    return prob


def sigmoid(x):
    x = np.array(x)
    return 1.0 / (1.0 + np.exp(-1.0 * x))


def print_action(rgb_score):
    rgb_score = sigmoid(np.array(rgb_score))
    print("RGB action {0} with probability {1}".format(
        np.argmax(rgb_score), np.max(rgb_score)))


parser = argparse.ArgumentParser()
parser.add_argument('--src_dir', type=str)
parser.add_argument('--dst_dir', type=str)
parser.add_argument('--temp_scaling', type=float, default=1.0)

args = parser.parse_args()
src_dir = args.src_dir
dst_dir = args.dst_dir
temp_scaling = args.temp_scaling
temp_tensor = torch.FloatTensor([temp_scaling])
print("Temperature tensor: ", temp_tensor)
filelist = os.listdir(src_dir)
for file in filelist:
    src_file = os.path.join(src_dir, file)

    try:
        with open(src_file) as json_data:
            rgb_scores = json.load(json_data)
    except:
        print("Exception")
        continue

    print("Video", file)
    rgb_window_scores = []
    output_dict = defaultdict(lambda: defaultdict(list))
    for key in rgb_scores:
        score_tensor = torch.FloatTensor(rgb_scores[key]["rgb_scores"])
        score_tensor = score_tensor / temp_tensor
        rgb_score = F.softmax(score_tensor, dim=-1)
        rgb_score = rgb_score.cpu().detach().numpy().flatten()

        output_dict[key]["rgb_scores"] = rgb_score.tolist()

    with open(dst_dir + "/" + file, 'w') as outfile:
        json.dump(output_dict, outfile)
