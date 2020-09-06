'''
PKUMMD dataset handlers for siminet model
'''
import torch
import torch.utils.data as data
from PIL import Image
import os
import functools
import json
import copy
import math
import numpy as np

from utils import load_value_file
import itertools


def pil_loader(path):
    # open path as file to avoid ResourceWarning (https://github.com/python-pillow/Pillow/issues/835)
    with open(path, 'rb') as f:
        with Image.open(f) as img:
            return img.convert('RGB')


def accimage_loader(path):
    try:
        import accimage
        return accimage.Image(path)
    except IOError:
        # Potentially a decoding problem, fall back to PIL.Image
        return pil_loader(path)


def get_default_image_loader():
    from torchvision import get_image_backend
    if get_image_backend() == 'accimage':
        return accimage_loader
    else:
        return pil_loader


def video_loader(video_dir_path, frame_indices, image_loader):
    video = []
    for i in frame_indices:
        image_path = os.path.join(video_dir_path, 'image_{:05d}.jpg'.format(i))
        if os.path.exists(image_path):
            video.append(image_loader(image_path))
        else:
            return video

    return video


def get_default_video_loader():
    image_loader = get_default_image_loader()
    return functools.partial(video_loader, image_loader=image_loader)


def load_annotation_data(data_file_path):
    with open(data_file_path, 'r') as data_file:
        return json.load(data_file)


def get_class_labels(data):

    class_labels_map = {}
    for _, value in data['database'].items():
        label = int(value['annotations']['label'])
        label_name = str(value['annotations']['label_name'])
        class_labels_map[label_name] = label

    return class_labels_map


def get_video_names_and_annotations(data, subset):
    video_names = []
    annotations = []

    for key, value in data['database'].items():
        this_subset = value['subset']
        if this_subset == subset:
            if subset == 'testing':
                video_names.append('{}'.format(key))
            else:
                video_names.append('{}'.format(key))
                annotations.append(value['annotations'])

    return video_names, annotations


def get_video_from_clip(video_name):
    return video_name.split("-clip")[0]


def modify_frame_indices(video_dir_path, frame_indices):
    modified_indices = []
    for i in frame_indices:
        image_path = os.path.join(video_dir_path, 'image_{:05d}.jpg'.format(i))
        if not os.path.exists(image_path):
            return modified_indices
        modified_indices.append(i)
    return modified_indices


def get_sample(video_path, video_id, begin_t, end_t, label, n_samples_for_each_video, sample_duration):

    sample = {
        'video': video_path,
        'segment': [begin_t, end_t],
        'fps': 0,
        'video_id': video_id
    }
    assert label != -1, "Label is -1"
    n_frames = end_t - begin_t

    samples = []
    if n_samples_for_each_video == 1:
        frame_indices = list(range(begin_t, end_t))
        frame_indices = modify_frame_indices(sample['video'],
                                             frame_indices)
        if len(frame_indices) < sample_duration:
            return samples
        sample['frame_indices'] = frame_indices
        samples.append(sample)

    else:
        if n_samples_for_each_video > 1:
            step = max(1,
                       math.ceil((n_frames - 1 - sample_duration) /
                                 (n_samples_for_each_video - 1)))
        else:
            step = sample_duration
        for j in range(begin_t, end_t, step):
            sample_j = copy.deepcopy(sample)
            frame_indices = list(range(j, j + sample_duration))
            frame_indices = modify_frame_indices(
                sample_j['video'], frame_indices)
            if len(frame_indices) < sample_duration:
                continue
            sample_j['frame_indices'] = frame_indices
            samples.append(sample_j)

    return samples


def make_dataset(root_path, annotation_path, subset, n_samples_for_each_video,
                 sample_duration):
    data = load_annotation_data(annotation_path)
    video_names, annotations = get_video_names_and_annotations(data, subset)
    class_to_idx = get_class_labels(data)
    idx_to_class = {}
    for name, label in class_to_idx.items():
        idx_to_class[label] = name

    dataset = []
    for i in range(len(video_names)):
        if i != 0 and i % 1000 == 0:
            print('dataset loading [{}/{}]'.format(i, len(video_names)))
            # return dataset, idx_to_class # early exit

        video_path = os.path.join(
            root_path, get_video_from_clip(video_names[i]))
        if not os.path.exists(video_path):
            continue

        annotation = annotations[i]
        begin_t = annotation['start_frame']
        end_t = annotation['end_frame']
        if begin_t == 0:
            begin_t = 1

        # cur frame segment
        label = int(annotation['label'])
        cur_samples = get_sample(
            video_path, video_names[i], begin_t, end_t, label, n_samples_for_each_video, sample_duration)

        # prev frame segment
        prev_begin_t = begin_t - sample_duration * 2
        prev_end_t = begin_t + sample_duration // 2
        prev_samples = get_sample(video_path, video_names[i], prev_begin_t,
                                  prev_end_t, label, n_samples_for_each_video, sample_duration)

        # Next frame segment
        next_begin_t = end_t - sample_duration // 2
        next_end_t = end_t + sample_duration * 2
        next_samples = get_sample(video_path, video_names[i], next_begin_t,
                                  next_end_t, label, n_samples_for_each_video, sample_duration)

        label = 1
        sample_list = [cur_samples, cur_samples]
        sample_list = list(itertools.product(*sample_list))
        for sample_pair in sample_list:
            sample = {
                "video": video_path,
                "video_id": video_names[i],
                "sample_pair": sample_pair,
                "label": label
            }
            dataset.append(sample)

        if len(prev_samples) > 0:
            label = 0
            sample_list = [prev_samples, cur_samples]
            sample_list = list(itertools.product(*sample_list))
            for sample_pair in sample_list:
                sample = {
                    "video": video_path,
                    "video_id": video_names[i],
                    "sample_pair": sample_pair,
                    "label": label
                }
                dataset.append(sample)
        elif len(next_samples) > 0:
            # Note that, we do not consider this next frame segment for training.
            # This unintentional feature (or bug) have led to a balanced number of training samples
            # for the binary classification task.
            label = 0
            sample_list = [cur_samples, next_samples]
            sample_list = list(itertools.product(*sample_list))
            for sample_pair in sample_list:
                sample = {
                    "video": video_path,
                    "video_id": video_names[i],
                    "sample_pair": sample_pair,
                    "label": label
                }
                dataset.append(sample)

    return dataset, idx_to_class


class PKUMMD_SIM(data.Dataset):
    """
    Args:
        root (string): Root directory path.
        spatial_transform (callable, optional): A function/transform that  takes in an PIL image
            and returns a transformed version. E.g, ``transforms.RandomCrop``
        temporal_transform (callable, optional): A function/transform that  takes in a list of frame indices
            and returns a transformed version
        target_transform (callable, optional): A function/transform that takes in the
            target and transforms it.
        loader (callable, optional): A function to load an video given its path and frame indices.
     Attributes:
        classes (list): List of the class names.
        class_to_idx (dict): Dict with items (class_name, class_index).
        imgs (list): List of (image path, class_index) tuples
    """

    def __init__(self,
                 root_path,
                 annotation_path,
                 subset,
                 is_untrimmed_setting=False,
                 n_samples_for_each_video=1,
                 spatial_transform=None,
                 temporal_transform=None,
                 target_transform=None,
                 sample_duration=16,
                 window_size=None,
                 window_stride=None,
                 get_loader=get_default_video_loader,
                 scores_dump_path=None):
        self.is_untrimmed_setting = is_untrimmed_setting

        self.data, self.class_names = make_dataset(
            root_path, annotation_path, subset, n_samples_for_each_video,
            sample_duration)

        self.spatial_transform = spatial_transform
        self.temporal_transform = temporal_transform
        self.target_transform = target_transform
        self.loader = get_loader()

    def __getitem__(self, index):
        """
        Args:
            index (int): Index
        Returns:
            tuple: (image, target) where target is class_index of the target class.
        """
        path = self.data[index]['video']

        pair1 = self.data[index]['sample_pair'][0]
        pair2 = self.data[index]['sample_pair'][1]

        # Pair 1
        frame_indices = pair1['frame_indices']
        if self.temporal_transform is not None:
            frame_indices = self.temporal_transform(frame_indices)
        clip1 = self.loader(path, frame_indices)
        if self.spatial_transform is not None:
            self.spatial_transform.randomize_parameters()
            clip1 = [self.spatial_transform(img) for img in clip1]
        clip1 = torch.stack(clip1, 0).permute(1, 0, 2, 3)

        # Pair 2
        frame_indices = pair2['frame_indices']
        if self.temporal_transform is not None:
            frame_indices = self.temporal_transform(frame_indices)
        clip2 = self.loader(path, frame_indices)
        if self.spatial_transform is not None:
            self.spatial_transform.randomize_parameters()
            clip2 = [self.spatial_transform(img) for img in clip2]
        clip2 = torch.stack(clip2, 0).permute(1, 0, 2, 3)

        # Target
        target = np.asarray(self.data[index]['label'], dtype=np.float32)

        return clip1, clip2, target, index

    def __len__(self):
        return len(self.data)
