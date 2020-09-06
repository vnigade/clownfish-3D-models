from __future__ import print_function, division
import os
import sys
import json
import pandas as pd


def convert_csv_to_dict(csv_path, idx_to_class, subset):
    data = pd.read_csv(csv_path, delimiter=',', header=None)
    data = data.sort_values(by=1)

    database = {}
    file_name, file_ext = os.path.splitext(os.path.basename(csv_path))
    for i in range(data.shape[0]):
        # row = data.ix[i, :]
        row = data.loc[i, :]
        key = file_name + '-clip-' + str(i)
        # Actual index starts from 1 but we need from 0.
        key_label = int(row[0]) - 1
        start_frame = int(row[1])
        end_frame = int(row[2])
        confidence = int(row[3])
        database[key] = {}
        database[key]['subset'] = subset
        database[key]['annotations'] = {
            'label': key_label, 'label_name': idx_to_class[key_label],
            'start_frame': start_frame,
            'end_frame': end_frame, 'confidence': confidence}

    return database


def load_labels(label_csv_path):
    labels = []

    data = pd.read_csv(label_csv_path, delimiter=',', header=None)
    file_name, file_ext = os.path.splitext(os.path.basename(label_csv_path))
    for i in range(data.shape[0]):
        # We treat each action instance as a separate clip
        labels.append(file_name + '-clip-'+str(i))
    return labels


def load_class_names(path):
    df = pd.read_excel(path)
    index_list = df['Label']
    names_list = df['Action']
    idx_to_class = {}
    for i in df.index:
        ind = int(index_list[i]) - 1  # We need label from 0 not from 1
        name = str(names_list[i])
        idx_to_class[ind] = name

    return idx_to_class


def convert_pkummd_csv_to_json(label_csv_path, train_videos, validation_videos, idx_to_class, dst_json_path):

    labels = []
    dst_data = {}
    dst_data['database'] = {}

    for file_name in train_videos:
        file_path = os.path.join(label_csv_path, file_name + '.txt')
        if os.path.exists(file_path):
            labels.extend(load_labels(file_path))
            database = convert_csv_to_dict(
                file_path, idx_to_class, 'training')
            dst_data['database'].update(database)

    for file_name in validation_videos:
        file_path = os.path.join(label_csv_path, file_name + '.txt')
        if os.path.exists(file_path):
            labels.extend(load_labels(file_path))
            database = convert_csv_to_dict(
                file_path, idx_to_class, 'validation')
            dst_data['database'].update(database)

    dst_data['labels'] = labels

    with open(dst_json_path, 'w') as dst_file:
        json.dump(dst_data, dst_file)


def get_splits_from_file(split_file):
    if not os.path.exists(split_file):
        print("Split file path does not exist")
        return

    data = pd.read_csv(split_file, delimiter=',', header=None)
    train = []
    validation = []
    for i in range(data.shape[0]):
        # row = data.ix[i, :]
        row = data.loc[i, :]
        if row[0] == 'training_videos':
            train.extend(row[1:])
        elif row[0] == 'validation_videos':
            video_list = [video for video in row[1:] if str(video) != 'nan']
            validation.extend(video_list)

    train = [video.strip() for video in train]
    validation = [video.strip() for video in validation]
    return train, validation


def get_default_splits():
    SPLIT_TYPE = 'M'  # (L, M R)
    N_TRAIN_SPLIT = 255
    TOTAL_VIDEOS = 364

    train = []
    validation = []
    for i in range(1, TOTAL_VIDEOS + 1):
        file_format = '{:04d}-' + SPLIT_TYPE + '.txt'
        if i <= N_TRAIN_SPLIT:
            train.append(file_format.format(i))
        else:
            validation.append(file_format.format(i))
    return train, validation


if __name__ == '__main__':

    split_file = None
    if len(sys.argv) == 4:
        split_file = sys.argv[3]  # Split file
    label_csv_path = sys.argv[1]  # Directory containing per video annotations
    class_label_path = sys.argv[2]  # Excel file

    if split_file is not None:
        train_videos, validation_videos = get_splits_from_file(split_file)
    else:
        train_videos, validation_videos = get_default_splits()

    dst_json_path = os.path.join(
        label_csv_path, 'pkummd.json')

    idx_to_class = load_class_names(class_label_path)
    convert_pkummd_csv_to_json(
        label_csv_path, train_videos, validation_videos, idx_to_class, dst_json_path)
