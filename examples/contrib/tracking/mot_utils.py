import torch
import numpy as np
import warnings
import csv
import os


def read_mot(filename, zero_based=True):
    '''
    format is stated be to  be 10 columns:
    <frame>, <id>, <bb_left>, <bb_top>, <bb_width>, <bb_height>, <conf>, <x>, <y>, <z>
    However, gt files have 9 columns, therefore we are going to only look at first 7 columns.
    '''
    with open(filename, encoding='utf-8') as data_file:
        reader = csv.reader(data_file)
        data = torch.tensor([list(map(float, row)) for row in reader], dtype=torch.float)
    frames = torch.tensor(data[:, 0], requires_grad=False, dtype=torch.long)
    object_id = torch.tensor(data[:, 1], requires_grad=False, dtype=torch.long)
    positions = torch.tensor(data[:, 2:4], requires_grad=False, dtype=torch.float)
    sizes = torch.tensor(data[:, 4:6], requires_grad=False, dtype=torch.float)
    features = {'confidence': torch.tensor(data[:, 6], requires_grad=False, dtype=torch.float)}
    if zero_based:
        frames -= 1
        object_id[object_id > 0] -= 1
        positions -= 1
    return frames, object_id, positions, sizes, features


def write_mot(filename, frames, object_id, positions, sizes, confidence, zero_based=True):
    '''
    format is stated be to  be 10 columns:
    <frame>, <id>, <bb_left>, <bb_top>, <bb_width>, <bb_height>, <conf>, <x>, <y>, <z>
    However, gt files have 9 columns, therefore we are going to only look at first 7 columns.
    '''
    num_rows = frames.shape[0]
    assert frames.dim() == 1
    assert object_id.dim() == 1 and object_id.shape[0] == num_rows
    assert positions.dim() == 2 and positions.shape[0] == num_rows
    assert sizes.dim() == 2 and sizes.shape[0] == num_rows
    assert confidence.dim() == 1 and confidence.shape[0] == num_rows
    if zero_based:
        frames += 1
        object_id += 1
        positions += 1
    combined = torch.cat([frames.unsqueeze(-1).float(),
                          object_id.unsqueeze(-1).float(),
                          positions.float(),
                          sizes.float(),
                          confidence.unsqueeze(-1).float(),
                          -1 * torch.ones(num_rows, 3, dtype=torch.float)
                          ], 1)
    _, indx, counts = np.unique(combined.detach().numpy()[:, :2], axis=0, return_index=True, return_counts=True)
    if np.any(counts != 1):
        warnings.warn("Duplicates entries found, removing...")
        combined = combined[torch.from_numpy(indx)]
    combined = combined.tolist()
    with open(filename, 'w', newline='') as data_file:
        writer = csv.writer(data_file)

        def fmt(x):
            return int(x) if x.is_integer() else x
        for i in range(len(combined)):
            writer.writerow(map(fmt, combined[i]))


def test_read_mot():
    frames, object_id, positions, sizes, features = read_mot('testfile-gt.txt')
    num_rows = frames.shape[0]
    assert frames.dim() == 1
    assert object_id.dim() == 1 and object_id.shape[0] == num_rows
    assert positions.dim() == 2 and positions.shape[0] == num_rows
    assert sizes.dim() == 2 and sizes.shape[0] == num_rows
    assert features['confidence'].dim() == 1 and features['confidence'].shape[0] == num_rows


def test_write_mot():
    frames, object_id, positions, sizes, features = read_mot('testfile-gt.txt')
    write_mot('test-testfile-gt.txt', frames, object_id, positions, sizes, features['confidence'])
    test_frames, test_object_id, test_positions, test_sizes, test_features = read_mot('test-testfile-gt.txt')
    assert (test_frames == frames).all()
    assert (test_object_id == object_id).all()
    assert (test_positions == positions).all()
    assert (test_sizes == sizes).all()
    assert (test_features['confidence'] == features['confidence']).all()


def setup_function(self):
    ex = [[69, -1, 912.8, 482.9, 97.6, 112.6, 1],
          [69, -1, 835.8, 472.2, 53.7, 77, 1],
          [69, -1, 374.4, 447.1, 43.4, 105.1, 1],
          [69, -1, 1261, 447.7, 34.9, 100.6, 1],
          [69, -1, 419, 458.4, 42.7, 85.3, 1],
          [69, -1, 501.4, 440.4, 129.3, 324, 1],
          [69, -1, 1543.7, 429.8, 52, 127.2, 1],
          [69, -1, 1088.7, 482.7, 35.6, 117.1, 1],
          [69, -1, 1004.6, 441.5, 42.1, 112, 1]]
    files = os.listdir('.')
    if 'testfile-gt.txt' in files:
        os.remove('testfile-gt.txt')
    with open('testfile-gt.txt', 'w', newline='') as data_file:
        writer = csv.writer(data_file)
        writer.writerows(ex)


def teardown_function(self):
    files = os.listdir('.')
    if 'testfile-gt.txt' in files:
        os.remove('testfile-gt.txt')
    if 'test-testfile-gt.txt' in files:
        os.remove('test-testfile-gt.txt')
