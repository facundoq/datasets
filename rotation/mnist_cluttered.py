import os
import numpy as np
import h5py
from fuel.converters.base import fill_hdf5_file
from scipy.io import loadmat
from os import listdir
from os.path import isfile, join
from PIL import Image
import shutil
from argparse import ArgumentParser


def main(path):
    train_features = []
    train_locations = []
    train_labels = []
    test_features = []
    test_locations = []
    test_labels = []
    for f in listdir('images'):
        if isfile(join('images', f)):
            number, label, x, y = f.split('.')[0].split('_')
            location = np.array((0.28, 0, (int(x) + 14.0 - 50.0) / 50.0, 0, 0.28, (int(y) + 14.0 - 50.0) / 50.0), ndmin=1, dtype=np.float32)
            image = np.array(Image.open(join('images', f)), ndmin=3, dtype=np.uint8)
            label = int(label)
            if int(number) <= 60000:
                train_features.append(image)
                train_locations.append(location)
                train_labels.append(label)
            else:
                test_features.append(image)
                test_locations.append(location)
                test_labels.append(label)

    h5file = h5py.File(path, mode='w')

    data = (
            ('train', 'features', np.array(train_features)),
            ('test', 'features', np.array(test_features)),
            ('train', 'locations', np.array(train_locations)),
            ('test', 'locations', np.array(test_locations)),
            ('train', 'labels', np.array(train_labels, dtype=np.uint8)),
            ('test', 'labels', np.array(test_labels, dtype=np.uint8)),
    )
    fill_hdf5_file(h5file, data)
    for i, label in enumerate(('batch', 'channel', 'height', 'width')):
        h5file['features'].dims[i].label = label
    for i, label in enumerate(('batch', 'index')):
        h5file['locations'].dims[i].label = label
    for i, label in enumerate(('batch',)):
        h5file['labels'].dims[i].label = label

    h5file.flush()
    h5file.close()

    shutil.rmtree('images')



if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument("--path", type=str, dest="path",
                        default='../datasets/mnist_cluttered_test.hdf5', help="Path to dataset file")
    args = parser.parse_args()
    main(**vars(args))
