import numpy as np
import os

def connect_color_basic(path='./data/real/'):
    for file_ in os.listdir(path):
        if 'color' in file_:
            continue

        basic_data = np.load(os.path.join(path, file_))
        file_ = file_.replace('basic', 'color')
        color_data = np.load(os.path.join(path, file_))

        data = np.concatenate((basic_data, color_data), axis=2)

        file_ = file_.replace('color', '')
        np.save(os.path.join(path, file_), data)
        print(data.shape)



def main(path='./data/synth/'):
    for file_ in os.listdir(path):
        if os.path.isdir(os.path.join(path, file_)):
            continue
        data = np.load(os.path.join(path, file_))

        if data.shape[2] == 8:
            data = data[:, :, :-1]
            print(data.shape)
            np.save(os.path.join(path, file_), data)


if __name__ == '__main__':
    connect_color_basic()