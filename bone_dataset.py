import cv2
# import tensorflow as tf
from tensorflow import keras
import numpy as np
# tf.config.set_visible_devices([], 'GPU')
# tf.device("cpu")


def load_image(image_path):
    if image_path.endswith(".gif"):
        cap = cv2.VideoCapture(image_path)
        ret, data = cap.read()
    else:
        data = cv2.imread(image_path)
    return data

def reshape(data, shape):
    try:
        r_data = cv2.resize(data, shape)
        return r_data
    except Exception as e:
        print(e)
        exit(11)
        
def merge_label(base_data, new_data):
        if base_data is not None:
            base_data |= new_data
        else:
            base_data = new_data
        return base_data
        
def load_label(path_list):
    data = None
    for file_path in path_list:
        data = merge_label(data, np.load(file_path))
    return data

class BoneDataset(keras.utils.Sequence):
    """Helper to iterate over the data (as Numpy arrays)."""

    def __init__(self, batch_size, img_size, input_img_paths, target_img_paths):
        self.batch_size = batch_size
        self.img_size = img_size
        self.input_img_paths = input_img_paths
        self.target_img_paths = target_img_paths

    def __len__(self):
        return len(self.target_img_paths) // self.batch_size

    # x dogrudan calisacak calismamasi icin bir sakinca yok
    # y biraz daha farkli bizim senaryoda
    def __getitem__(self, idx):
        """Returns tuple (input, target) correspond to batch #idx."""
        i = idx * self.batch_size
        batch_input_img_paths = self.input_img_paths[i : i + self.batch_size]
        batch_target_img_paths = self.target_img_paths[i : i + self.batch_size]
        x = np.zeros((self.batch_size,) + self.img_size + (3,), dtype="float32")
        for j, path in enumerate(batch_input_img_paths):
            img = load_image(image_path=path)
            img = reshape(data=img, shape=self.img_size)
            x[j] = img
        
        
        y = np.zeros((self.batch_size,) + self.img_size + (1,), dtype="uint8")
        for j, path_list in enumerate(batch_target_img_paths):
            img = load_label(path_list=path_list)
            img = reshape(data=img, shape=self.img_size)
            y[j] = np.expand_dims(img, 2)

        return x, y
