import os
from cv2 import data
import numpy as np
import cv2

class BoneData:
    SHAPE = (200, 200)
    # DATA_PATH = "/content/drive/MyDrive/New_masks"
    DATA_PATH = "data"
    MASK_PATH = "{}/New_masks".format(DATA_PATH)
    IMAGE_PATH = "{}/New_Labels".format(DATA_PATH)
    
    @staticmethod
    def __merge_label(base_data, new_data):
        if base_data is not None:
            base_data |= new_data
        else:
            base_data = new_data
        return base_data
            
    
    def __collect_all_label_files(self):
        sub_folders = os.listdir(self.MASK_PATH)
        all_files = {}
        for sub_folder in sub_folders:
            sub_folderpath = "{}/{}".format(self.MASK_PATH, sub_folder)
            # print(sub_folderpath)
            class_folders = os.listdir(sub_folderpath)
            if sub_folder not in all_files:
                all_files[sub_folder] = {}

            for class_folder in class_folders:
                if class_folder not in all_files[sub_folder]:
                    all_files[sub_folder][class_folder] = []
                class_folderpath = "{}/{}".format(sub_folderpath, class_folder)
                file_names = os.listdir(class_folderpath)
                
                for file_name in file_names:
                    filepath = "{}/{}".format(class_folderpath, file_name)
                    all_files[sub_folder][class_folder].append(filepath)
        return all_files
    
    
    def __collect_all_image_files(self):
        sub_folders = os.listdir(self.IMAGE_PATH)
        all_files = {}
        for sub_folder in sub_folders:
            sub_folderpath = "{}/{}".format(self.IMAGE_PATH, sub_folder)
            if os.path.isdir(sub_folderpath):
                file_names = os.listdir(sub_folderpath)
                
                for file_name in file_names:
                    if (file_name.endswith(".jpg") 
                        or file_name.endswith(".png") 
                        or file_name.endswith(".PNG") 
                        or file_name.endswith(".jpeg")
                        or file_name.endswith(".JPG")
                        or file_name.endswith(".JPEG")
                        or file_name.endswith(".gif")):
                        filepath = "{}/{}".format(sub_folderpath, file_name)
                        all_files[sub_folder] = filepath
        return all_files
    
    @staticmethod
    def __load_image(image_path):
        if image_path.endswith(".gif"):
            cap = cv2.VideoCapture(image_path)
            ret, data = cap.read()
        else:
            data = cv2.imread(image_path)
        return data
    
    def __load_images(self):
        images_data = {}
        for image_folder in self.all_data_files:
            file_path = self.all_data_files[image_folder]
            images_data[image_folder] = self.__load_image(image_path=file_path)
            # if images_data[image_folder] is None:
            #     print(file_path)
        data = self.__dict_to_list(images_data)
        shape = (data.shape[0], ) + BoneData.SHAPE + (3,)
        return data.reshape(shape)
        
    
    def __load_labels(self):
        
        image_label_data = {}
        for image_folder in self.all_label_files:
            image_label_data[image_folder] = None
            for  class_folder in self.all_label_files[image_folder]:
                for file_path in self.all_label_files[image_folder][class_folder]:
                    image_label_data[image_folder] = self.__merge_label(image_label_data[image_folder], np.load(file_path))
        # data = np.expand_dims(self.__dict_to_list(image_label_data), 2)
        data = self.__dict_to_list(image_label_data)
        shape = (data.shape[0], ) + BoneData.SHAPE + (1,)
        return data.reshape(shape)
    
    
    def __dict_to_list(self, data_asdict):
        num_image = len(data_asdict.keys())
        data_aslist = []
        for i in range(1, num_image+1):
            data = data_asdict[str(i)]
            data = self.__reshape(data=data)
            data_aslist.append(data)
        return np.array(data_aslist)
            
    def __reshape(self, data):
        try:
            r_data = cv2.resize(data, self.SHAPE)
            return r_data
        except Exception as e:
            print(e)
            return data
    
    def generate_data(self):
        return self.images_data, self.labels_data
    
    def __init__(self) -> None:
        self.all_label_files = self.__collect_all_label_files()
        self.labels_data = self.__load_labels()
        
        self.all_data_files = self.__collect_all_image_files()
        self.images_data = self.__load_images()
        
        self.data = self.generate_data()
        
bone_data = BoneData()
# from keras_model import get_model
# import tensorflow as tf
# from tensorflow import keras
# img_size = (200, 200)
# num_classes = 2
# with tf.device("cpu"):
#     # Free up RAM in case the model definition cells were run multiple times
#     keras.backend.clear_session()
#     model = get_model(img_size, num_classes)
    
# model.compile(optimizer="rmsprop", loss="sparse_categorical_crossentropy")
# # Train the model, doing validation at the end of each epoch.
# epochs = 2
# model.fit(bone_data.data, validation_data=bone_data.data, epochs=epochs)