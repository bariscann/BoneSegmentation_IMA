import os
import numpy as np

class BoneData:
    
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
                    if not file_name.endswith(".json"):
                        filepath = "{}/{}".format(sub_folderpath, file_name)
                        all_files[sub_folder] = filepath
        return self.__dict_to_list(all_files)
    
    def __load_labels(self):
        image_label_data = {}
        for image_folder in self.all_label_files:
            image_label_data[image_folder] = None
            for  class_folder in self.all_label_files[image_folder]:
                for file_path in self.all_label_files[image_folder][class_folder]:
                    image_label_data[image_folder] = self.__merge_label(image_label_data[image_folder], np.load(file_path))
        return self.__dict_to_list(image_label_data)
    
    
    @staticmethod
    def __dict_to_list(data_asdict):
        num_image = len(data_asdict.keys())
        data_aslist = []
        for i in range(1, num_image+1):
            data_aslist.append(data_asdict[str(i)])
        return data_aslist
            
            
        
    
    def __init__(self) -> None:
        self.all_label_files = self.__collect_all_label_files()
        self.image_label_data = self.__load_labels()
        self.image_data = self.__collect_all_image_files()
            
    
    