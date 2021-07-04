import os
from keras.initializers import LOCAL
import numpy as np

class BoneData:
    # SHAPE = (200, 200)
    COLAB = 0
    LOCAL_PC = 1
    DATA_PATHS = ["/content/drive/MyDrive",
                  "data"]
    
    def __collect_all_label_files(self):
        sub_folders = os.listdir(self.mask_path)
        all_files = {}
        for sub_folder in sub_folders:
            sub_folderpath = "{}/{}".format(self.mask_path, sub_folder)
            # print(sub_folderpath)
            class_folders = os.listdir(sub_folderpath)
            if sub_folder not in all_files:
                all_files[sub_folder] = []

            for class_folder in class_folders:
                class_folderpath = "{}/{}".format(sub_folderpath, class_folder)
                file_names = os.listdir(class_folderpath)
                
                for file_name in file_names:
                    filepath = "{}/{}".format(class_folderpath, file_name)
                    all_files[sub_folder].append(filepath)
        return self.__dict_to_list(all_files)
    
    def __collect_all_image_files(self):
        sub_folders = os.listdir(self.image_path)
        all_files = {}
        for sub_folder in sub_folders:
            sub_folderpath = "{}/{}".format(self.image_path, sub_folder)
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
        return self.__dict_to_list(all_files)
    
    def __dict_to_list(self, data_asdict):
        num_image = len(data_asdict.keys())
        data_aslist = []
        for i in range(1, num_image+1):
            data = data_asdict[str(i)]
            data_aslist.append(data)
        return np.array(data_aslist)
    
    def __init__(self, work_area=LOCAL_PC) -> None:
        self.mask_path = "{}/New_masks".format(self.DATA_PATHS[work_area])
        self.image_path = "{}/New_Labels".format(self.DATA_PATHS[work_area])
        self.all_label_files = self.__collect_all_label_files()      
        self.all_data_files = self.__collect_all_image_files()
