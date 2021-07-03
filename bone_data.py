import os

class BoneData:
    SHAPE = (200, 200)
    # DATA_PATH = "/content/drive/MyDrive/New_masks"
    DATA_PATH = "data"
    MASK_PATH = "{}/New_masks".format(DATA_PATH)
    IMAGE_PATH = "{}/New_Labels".format(DATA_PATH)

    def __collect_all_label_files(self):
        sub_folders = os.listdir(self.MASK_PATH)
        all_files = {}
        for sub_folder in sub_folders:
            sub_folderpath = "{}/{}".format(self.MASK_PATH, sub_folder)
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
        return self.__dict_to_list(all_files)
    
    def __dict_to_list(self, data_asdict):
        num_image = len(data_asdict.keys())
        data_aslist = []
        for i in range(1, num_image+1):
            data = data_asdict[str(i)]
            data_aslist.append(data)
        return data_aslist
    
    def __init__(self) -> None:
        self.all_label_files = self.__collect_all_label_files()      
        self.all_data_files = self.__collect_all_image_files()
