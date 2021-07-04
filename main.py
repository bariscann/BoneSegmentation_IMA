from bone_data import BoneData
from bone_dataset import BoneDataset
from segmentation import kfold_xnet_test
from tensorflow import keras

keras.backend.clear_session()


bone_data = BoneData()


img_size = (160, 160)
# num_classes = 2
# batch_size = 8
# epochs = 2 # to test segmentation immediatly
all_gen = BoneDataset(batch_size=1,
                      img_size=img_size,
                      input_img_paths=bone_data.all_data_files,
                      target_img_paths=bone_data.all_label_files)


metric_names_list, scores_list = kfold_xnet_test(img_size=img_size, 
                                                    all_gen=all_gen,
                                                    bone_data=bone_data,
                                                    batch_size=1,
                                                    epochs=50,
                                                    is_callbacks_enable=True)

print(metric_names_list)
print(scores_list)