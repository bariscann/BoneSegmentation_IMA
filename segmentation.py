from bone_data import BoneData
from bone_dataset import BoneDataset
from xnet import model as XnetModel
from tensorflow import keras

bone_data = BoneData()
img_size = (160, 160)
num_classes = 2
batch_size = 32
all_gen = BoneDataset(batch_size=batch_size,
                      img_size=img_size,
                      input_img_paths=bone_data.all_data_files,
                      target_img_paths=bone_data.all_label_files)


# Free up RAM in case the model definition cells were run multiple times
keras.backend.clear_session()
xnet_model = XnetModel(input_shape=(160, 160, 3), classes=2)
xnet_model.compile(optimizer="rmsprop", loss="sparse_categorical_crossentropy")
epochs = 2
xnet_model.fit(all_gen, epochs=epochs)